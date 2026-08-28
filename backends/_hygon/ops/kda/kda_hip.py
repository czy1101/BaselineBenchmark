"""DTK-GEMM chunk KDA candidate.

The intra-chunk and state/output products are expressed as batched matmuls so
the Hygon PyTorch backend can select its DTK GEMM implementation.  The only
serial dimension left on the host is the number of BT16 chunks.
"""

import torch
import torch.nn.functional as F


def _bmm(a, b, gemm_dtype=None):
    if gemm_dtype is not None and a.dtype != gemm_dtype:
        a = a.to(gemm_dtype)
    if gemm_dtype is not None and b.dtype != gemm_dtype:
        b = b.to(gemm_dtype)
    return torch.bmm(a, b).float()


@torch.no_grad()
def chunk_kda_dtk(
    q, k, v, g, beta, *, scale=None, initial_state=None,
    output_final_state=False, state_v_first=False,
    use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
    use_beta_sigmoid_in_kernel=True, safe_gate=True, lower_bound=-5.0,
    A_log=None, dt_bias=None, chunk_size=16, gemm_dtype=None, hip_backend=None,
):
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        raise ValueError("q/k/v must be rank-4 and k must match q")
    B, T, H, K = q.shape
    HV, V = v.shape[2:]
    if K != 128 or V != 128 or H != HV or chunk_size not in (16, 32, 64):
        raise ValueError("DTK KDA requires K=V=128, H=HV, chunk_size in {16,32,64}")
    if g.shape != (B, T, HV, K) or beta.shape != (B, T, HV):
        raise ValueError("invalid g or beta shape")
    if scale is None:
        scale = K ** -0.5
    if gemm_dtype is None:
        gemm_dtype = q.dtype

    qf = q.float()
    kf = k.float()
    if use_qk_l2norm_in_kernel:
        qf = F.normalize(qf, dim=-1, eps=1e-6)
        kf = F.normalize(kf, dim=-1, eps=1e-6)
    if H != HV:
        qf = qf.repeat_interleave(HV // H, dim=2)
        kf = kf.repeat_interleave(HV // H, dim=2)

    gf = g.float()
    if use_gate_in_kernel and A_log is not None:
        al = A_log.reshape(HV, 1).float().to(q.device)
        db = (torch.zeros(HV, K, device=q.device, dtype=torch.float32)
              if dt_bias is None else dt_bias.reshape(HV, K).float())
        x = gf + db.view(1, 1, HV, K)
        a = al.exp().view(1, 1, HV, 1)
        gf = (float(lower_bound) * torch.sigmoid(a * x) if safe_gate
              else -a * F.softplus(x))
    be = torch.sigmoid(beta.float()) if use_beta_sigmoid_in_kernel else beta.float()

    if state_v_first and initial_state is not None:
        initial_state = initial_state.transpose(-1, -2).contiguous()
    state = (torch.zeros(B, HV, K, V, device=q.device, dtype=torch.float32)
             if initial_state is None else initial_state.float().contiguous())

    chunks = (T + chunk_size - 1) // chunk_size
    padded = chunks * chunk_size
    def pad_time(x, value=0.0):
        if x.shape[1] == padded:
            return x
        pad_shape = list(x.shape)
        pad_shape[1] = padded - T
        tail = torch.full(pad_shape, value, device=x.device, dtype=x.dtype)
        return torch.cat((x, tail), dim=1)

    qc = pad_time(qf).reshape(B, chunks, chunk_size, HV, K)
    kc = pad_time(kf).reshape(B, chunks, chunk_size, HV, K)
    gc = pad_time(gf).reshape(B, chunks, chunk_size, HV, K)
    bc = pad_time(be).reshape(B, chunks, chunk_size, HV)
    vc = pad_time(v.float()).reshape(B, chunks, chunk_size, HV, V)
    # Move value-head before chunk so every bmm batch is [B*HV, ...].
    qc = qc.permute(0, 3, 1, 2, 4).reshape(B * HV, chunks, chunk_size, K)
    kc = kc.permute(0, 3, 1, 2, 4).reshape(B * HV, chunks, chunk_size, K)
    gc = gc.permute(0, 3, 1, 2, 4).reshape(B * HV, chunks, chunk_size, K)
    bc = bc.permute(0, 3, 1, 2).reshape(B * HV, chunks, chunk_size)
    vc = vc.permute(0, 3, 1, 2, 4).reshape(B * HV, chunks, chunk_size, V)

    gcum = gc.cumsum(dim=2)
    qg = qc * gcum.exp()
    kneg = kc * (-gcum).exp()
    kpos = kc * gcum.exp()
    # These tensors feed several GEMMs in every chunk.  Cast once and reuse
    # the GEMM-dtype copies instead of repeating FP32->FP16/BF16 conversions
    # inside the recurrent chunk loop.
    qg_gemm = qg.to(gemm_dtype)
    kneg_gemm = kneg.to(gemm_dtype)
    kpos_gemm = kpos.to(gemm_dtype)
    beta_k = bc.unsqueeze(-1)

    # Intra-chunk GEMMs: QG*Kneg^T and Kpos*Kneg^T.
    qg2 = qg_gemm.reshape(B * HV * chunks, chunk_size, K)
    kn2 = kneg_gemm.reshape(B * HV * chunks, chunk_size, K)
    kp2 = kpos_gemm.reshape(B * HV * chunks, chunk_size, K)
    aqk = float(scale) * _bmm(qg2, kn2.transpose(1, 2), gemm_dtype)
    kk = _bmm(kp2, kn2.transpose(1, 2), gemm_dtype)
    aqk_lower = torch.tril(
        torch.ones(chunk_size, chunk_size, device=q.device,
                   dtype=torch.bool), diagonal=0)
    aqk = torch.where(aqk_lower, aqk, torch.zeros_like(aqk))
    b2 = bc.reshape(B * HV * chunks, chunk_size)
    lower = torch.tril(torch.ones(chunk_size, chunk_size,
                                  device=q.device, dtype=torch.bool), diagonal=-1)
    L = kk * b2.unsqueeze(-1)
    L = torch.where(lower, L, torch.zeros_like(L))
    eye = torch.eye(chunk_size, device=q.device, dtype=torch.float32)
    ainv = torch.linalg.solve_triangular(
        eye + L, eye.expand(L.shape[0], -1, -1), upper=False)
    ainv = ainv.reshape(B * HV, chunks, chunk_size, chunk_size)
    aqk = aqk.reshape(B * HV, chunks, chunk_size, chunk_size)

    w = (beta_k * kpos).to(gemm_dtype)
    kg = (kneg * gcum[:, :, -1:, :].exp()).to(gemm_dtype)
    # The FP32 source tensors are no longer needed after their GEMM copies
    # and state operands have been prepared.
    del qg, kneg, kpos
    outputs = []
    state = state.reshape(B * HV, K, V)
    for chunk in range(chunks):
        qgi = qg_gemm[:, chunk]
        wi = w[:, chunk]
        kgi = kg[:, chunk]
        vi = vc[:, chunk]
        betai = bc[:, chunk]
        # State/output products remain separate here.  Keeping the operands
        # contiguous avoids the temporary concatenation cost on DTK.
        state_gemm = state.to(gemm_dtype) if state.dtype != gemm_dtype else state
        w_state = _bmm(wi, state_gemm, gemm_dtype)
        qh = _bmm(qgi, state_gemm, gemm_dtype)
        residual = betai.unsqueeze(-1) * vi - w_state
        solved = _bmm(ainv[:, chunk], residual, None)
        intra = _bmm(aqk[:, chunk], solved, None)
        outputs.append(float(scale) * qh + intra)
        decay = gcum[:, chunk, -1].exp()
        state = state * decay.unsqueeze(-1) + _bmm(
            kgi.transpose(1, 2), solved, gemm_dtype)

    out = torch.stack(outputs, dim=1)
    out = out.reshape(B, HV, chunks, chunk_size, V).permute(0, 2, 3, 1, 4)
    out = out.reshape(B, padded, HV, V)[:, :T].to(v.dtype)
    final_state = state.reshape(B, HV, K, V)
    if state_v_first:
        final_state = final_state.transpose(-1, -2).contiguous()
    return out, final_state if output_final_state else None
