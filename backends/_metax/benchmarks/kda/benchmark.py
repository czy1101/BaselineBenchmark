import torch

from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase
import mcoplib._C  # noqa: F401


class Chunk_kda_fwd_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)

        self.batch = config.get("batch", 1)
        self.seqlen = config.get("seqlen", 256)
        self.heads = config.get("heads", 8)
        self.key_dim = config.get("key_dim", 128)
        self.value_dim = config.get("value_dim", 128)
        self.has_initial_state = config.get(
            "has_initial_state", False
        )
        self.lower_bound = float(
            config.get("lower_bound", -5.0)
        )
        self.scale = float(
            config.get(
                "scale",
                self.key_dim ** -0.5,
            )
        )
        self.seed = int(config.get("seed", 20260814))

        if self.dtype != torch.bfloat16:
            raise ValueError(
                "native chunk_kda_fwd currently requires bfloat16"
            )
        if self.key_dim != 128 or self.value_dim != 128:
            raise ValueError(
                "native chunk_kda_fwd currently requires K=V=128"
            )
        if not -5.0 <= self.lower_bound < 0.0:
            raise ValueError(
                "lower_bound must satisfy -5 <= lower_bound < 0"
            )

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", str(self.dtype))
        state.add_summary(
            "Shape",
            (
                f"(B={self.batch},T={self.seqlen},"
                f"H={self.heads},K={self.key_dim},"
                f"V={self.value_dim})"
            ),
        )
        state.add_summary(
            "initial_state",
            str(self.has_initial_state),
        )

        token_heads = self.batch * self.seqlen * self.heads
        output_elements = token_heads * self.value_dim
        state_elements = (
            self.batch
            * self.heads
            * self.value_dim
            * self.key_dim
        )

        state.add_element_count(output_elements)

        # q/k/v/g + beta + A_log/dt_bias + optional initial state.
        read_bytes = (
            4 * output_elements * 2
            + token_heads * 2
            + (self.heads + self.heads * self.key_dim) * 4
        )
        if self.has_initial_state:
            read_bytes += state_elements * 4

        write_bytes = output_elements * 2 + state_elements * 4

        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

    def _make_inputs(self, dev):
        generator = torch.Generator(device=dev)
        generator.manual_seed(self.seed)

        def random_bf16(shape, magnitude):
            return (
                torch.randn(
                    shape,
                    generator=generator,
                    dtype=self.dtype,
                    device=dev,
                )
                * magnitude
            ).contiguous()

        q = random_bf16(
            (
                self.batch,
                self.seqlen,
                self.heads,
                self.key_dim,
            ),
            0.20,
        )
        k = random_bf16(q.shape, 0.20)
        v = random_bf16(
            (
                self.batch,
                self.seqlen,
                self.heads,
                self.value_dim,
            ),
            0.10,
        )
        g = random_bf16(q.shape, 0.30)
        beta = random_bf16(
            (
                self.batch,
                self.seqlen,
                self.heads,
            ),
            0.50,
        )

        A_log = torch.linspace(
            -0.7,
            0.2,
            self.heads,
            dtype=torch.float32,
            device=dev,
        ).contiguous()

        dt_bias = (
            torch.randn(
                self.heads,
                self.key_dim,
                generator=generator,
                dtype=torch.float32,
                device=dev,
            )
            * 0.10
        ).contiguous()

        initial_state = None
        if self.has_initial_state:
            initial_state = (
                torch.randn(
                    self.batch,
                    self.heads,
                    self.value_dim,
                    self.key_dim,
                    generator=generator,
                    dtype=torch.float32,
                    device=dev,
                )
                * 0.01
            ).contiguous()

        output = torch.empty(
            self.batch,
            self.seqlen,
            self.heads,
            self.value_dim,
            dtype=self.dtype,
            device=dev,
        )
        final_state = torch.empty(
            self.batch,
            self.heads,
            self.value_dim,
            self.key_dim,
            dtype=torch.float32,
            device=dev,
        )

        tensors = (
            q,
            k,
            v,
            g,
            beta,
            A_log,
            dt_bias,
            output,
            final_state,
        )
        assert all(tensor.is_contiguous() for tensor in tensors)
        if initial_state is not None:
            assert initial_state.is_contiguous()

        return (
            q,
            k,
            v,
            g,
            beta,
            A_log,
            dt_bias,
            initial_state,
            output,
            final_state,
        )

    def _call_native(self, inputs):
        (
            q,
            k,
            v,
            g,
            beta,
            A_log,
            dt_bias,
            initial_state,
            output,
            final_state,
        ) = inputs

        torch.ops._C.chunk_kda_fwd(
            output,
            final_state,
            q,
            k,
            v,
            g,
            beta,
            A_log,
            dt_bias,
            initial_state,
            self.scale,
            self.lower_bound,
        )

        return output, final_state

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f"cuda:{dev_id}"
            inputs = self._make_inputs(dev)

            def op_closure():
                self._call_native(inputs)

        return self.make_launcher(dev_id, op_closure)

    def _reference(self, inputs):
        (
            q,
            k,
            v,
            g,
            beta,
            A_log,
            dt_bias,
            initial_state,
            _,
            _,
        ) = inputs

        q = q.float()
        k = k.float()
        v = v.float()
        g = g.float()
        beta = beta.float()

        q = q / torch.linalg.vector_norm(
            q, dim=-1, keepdim=True
        ).clamp_min(1.0e-12)
        k = k / torch.linalg.vector_norm(
            k, dim=-1, keepdim=True
        ).clamp_min(1.0e-12)

        gate = self.lower_bound * torch.sigmoid(
            torch.exp(A_log).view(
                1, 1, self.heads, 1
            )
            * (
                g
                + dt_bias.view(
                    1, 1, self.heads, self.key_dim
                )
            )
        )
        alpha = torch.exp(
            gate.clamp(
                min=self.lower_bound,
                max=0.0,
            )
        )
        beta = torch.sigmoid(beta)

        if initial_state is None:
            state = torch.zeros(
                self.batch,
                self.heads,
                self.key_dim,
                self.value_dim,
                dtype=torch.float32,
                device=q.device,
            )
        else:
            state = (
                initial_state.float()
                .transpose(-1, -2)
                .contiguous()
            )

        output = torch.empty(
            self.batch,
            self.seqlen,
            self.heads,
            self.value_dim,
            dtype=torch.float32,
            device=q.device,
        )

        for token in range(self.seqlen):
            state = state * alpha[:, token].unsqueeze(-1)

            prediction = (
                k[:, token].unsqueeze(-1) * state
            ).sum(dim=-2)

            residual = beta[:, token].unsqueeze(-1) * (
                v[:, token] - prediction
            )

            state = state + (
                k[:, token].unsqueeze(-1)
                * residual.unsqueeze(-2)
            )

            output[:, token] = self.scale * (
                q[:, token].unsqueeze(-1) * state
            ).sum(dim=-2)

        return (
            output.to(self.dtype),
            state.transpose(-1, -2).contiguous(),
        )

    def run_verification(self, dev_id):
        dev = f"cuda:{dev_id}"
        inputs = self._make_inputs(dev)

        native_output, native_state = self._call_native(inputs)
        reference_output, reference_state = self._reference(inputs)

        torch.cuda.synchronize(dev_id)

        output_diff = (
            torch.linalg.vector_norm(
                native_output.float()
                - reference_output.float()
            )
            / torch.linalg.vector_norm(
                reference_output.float()
            ).clamp_min(1.0e-12)
        ).item()

        state_diff = (
            torch.linalg.vector_norm(
                native_state - reference_state
            )
            / torch.linalg.vector_norm(
                reference_state
            ).clamp_min(1.0e-12)
        ).item()

        passed = (
            torch.isfinite(native_output).all().item()
            and torch.isfinite(native_state).all().item()
            and torch.allclose(
                native_output.float(),
                reference_output.float(),
                atol=2.0e-3,
                rtol=2.0e-2,
            )
            and torch.allclose(
                native_state,
                reference_state,
                atol=2.0e-4,
                rtol=5.0e-3,
            )
        )

        return passed, max(output_diff, state_diff)
