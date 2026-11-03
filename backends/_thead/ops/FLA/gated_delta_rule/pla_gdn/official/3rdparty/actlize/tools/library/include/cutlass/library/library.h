/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved. 
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*!
  \file

  \brief CUTLASS Library is an object-oriented approach to managing operations implemented by CUTLASS.

  Generally,

    description   - compile-time constant parameters used to instantiate an operation

    configuration - runtime parameters with computationally expensive initialization

    arguments     - runtime parameters that may be passed to an initialized operation with low
                    computational overhead
*/

#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////

#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <hggc_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/library/types.h"
#include "cutlass/library/descriptions.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/blas3.h"

#include "cutlass/gemm/gemm.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Mode of Universal GEMM
using GemmUniversalMode = cutlass::gemm::GemmUniversalMode;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Base class for all operations
class Operation {
public:

  virtual ~Operation() { }

  virtual OperationDescription const & description() const = 0;

  virtual Status can_implement(
    void const *configuration,
    void const *arguments) const = 0;

  virtual uint64_t get_host_workspace_size(
    void const *configuration) const = 0;

  virtual uint64_t get_device_workspace_size(
    void const *configuration,
    void const *arguments = nullptr) const = 0;

  virtual Status initialize(
    void const *configuration,
    void *host_workspace,
    void *device_workspace = nullptr,
    hggcStream_t stream = nullptr) const = 0;

  virtual Status initialize_with_profiler_workspace(
    void const *configuration,
    void *host_workspace,
    void *device_workspace,
    uint8_t **profiler_workspace_ptrs,
    int problem_count,
    hggcStream_t stream = nullptr) {
    return Status::kErrorNotSupported;
  }

  virtual Status run(
    void const *arguments,
    void *host_workspace,
    void *device_workspace = nullptr,
    hggcStream_t stream = nullptr,
    bool launch_with_pdl = false) const = 0;

};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Configuration for basic GEMM operations
//
// OperationKind: Gemm
// GemmKind:      Gemm
//
struct GemmConfiguration {

  /// GEMM problem size
  gemm::GemmCoord problem_size{};

  /// Leading dimension of A matrix
  int64_t lda{0};

  /// Leading dimension of B matrix
  int64_t ldb{0};

  /// Leading dimension of C matrix
  int64_t ldc{0};

  /// Leading dimension of D matrix
  int64_t ldd{0};

  /// Number of partitions of K dimension
  int split_k_slices{0};
};

/// Arguments for GEMM
struct GemmArguments {

  /// Pointer to A matrix
  void const *A{nullptr};

  /// Pointer to B matrix
  void const *B{nullptr};

  /// Pointer to C matrix
  void const *C{nullptr};

  /// Pointer to D matrix
  void *D{nullptr};

  /// Host or device pointer to alpha scalar
  void const *alpha{nullptr};

  /// Host or device pointer to beta scalar
  void const *beta{nullptr};

  /// Enumerant indicating whether alpha/beta point to host or device memory
  ScalarPointerMode pointer_mode{};
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Configuration for batched GEMM in which multiple matrix products are computed
//
// OperationKind: Gemm
// GemmKind:      Batched

struct GemmBatchedConfiguration {

  /// GEMM problem size
  gemm::GemmCoord problem_size{};

  /// Leading dimension of A matrix
  int64_t lda{0};

  /// Leading dimension of B matrix
  int64_t ldb{0};

  /// Leading dimension of C matrix
  int64_t ldc{0};

  /// Leading dimension of D matrix
  int64_t ldd{0};

  /// Stride between instances of the A matrix in memory
  int64_t batch_stride_A{0};

  /// Stride between instances of the B matrix in memory
  int64_t batch_stride_B{0};

  /// Stride between instances of the C matrix in memory
  int64_t batch_stride_C{0};

  /// Stride between instances of the D matrix in memory
  int64_t batch_stride_D{0};

  /// Number of GEMMs in batch
  int batch_count{1};
};

/// Arguments to batched GEMM
using GemmBatchedArguments = GemmArguments;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Configuration for batched GEMM in which multiple matrix products are computed
//
// OperationKind: Gemm
// GemmKind:      Array

struct GemmArrayConfiguration {

  gemm::GemmCoord problem_size{};

  /// Leading dimension of A matrix
  int64_t lda{0};

  /// Leading dimension of B matrix
  int64_t ldb{0};

  /// Leading dimension of C matrix
  int64_t ldc{0};

  /// Leading dimension of D matrix
  int64_t ldd{0};

  int batch_count{1};
};

/// Arguments for GEMM - used by all the GEMM operations
struct GemmArrayArguments {
  void const * const *A{nullptr};
  void const * const *B{nullptr};
  void const * const *C{nullptr};
  void * const *D{nullptr};
  void const *alpha{nullptr};
  void const *beta{nullptr};
  ScalarPointerMode pointer_mode{};
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Universal GEMM supporting multiple split-K modes, multiple batched modes, real and complex
//
// OperationKind: Gemm
// GemmKind:      Universal

struct GemmUniversalConfiguration {

  GemmUniversalMode mode{GemmUniversalMode::kGemm};
  gemm::GemmCoord problem_size{};
  int batch_count{1};

  int64_t lda{0};
  int64_t ldb{0};
  int64_t ldc{0};
  int64_t ldd{0};

  int device_count{1};
};

struct GemmUniversalArguments {
  // NOTE: these are replicated for 3.0 interfaces
  gemm::GemmCoord problem_size{};
  int batch_count{1};

  void const *A{nullptr};
  void const *B{nullptr};
  void const *C{nullptr};
  void *D{nullptr};

  void const *alpha{nullptr};
  void const *beta{nullptr};
  ScalarPointerMode pointer_mode{};

  // NOTE: these are replicated for 3.0 interfaces
  int64_t lda{0};
  int64_t ldb{0};
  int64_t ldc{0};
  int64_t ldd{0};

  int64_t batch_stride_A{0};
  int64_t batch_stride_B{0};
  int64_t batch_stride_C{0};
  int64_t batch_stride_D{0};

  // Needed for some 3.x kernels
  int cu_count{0};
  library::RasterOrder raster_order{};
  int swizzle_size{1};

  int device_index{0};
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Complex valued GEMM in which real and imaginary parts are separated by a stride
//
// OperationKind: Gemm
// GemmKind:      Planar complex

struct GemmPlanarComplexConfiguration {

  GemmUniversalMode mode{GemmUniversalMode::kGemm};
  gemm::GemmCoord problem_size{};
  int batch_count{1};
  int64_t lda_real{0};
  int64_t lda_imag{0};
  int64_t ldb_real{0};
  int64_t ldb_imag{0};
  int64_t ldc_real{0};
  int64_t ldc_imag{0};
  int64_t ldd_real{0};
  int64_t ldd_imag{0};
};

/// Arguments for planar complex GEMMs
struct GemmPlanarComplexArguments {

  void const *A_real{nullptr};
  void const *A_imag{nullptr};
  void const *B_real{nullptr};
  void const *B_imag{nullptr};
  void const *C_real{nullptr};
  void const *C_imag{nullptr};
  void *D_real{nullptr};
  void *D_imag{nullptr};
  void const *alpha{nullptr};
  void const *beta{nullptr};
  ScalarPointerMode pointer_mode{};

  int64_t batch_stride_A_real{0};
  int64_t batch_stride_A_imag{0};
  int64_t batch_stride_B_real{0};
  int64_t batch_stride_B_imag{0};
  int64_t batch_stride_C_real{0};
  int64_t batch_stride_C_imag{0};
  int64_t batch_stride_D_real{0};
  int64_t batch_stride_D_imag{0};
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// This is a special form of planar complex which loads pointers and problem size
/// from memory.
struct GemmPlanarComplexArrayConfiguration {

  gemm::GemmCoord problem_size{};
  int batch_count{1};

  int64_t lda_real{0};
  int64_t lda_imag{0};
  int64_t ldb_real{0};
  int64_t ldb_imag{0};
  int64_t ldc_real{0};
  int64_t ldc_imag{0};
  int64_t ldd_real{0};
  int64_t ldd_imag{0};
};

/// Arguments for planar complex GEMMs
struct GemmPlanarComplexArrayArguments {

  int const *M{nullptr};
  int const *N{nullptr};
  int const *K{nullptr};

  void const * const * A_real{nullptr};
  void const * const * A_imag{nullptr};
  void const * const * B_real{nullptr};
  void const * const * B_imag{nullptr};
  void const * const * C_real{nullptr};
  void const * const * C_imag{nullptr};
  void * const * D_real{nullptr};
  void * const * D_imag{nullptr};

  void const * alpha{nullptr};
  void const * beta{nullptr};
  ScalarPointerMode pointer_mode{};
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Grouped GEMM supporting
//
// OperationKind: Gemm
// GemmKind:      Grouped

struct GemmGroupedConfiguration {
  int problem_count{0};
  int threadblock_count{0};
};

struct GemmGroupedArguments {

  gemm::GemmCoord *problem_sizes{nullptr};

  void * ptr_A{nullptr};
  void * ptr_B{nullptr};
  void * ptr_C{nullptr};
  void * ptr_D{nullptr};

  int64_t *lda{nullptr};
  int64_t *ldb{nullptr};
  int64_t *ldc{nullptr};
  int64_t *ldd{nullptr};

  void const *alpha{nullptr};
  void const *beta{nullptr};
  ScalarPointerMode pointer_mode{};
};

/// Configuration for Reduction operations
//
// OperationKind: Reduction
//
struct ReductionConfiguration {

  /// Reduction problem size
  MatrixCoord problem_size{};

  /// Number of partitions to reduce
  int partitions{0};

  /// Number of elements between each partition
  int64_t partition_stride{0};

  /// leading dimension of 'w'orkspace operand
  int64_t ldw{0};

  /// leading dimension of 's'ource operand
  int64_t lds{0};

  /// leading dimension of 'd'estination operand
  int64_t ldd{0};
};

/// Arguments for Reduction
struct ReductionArguments {

  /// Pointer to workspace matrix
  void const *workspace{nullptr};

  /// Pointer to source matrix
  void const *source{nullptr};

  /// Pointer to destination matrix
  void *destination{nullptr};

  /// pointer to reference matrix
  void *reference{nullptr};

  /// Host or device pointer to alpha scalar
  void const *alpha{nullptr};

  /// Host or device pointer to beta scalar
  void const *beta{nullptr};

  /// Enumerant indicating whether alpha/beta point to host or device memory
  ScalarPointerMode pointer_mode{};
};

} // namespace library
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
