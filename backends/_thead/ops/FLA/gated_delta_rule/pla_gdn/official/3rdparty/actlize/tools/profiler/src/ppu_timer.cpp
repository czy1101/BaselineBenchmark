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

/* \file
   \brief Defines a math function
*/

#include <stdexcept>
#include <cstring>

#include "cutlass/profiler/ppu_timer.h"


namespace cutlass {
namespace profiler {

/////////////////////////////////////////////////////////////////////////////////////////////////

PpuTimer::PpuTimer() {
  hggcError_t result;

  for (auto & event : events) {
    result = hggcEventCreate(&event);
    if (result != hggcSuccess) {
      throw std::runtime_error("Failed to create device event");
    }
  }
}

PpuTimer::PpuTimer(PpuTimer&& ppu_timer) noexcept {
  memcpy(events, ppu_timer.events, sizeof(events));
  memset(ppu_timer.events, 0, sizeof(ppu_timer.events));
}

PpuTimer::~PpuTimer() {
  for (const auto & event : events) {
    if (event != nullptr) {
      hggcEventDestroy(event);
    }
  }
}

/// Records a start event in the stream, the flag is for hggcEventRecordWithFlags
void PpuTimer::start(hggcStream_t stream, const unsigned int flag) {
  hggcError_t result = hggcEventRecordWithFlags(events[0], stream, flag);
  if (result != hggcSuccess) {
    throw std::runtime_error("Failed to record start event.");
  }
}

/// Records a stop event in the stream, the flag is for hggcEventRecordWithFlags
void PpuTimer::stop(hggcStream_t stream, const unsigned int flag) {
hggcError_t result = hggcEventRecordWithFlags(events[1], stream, flag);
  if (result != hggcSuccess) {
    throw std::runtime_error("Failed to record stop event.");
  }
}

/// Records a stop event in the stream and synchronizes on the stream, the flag is for hggcEventRecordWithFlags
void PpuTimer::stop_and_wait(hggcStream_t stream, const unsigned int flag) {

  stop(stream, flag);

  hggcError_t result;
  if (stream) {
    result = hggcStreamSynchronize(stream);
    if (result != hggcSuccess) {
      throw std::runtime_error("Failed to synchronize with non-null device stream.");
    }
  }
  else {
    result = hggcDeviceSynchronize();
    if (result != hggcSuccess) {
      throw std::runtime_error("Failed to synchronize with device.");
    }
  }
}

/// Returns the duration in milliseconds
double PpuTimer::duration(int iterations) const {

  float avg_ms;

  hggcError_t result = hggcEventElapsedTime(&avg_ms, events[0], events[1]);
  if (result != hggcSuccess) {
    throw std::runtime_error("Failed to query elapsed time from device events.");
  }

  return double(avg_ms) / double(iterations);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace profiler
} // namespace cutlass
