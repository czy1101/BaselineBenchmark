# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from backends._enflame.benchmarks._parallax_compare import (
    PARALLEL_CASES as CASES,
    PARALLAX_MARKS,
    parametrize_cases,
    run_parallel_comparison,
)


pytestmark = PARALLAX_MARKS


def pytest_generate_tests(metafunc):
    parametrize_cases(metafunc, CASES)


def test_parallax_torch_benchmark(case, benchmark_options):
    run_parallel_comparison(case, "torch", benchmark_options)
