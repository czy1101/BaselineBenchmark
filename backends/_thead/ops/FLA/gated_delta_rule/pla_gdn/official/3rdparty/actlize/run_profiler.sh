#!/bin/bash

set -ex
set -o pipefail

current_dir=$(pwd)
echo "Current path is: ${current_dir}"
if [[ ! $current_dir =~ /build$ ]]; then
  if [ -d "./build" ]; then
    cd ./build
    echo "Change to path: ${current_dir}/build"
  else
    echo "ERROR: build directory not found! Please check."
    exit 1
  fi
fi


if [ "$1" == "10000" ]; then
  echo "run profiler start with 10000 SDK"
elif [ "$1" == "10500" ]; then
  echo "run profiler start for 10500 SDK"
else
  echo "Run cmd: ./run.sh ppu_type"
  echo "Unsupported ppu_type! Now support 10000/10500. Please double check."
  exit 1
fi

./tools/profiler/cutlass_profiler --operation=Gemm


