#!/bin/bash

BASE_DIR=$(pwd)

if ! command -v cmake &> /dev/null
then
    echo "Cmake is uninstalled. Please install cmake to proceed."
    exit 1
fi

mkdir -p "${BASE_DIR}/build"
cd "${BASE_DIR}/build" || exit

cmake .. --trace-expand
make