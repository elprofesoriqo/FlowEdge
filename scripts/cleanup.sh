#!/bin/bash

BASE_DIR=$(pwd)
BUILD_DIR="${BASE_DIR}/build"

echo "Cleaning up build directory: ${BUILD_DIR}"

rm -rf "${BUILD_DIR:?}"