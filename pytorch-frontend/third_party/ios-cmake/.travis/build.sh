#!/bin/bash

mkdir samples/hello-lib/build
cd samples/hello-lib/build
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=../../../toolchain/iOS.cmake \
    -DIOS_PLATFORM=$IOS_PLATFORM \
    || exit 1
make || exit 1
make install || exit 1
