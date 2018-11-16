#!/bin/bash

rm -rf build
rm -rf install
mkdir build
mkdir install

cd build

SLATEHOME=/home/pieterg/local/slate
cmake ../ \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=../install \
      -DSTRUMPACK_BUILD_TESTS=OFF \
      -DSTRUMPACK_C_INTERFACE=OFF \
      -DTPL_ENABLE_SLATE=ON \
      -DTPL_SLATE_INCLUDE_DIRS="$SLATEHOME/;$SLATEHOME/blaspp/include;$SLATEHOME/lapackpp/include" \
      -DTPL_SLATE_LIBRARIES=$SLATEHOME/lib/libslate.so \
      -DTPL_METIS_INCLUDE_DIRS=$HOME/local/parmetis-4.0.3/metis/include \
      -DTPL_METIS_LIBRARIES=$HOME/local/parmetis-4.0.3/build/Linux-x86_64/libmetis/libmetis.a

cd build
make install
cd examples
make testPoisson2dMPIDist
OMP_NUM_THREADS=1 mpirun -n 4 /home/pieterg/LBL/STRUMPACK/STRUMPACK/build/examples/testPoisson2dMPIDist 1000
