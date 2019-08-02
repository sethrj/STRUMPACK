#!/bin/bash

COMBBLASHOME=/global/homes/p/pghysels/cori/CombBLAS_beta_16_1/
COMBBLASBUILD=/global/homes/p/pghysels/cori/CombBLAS_beta_16_1/build/
PARMETISHOME=$HOME/local/parmetis-4.0.3/
SCOTCHHOME=$HOME/local/cori/scotch_6.0.4/
SLATEHOME=$HOME/local/slate
MAGMAHOME=$HOME/local/magma-2.5.1-alpha1

rm -rf build_GPUON
rm -rf build_GPUOFF
rm -rf install_GPUON
rm -rf install_GPUOFF
mkdir build_GPUON
mkdir build_GPUOFF
mkdir install_GPUON
mkdir install_GPUOFF

#-DCMAKE_EXE_LINKER_FLAGS="-L${OLCF_CUDA_ROOT}/lib64/ -lcublas -lcudart" \
#-DCMAKE_INSTALL_PREFIX=../install_GPU${GPU} \

cd build_GPUON
cmake ../ \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_INSTALL_PREFIX=../install_GPUON \
    -DCMAKE_CXX_COMPILER=mpiCC \
    -DCMAKE_C_COMPILER=mpicc \
    -DCMAKE_Fortran_COMPILER=mpif90 \
    -DSTRUMPACK_USE_OPENMP=ON \
    -DSTRUMPACK_USE_CUDA=ON \
    -DTPL_ENABLE_SLATE=OFF \
    -DTPL_ENABLE_MAGMA=ON \
    -DTPL_SLATE_INCLUDE_DIRS="$SLATEHOME/include/;$SLATEHOME/blaspp/include;$SLATEHOME/lapackpp/include" \
    -DTPL_SLATE_LIBRARIES="$SLATEHOME/lib/libslate.a;$SLATEHOME/blaspp/lib/libblaspp.so;$SLATEHOME/lapackpp/lib/liblapackpp.so" \
    -DTPL_BLAS_LIBRARIES="${OLCF_ESSL_ROOT}/lib64/libessl.so;${OLCF_NETLIB_LAPACK_ROOT}/lib64/libblas.so;${CUDA_DIR}/lib64/libcudart.so;${CUDA_DIR}/lib64/libcublas.so" \
    -DTPL_LAPACK_LIBRARIES="${OLCF_ESSL_ROOT}/lib64/libessl.so;${OLCF_NETLIB_LAPACK_ROOT}/lib64/liblapack.so;${CUDA_DIR}/lib64/libcusolver.so" \
    -DTPL_SCALAPACK_LIBRARIES="${OLCF_NETLIB_SCALAPACK_ROOT}/lib/libscalapack.so" \
    -DTPL_MAGMA_LIBRARIES="$MAGMAHOME/lib/libmagma.so" \
    -DTPL_MAGMA_INCLUDE_DIRS="$MAGMAHOME/include" \
    -DSTRUMPACK_DEV_TESTING=OFF \
    -DSTRUMPACK_BUILD_TESTS=OFF \
    -DSTRUMPACK_C_INTERFACE=OFF \
    -DSTRUMPACK_COUNT_FLOPS=ON \
    -DSTRUMPACK_TASK_TIMERS=OFF \
    -DTPL_METIS_INCLUDE_DIRS=${PARMETISHOME}metis/include \
    -DTPL_METIS_LIBRARIES=${PARMETISHOME}build/Linux-ppc64le/libmetis/libmetis.a \
    -DTPL_ENABLE_PARMETIS=OFF \
    -DTPL_ENABLE_SCOTCH=OFF 
make install VERBOSE=1
cd examples
make testPoisson2d testPoisson2dMPIDist
cd ../../


# cd build_GPUOFF
# cmake ../ \
#     -DCMAKE_BUILD_TYPE=Release \
#     -DCMAKE_INSTALL_PREFIX=../install_GPUOFF \
#     -DCMAKE_CXX_COMPILER=mpiCC \
#     -DCMAKE_C_COMPILER=mpicc \
#     -DCMAKE_Fortran_COMPILER=mpif90 \
#     -DSTRUMPACK_USE_OPENMP=OFF \
#     -DSTRUMPACK_USE_CUDA=OFF \
#     -DTPL_ENABLE_SLATE=OFF \
#     -DTPL_SLATE_INCLUDE_DIRS="$SLATEHOME/include/;$SLATEHOME/blaspp/include;$SLATEHOME/lapackpp/include" \
#     -DTPL_SLATE_LIBRARIES="$SLATEHOME/lib/libslate.a;$SLATEHOME/blaspp/lib/libblaspp.so;$SLATEHOME/lapackpp/lib/liblapackpp.so" \
#     -DTPL_BLAS_LIBRARIES="${OLCF_ESSL_ROOT}/lib64/libessl.so;${OLCF_NETLIB_LAPACK_ROOT}/lib64/libblas.so" \
#     -DTPL_LAPACK_LIBRARIES="${OLCF_ESSL_ROOT}/lib64/libessl.so;${OLCF_NETLIB_LAPACK_ROOT}/lib64/liblapack.so" \
#     -DTPL_SCALAPACK_LIBRARIES="${OLCF_NETLIB_SCALAPACK_ROOT}/lib/libscalapack.so" \
#     -DSTRUMPACK_DEV_TESTING=OFF \
#     -DSTRUMPACK_BUILD_TESTS=OFF \
#     -DSTRUMPACK_C_INTERFACE=OFF \
#     -DSTRUMPACK_COUNT_FLOPS=ON \
#     -DSTRUMPACK_TASK_TIMERS=OFF \
#     -DTPL_METIS_INCLUDE_DIRS=${PARMETISHOME}metis/include \
#     -DTPL_METIS_LIBRARIES=${PARMETISHOME}build/Linux-ppc64le/libmetis/libmetis.a \
#     -DTPL_ENABLE_PARMETIS=OFF \
#     -DTPL_ENABLE_SCOTCH=OFF 
# make install VERBOSE=1
# cd examples
# make testPoisson2d testPoisson2dMPIDist
# cd ../../
