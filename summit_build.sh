#!/bin/bash


COMBBLASHOME=/global/homes/p/pghysels/cori/CombBLAS_beta_16_1/
COMBBLASBUILD=/global/homes/p/pghysels/cori/CombBLAS_beta_16_1/build/
PARMETISHOME=$HOME/local/parmetis-4.0.3/
SCOTCHHOME=$HOME/local/cori/scotch_6.0.4/
SLATEHOME=$HOME/local/slate

for GPU in ON OFF; do

    rm -rf build_GPU${GPU}
    rm -rf install_GPU${GPU}
    mkdir build_GPU${GPU}
    mkdir install_GPU${GPU}
    
    cd build_GPU${GPU}

#-DCMAKE_EXE_LINKER_FLAGS="-L${OLCF_CUDA_ROOT}/lib64/ -lcublas -lcudart" \

    cmake ../ \
	-DCMAKE_BUILD_TYPE=Debug \
	-DCMAKE_INSTALL_PREFIX=../install_GPU${GPU} \
	-DCMAKE_CXX_COMPILER=mpiCC \
	-DCMAKE_C_COMPILER=mpicc \
	-DCMAKE_Fortran_COMPILER=mpif90 \
	-DSTRUMPACK_USE_OPENMP=OFF \
	-DSTRUMPACK_USE_CUDA=$GPU \
    -DTPL_ENABLE_SLATE=$GPU \
    -DTPL_SLATE_INCLUDE_DIRS="$SLATEHOME/include/;$SLATEHOME/blaspp/include;$SLATEHOME/lapackpp/include" \
    -DTPL_SLATE_LIBRARIES="$SLATEHOME/lib/libslate.a;$SLATEHOME/blaspp/lib/libblaspp.so;$SLATEHOME/lapackpp/lib/liblapackpp.so" \
	-DTPL_BLAS_LIBRARIES="${OLCF_ESSL_ROOT}/lib64/libessl.so;${OLCF_NETLIB_LAPACK_ROOT}/lib64/libblas.so;${CUDA_DIR}/lib64/libcudart.so;${CUDA_DIR}/lib64/libcublas.so" \
	-DTPL_LAPACK_LIBRARIES="${OLCF_ESSL_ROOT}/lib64/libessl.so;${OLCF_NETLIB_LAPACK_ROOT}/lib64/liblapack.so" \
	-DTPL_SCALAPACK_LIBRARIES="${OLCF_NETLIB_SCALAPACK_ROOT}/lib/libscalapack.so" \
	-DSTRUMPACK_DEV_TESTING=OFF \
	-DSTRUMPACK_BUILD_TESTS=OFF \
	-DSTRUMPACK_C_INTERFACE=OFF \
	-DSTRUMPACK_COUNT_FLOPS=ON \
	-DSTRUMPACK_TASK_TIMERS=OFF \
	-DTPL_METIS_INCLUDE_DIRS=${PARMETISHOME}metis/include \
	-DTPL_METIS_LIBRARIES=${PARMETISHOME}build/Linux-ppc64le/libmetis/libmetis.a \
	-DTPL_ENABLE_PARMETIS=OFF \
	-DTPL_ENABLE_SCOTCH=OFF 
#	-DCMAKE_LIBRARY_PATH_FLAG="${OLCF_CUDA_ROOT}/lib64/" \
#	-DCMAKE_LINK_LIBRARY_FLAG="-lcublas;-lcudart" \
	
    make VERBOSE=1
    cd ../
done
#make test
#cd examples
#make -k
#cd ../../