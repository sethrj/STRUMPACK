#!/bin/bash

mkdir build
cd build

export CRAYPE_LINK_TYPE="dynamic"
export PARMETIS_INSTALL="/global/homes/l/liuyangz/Cori/my_software/parmetis-4.0.3"
cmake .. \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_PREFIX=. \
-DCMAKE_CXX_COMPILER=CC \
-DCMAKE_C_COMPILER=cc \
-DSTRUMPACK_C_INTERFACE=OFF \
-DSTRUMPACK_TASK_TIMERS=ON \
-DCMAKE_Fortran_COMPILER=ftn \
-DCMAKE_EXE_LINKER_FLAGS="" \
-DCMAKE_CXX_FLAGS="-std=c++11" \
-DMETIS_INCLUDES=$PARMETIS_INSTALL/metis/include \
-DMETIS_LIBRARIES=$PARMETIS_INSTALL/build/Linux-x86_64/libmetis/libmetis.a \
-DHMATRIX_INCLUDES="/global/homes/l/liuyangz/Cori/my_research/hod-lr-bf_BACA_paperdata/hod-lr-bf" \
-DHMATRIX_LIBRARIES="~/Cori/my_research/STRUMPACK_H_RBF/h_matrix_rbf_randomization/build/SRC/libhmatrix.a;/global/homes/l/liuyangz/Cori/my_research/hod-lr-bf_BACA_paperdata/hod-lr-bf/build/SRC_DOUBLE/libdhodlrbf.a;/global/homes/l/liuyangz/Cori/my_research/hod-lr-bf_BACA_paperdata/hod-lr-bf/build/SRC_DOUBLECOMPLEX/libzhodlrbf.a"
make install VERBOSE=1
cd examples
make
#cd ../test
#make
