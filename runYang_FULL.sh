
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread


ACC=1e-3
LEAF=64
nmpi=64
export EXEC=./build/examples/DenseSchur_HODLR
export DATA=Hsolver/front_2d_5000
export DATADIR=/project/projectdirs/m2957/liuyangz/my_research/matrix/
N=5000
srun -N 32 -n ${nmpi} -c 2 --cpu_bind=cores ${EXEC} ${N} ${DATADIR}${DATA} \
--hss_leaf_size ${LEAF} --hss_rel_tol ${ACC} --hss_compression_algorithm original --hss_d0 128 --hss_max_rank 1024 2>&1 | tee -a full_front2d_5000.out


ACC=1e-3
LEAF=128
nmpi=64
export EXEC=./build/examples/DenseSchur_HODLR
export DATA=Hsolver/front_3d_10000
export DATADIR=/project/projectdirs/m2957/liuyangz/my_research/matrix/
N=10000
srun -N 32 -n ${nmpi} -c 2 --cpu_bind=cores ${EXEC} ${N} ${DATADIR}${DATA} \
--hss_leaf_size ${LEAF} --hss_rel_tol ${ACC} --hss_compression_algorithm original --hss_d0 128 --hss_max_rank 1024 2>&1 | tee -a full_front3d_10000.out


ACC=1e-3
LEAF=128
nmpi=64
export EXEC=./build/examples/DenseBEMAirbus_HODLR
srun -N 32 -n ${nmpi} -c 2 --cpu_bind=cores ${EXEC} ${N} ${DATADIR}${DATA} \
--hss_leaf_size ${LEAF} --hss_rel_tol ${ACC} --hss_compression_algorithm original --hss_d0 128 --hss_max_rank 1024 2>&1 | tee -a full_BEM_sphere10002.out



# ACC=1e-3
# LEAF=128
# nmpi=64
# export DATA=Hsolver/front_3d_90000
# export DATADIR=/project/projectdirs/m2957/liuyangz/my_research/matrix/
# N=90000
# srun -N 32 -n ${nmpi} -c 2 --cpu_bind=cores ${EXEC} ${N} ${DATADIR}${DATA} \
# --hss_leaf_size ${LEAF} --hss_rel_tol ${ACC} --hss_compression_algorithm original --hss_d0 128 --hss_max_rank 1024 2>&1 | tee -a full_front3d_90000.out



# ACC=1e-3
# LEAF=128
# nmpi=64
# export DATA=Hsolver/front_3d_250000
# export DATADIR=/project/projectdirs/m2957/liuyangz/my_research/matrix/
# N=250000
# srun -N 32 -n ${nmpi} -c 2 --cpu_bind=cores ${EXEC} ${N} ${DATADIR}${DATA} \
# --hss_leaf_size ${LEAF} --hss_rel_tol ${ACC} --hss_compression_algorithm original --hss_d0 128 --hss_max_rank 1024 2>&1 | tee -a full_front3d_250000.out





