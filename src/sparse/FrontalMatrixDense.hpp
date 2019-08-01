/*
 * STRUMPACK -- STRUctured Matrices PACKage, Copyright (c) 2014, The
 * Regents of the University of California, through Lawrence Berkeley
 * National Laboratory (subject to receipt of any required approvals
 * from the U.S. Dept. of Energy).  All rights reserved.
 *
 * If you have questions about your rights to use or distribute this
 * software, please contact Berkeley Lab's Technology Transfer
 * Department at TTD@lbl.gov.
 *
 * NOTICE. This software is owned by the U.S. Department of Energy. As
 * such, the U.S. Government has been granted for itself and others
 * acting on its behalf a paid-up, nonexclusive, irrevocable,
 * worldwide license in the Software to reproduce, prepare derivative
 * works, and perform publicly and display publicly.  Beginning five
 * (5) years after the date permission to assert copyright is obtained
 * from the U.S. Department of Energy, and subject to any subsequent
 * five (5) year renewals, the U.S. Government is granted for itself
 * and others acting on its behalf a paid-up, nonexclusive,
 * irrevocable, worldwide license in the Software to reproduce,
 * prepare derivative works, distribute copies to the public, perform
 * publicly and display publicly, and to permit others to do so.
 *
 * Developers: Pieter Ghysels, Francois-Henry Rouet, Xiaoye S. Li.
 *             (Lawrence Berkeley National Lab, Computational Research
 *             Division).
 *
 */
#ifndef FRONTAL_MATRIX_DENSE_HPP
#define FRONTAL_MATRIX_DENSE_HPP

#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <random>
#include <assert.h>
#include "misc/TaskTimer.hpp"
#include "dense/BLASLAPACKWrapper.hpp"
#include "CompressedSparseMatrix.hpp"
#include "MatrixReordering.hpp"
#if defined(STRUMPACK_USE_MPI)
#include "ExtendAdd.hpp"
#endif
#if defined(STRUMPACK_USE_CUDA)
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "FrontalMatrixDenseCudaWrapper.h"
#endif

namespace strumpack {

  template<typename scalar_t,typename integer_t> class FrontalMatrixDense;

  template<typename scalar_t, typename integer_t> class LevelInfo {
  public:
    std::vector<FrontalMatrixDense<scalar_t,integer_t>*> f;
    std::size_t nnodes_small = 0;
    std::size_t factor_mem_size = 0;
    std::size_t schur_mem_size = 0;
    std::size_t piv_mem_size = 0;
    int getrf_work_size = 0;

    std::size_t total_work_mem_size = 0;

    std::size_t* dsep = nullptr;
    std::size_t* dupd = nullptr;
    
    scalar_t **pF11 = nullptr;
    scalar_t **pF12 = nullptr;
    scalar_t **pF21 = nullptr;
    scalar_t **pF22 = nullptr;

    int** ppiv = nullptr;
    int** ppiv_small = nullptr;
    
    scalar_t* getrf = nullptr;
    int* getrf_err = nullptr;

    void reserve(std::size_t n) { f.reserve(n); }
  };


  template<typename scalar_t,typename integer_t> class FrontalMatrixDense
    : public FrontalMatrix<scalar_t,integer_t> {
    using F_t = FrontalMatrix<scalar_t,integer_t>;
    using DenseM_t = DenseMatrix<scalar_t>;
    using DenseMW_t = DenseMatrixWrapper<scalar_t>;
    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;
    using LevelInfo_t = LevelInfo<scalar_t,integer_t>;
#if defined(STRUMPACK_USE_MPI)
    using ExtAdd = ExtendAdd<scalar_t,integer_t>;
#endif

  public:
    FrontalMatrixDense
    (integer_t sep, integer_t sep_begin, integer_t sep_end,
     std::vector<integer_t>& upd);

    void release_work_memory() override;
    void extend_add_to_dense
    (DenseM_t& paF11, DenseM_t& paF12, DenseM_t& paF21, DenseM_t& paF22,
     const F_t* p, int task_depth) override;

    void sample_CB
    (const SPOptions<scalar_t>& opts, const DenseM_t& R,
     DenseM_t& Sr, DenseM_t& Sc, F_t* pa, int task_depth) override;
    void sample_CB
    (Trans op, const DenseM_t& R, DenseM_t& S, F_t* pa,
     int task_depth=0) const override;

    void sample_CB_to_F11
    (Trans op, const DenseM_t& R, DenseM_t& S, F_t* pa,
     int task_depth=0) const override;
    void sample_CB_to_F12
    (Trans op, const DenseM_t& R, DenseM_t& S, F_t* pa,
     int task_depth=0) const override;
    void sample_CB_to_F21
    (Trans op, const DenseM_t& R, DenseM_t& S, F_t* pa,
     int task_depth=0) const override;
    void sample_CB_to_F22
    (Trans op, const DenseM_t& R, DenseM_t& S, F_t* pa,
     int task_depth=0) const override;

    void multifrontal_factorization
    (const SpMat_t& A, const SPOptions<scalar_t>& opts,
     int etree_level=0, int task_depth=0) override;

    void forward_multifrontal_solve
    (DenseM_t& b, DenseM_t* work, int etree_level=0,
     int task_depth=0) const override;
    void backward_multifrontal_solve
    (DenseM_t& y, DenseM_t* work, int etree_level=0,
     int task_depth=0) const override;

    void extract_CB_sub_matrix
    (const std::vector<std::size_t>& I, const std::vector<std::size_t>& J,
     DenseM_t& B, int task_depth) const override;

    std::string type() const override { return "FrontalMatrixDense"; }

#if defined(STRUMPACK_USE_CUDA)
    void factorization_by_level_cuda
    (const SpMat_t& A, const SPOptions<scalar_t>& opts);
#endif
    void factorization_by_level
    (const SpMat_t& A, const SPOptions<scalar_t>& opts);
    void setup_level
    (const SpMat_t& A, const SPOptions<scalar_t>& opts,
     LevelInfo_t& info, int etree_level, int l=0);

#if defined(STRUMPACK_USE_MPI)
    void extend_add_copy_to_buffers
    (std::vector<std::vector<scalar_t>>& sbuf,
     const FrontalMatrixMPI<scalar_t,integer_t>* pa) const override {
      ExtAdd::extend_add_seq_copy_to_buffers(F22_, sbuf, pa, this);
    }
#endif

  private:
    std::unique_ptr<scalar_t[], std::function<void(scalar_t*)>> factor_mem_, schur_mem_;
    DenseMW_t F11_, F12_, F21_, F22_;
    std::vector<int> piv; // regular int because it is passed to BLAS

    FrontalMatrixDense(const FrontalMatrixDense&) = delete;
    FrontalMatrixDense& operator=(FrontalMatrixDense const&) = delete;

    void factor_phase1
    (const SpMat_t& A, const SPOptions<scalar_t>& opts,
     int etree_level, int task_depth);
    void factor_phase2
    (const SpMat_t& A, const SPOptions<scalar_t>& opts,
     int etree_level, int task_depth);
    void cuda_factor_phase2
    (const SpMat_t& A, const SPOptions<scalar_t>& opts,
     int etree_level, int task_depth);

    void fwd_solve_phase2
    (DenseM_t& b, DenseM_t& bupd, int etree_level, int task_depth) const;
    void bwd_solve_phase1
    (DenseM_t& y, DenseM_t& yupd, int etree_level, int task_depth) const;

    using F_t::lchild_;
    using FrontalMatrix<scalar_t,integer_t>::rchild_;
    using FrontalMatrix<scalar_t,integer_t>::dim_sep;
    using FrontalMatrix<scalar_t,integer_t>::dim_upd;
  };

  template<typename scalar_t,typename integer_t>
  FrontalMatrixDense<scalar_t,integer_t>::FrontalMatrixDense
  (integer_t sep, integer_t sep_begin, integer_t sep_end,
   std::vector<integer_t>& upd)
    : F_t(nullptr, nullptr, sep, sep_begin, sep_end, upd) {}

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDense<scalar_t,integer_t>::release_work_memory() {
    F22_.clear(); 
    schur_mem_ = nullptr; 
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDense<scalar_t,integer_t>::extend_add_to_dense
  (DenseM_t& paF11, DenseM_t& paF12, DenseM_t& paF21, DenseM_t& paF22,
   const F_t* p, int task_depth) {
    const std::size_t pdsep = paF11.rows();
    const std::size_t dupd = dim_upd();
    std::size_t upd2sep;
    auto I = this->upd_to_parent(p, upd2sep);
#if defined(STRUMPACK_USE_OPENMP_TASKLOOP)
#pragma omp taskloop default(shared) grainsize(64)      \
  if(task_depth < params::task_recursion_cutoff_level)
#endif
    for (std::size_t c=0; c<dupd; c++) {
      auto pc = I[c];
      if (pc < pdsep) {
        for (std::size_t r=0; r<upd2sep; r++)
          paF11(I[r],pc) += F22_(r,c);
        for (std::size_t r=upd2sep; r<dupd; r++)
          paF21(I[r]-pdsep,pc) += F22_(r,c);
      } else {
        for (std::size_t r=0; r<upd2sep; r++)
          paF12(I[r],pc-pdsep) += F22_(r, c);
        for (std::size_t r=upd2sep; r<dupd; r++)
          paF22(I[r]-pdsep,pc-pdsep) += F22_(r,c);
      }
    }
    STRUMPACK_FLOPS((is_complex<scalar_t>()?2:1) * dupd * dupd);
    STRUMPACK_FULL_RANK_FLOPS((is_complex<scalar_t>()?2:1) * dupd * dupd);
    release_work_memory();
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDense<scalar_t,integer_t>::multifrontal_factorization
  (const SpMat_t& A, const SPOptions<scalar_t>& opts,
   int etree_level, int task_depth) {
    // TODO put some logic here to decide when to do the level by
    // level factorization
#if defined(STRUMPACK_USE_CUDA)
    if (opts.use_gpu())
      factorization_by_level_cuda(A, opts);
    else factorization_by_level(A, opts);
    return;
#else
    if (task_depth == 0) {
      // use tasking for children and for extend-add parallelism
#pragma omp parallel if(!omp_in_parallel()) default(shared)
#pragma omp single nowait
      factor_phase1(A, opts, etree_level, task_depth);
      // do not use tasking for blas/lapack parallelism (use system
      // blas threading!)
      factor_phase2(A, opts, etree_level, params::task_recursion_cutoff_level);
    } else {
      factor_phase1(A, opts, etree_level, task_depth);
      factor_phase2(A, opts, etree_level, task_depth);
    }
#endif
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDense<scalar_t,integer_t>::factor_phase1
  (const SpMat_t& A, const SPOptions<scalar_t>& opts,
   int etree_level, int task_depth) {
    if (task_depth < params::task_recursion_cutoff_level) {
      if (lchild_)
#pragma omp task default(shared)                                        \
  final(task_depth >= params::task_recursion_cutoff_level-1) mergeable
        lchild_->multifrontal_factorization
          (A, opts, etree_level+1, task_depth+1);
      if (rchild_)
#pragma omp task default(shared)                                        \
  final(task_depth >= params::task_recursion_cutoff_level-1) mergeable
        rchild_->multifrontal_factorization
          (A, opts, etree_level+1, task_depth+1);
#pragma omp taskwait
    } else {
      if (lchild_)
        lchild_->multifrontal_factorization
          (A, opts, etree_level+1, task_depth);
      if (rchild_)
        rchild_->multifrontal_factorization
          (A, opts, etree_level+1, task_depth);
    }
    // TODO can we allocate the memory in one go??
    const std::size_t dsep = dim_sep();
    const std::size_t dupd = dim_upd();
    factor_mem_ = std::unique_ptr<scalar_t[], std::function<void(scalar_t*)>>
      (new scalar_t[dsep*(dsep+2*dupd)], [](scalar_t* ptr) { delete[] ptr; });
    // F11_ = DenseM_t(dsep, dsep); F11_.zero();
    // F12_ = DenseM_t(dsep, dupd); F12_.zero();
    // F21_ = DenseM_t(dupd, dsep); F21_.zero();
    auto fm = factor_mem_.get();
    F11_ = DenseMW_t(dsep, dsep, fm, dsep); fm += dsep*dsep; F11_.zero();
    F12_ = DenseMW_t(dsep, dupd, fm, dsep); fm += dsep*dupd; F12_.zero();
    F21_ = DenseMW_t(dupd, dsep, fm, dupd); fm += dupd*dsep; F21_.zero();
    A.extract_front
      (F11_, F12_, F21_, this->sep_begin_, this->sep_end_,
       this->upd_, task_depth);
    if (dupd) {
      schur_mem_ = std::unique_ptr<scalar_t[], std::function<void(scalar_t*)>>
        (new scalar_t[dupd*dupd], [](scalar_t* ptr) { delete[] ptr; });
      //F22_ = DenseM_t(dupd, dupd);
      F22_ = DenseMW_t(dupd, dupd, schur_mem_.get(), dupd);
      F22_.zero();
    }
    if (lchild_)
      lchild_->extend_add_to_dense
        (F11_, F12_, F21_, F22_, this, task_depth);
    if (rchild_)
      rchild_->extend_add_to_dense
        (F11_, F12_, F21_, F22_, this, task_depth);
  }

  
// #if defined(STRUMPACK_USE_CUDA)
//   template<typename scalar_t,typename integer_t> void
//   FrontalMatrixDense<scalar_t,integer_t>::cuda_factor_phase2
//   (const SpMat_t& A, const SPOptions<scalar_t>& opts,
//    int etree_level, int task_depth) {
//       cublasHandle_t schur_handle;
//       cusolverDnHandle_t F11_handle;
//       cudaEvent_t LU_event;
      
//       cusolverStatus_t LU_stat;
//       cudaError_t cudaStat;

//       cublasStatus_t stat; 

//       stat = cublasCreate(&schur_handle);
//       cudaEventCreate(&LU_event);

//       cudaStream_t F11_stream;
//       cudaStream_t schur_stream;
//       cudaError_t stream_result;

//       stream_result = cudaStreamCreate(&F11_stream);
//       stream_result = cudaStreamCreate(&schur_stream);

//       double* F11_work;
//       int F11_worksize;
//       int *F11_piv, minsize, *F11_err;
      
//       cusolverDnCreate(&F11_handle);
//       cusolverDnSetStream(F11_handle, F11_stream);
      
//       if (F11_.rows() >= F11_.cols()) {
//         minsize = F11_.cols();
//       } else {
//         minsize = F11_.rows();
//       }

//       std::size_t size = F12_.rows()*F12_.cols() +
//                          F11_.rows()*F11_.cols() + 
//                          F21_.rows()*F21_.cols() +
//                          F22_.rows()*F22_.cols();
      
//       static std::size_t total_size_buff_host_bytes = 0;
//       static double *buff_host = nullptr;
//       std::size_t new_total_size_buff_host_bytes = size*sizeof(scalar_t) + minsize*sizeof(int);
       
//       if (new_total_size_buff_host_bytes > total_size_buff_host_bytes) {
//         total_size_buff_host_bytes = new_total_size_buff_host_bytes;
//         if(buff_host) cudaFree(buff_host);
//         cudaError_t buff_host_err = cudaMallocHost((void **)&buff_host, new_total_size_buff_host_bytes);
//         assert(buff_host_err == cudaSuccess);
//       }

//       double* h_F11_ = buff_host;
//       double* h_F12_ = h_F11_ + F11_.rows()*F11_.cols();
//       double* h_F21_ = h_F12_ + F12_.rows()*F12_.cols();
//       double* h_F22_ = h_F21_ + F21_.rows()*F21_.cols();
//       int* h_F11_piv = (int *)(h_F22_ + F22_.rows()*F22_.cols());


//       cudaError_t memcpy1 = cudaMemcpy(h_F11_, F11_.data(), F11_.rows()*F11_.cols()*sizeof(scalar_t), cudaMemcpyHostToHost);
//       cudaError_t memcpy2 = cudaMemcpy(h_F12_, F12_.data(), F12_.rows()*F12_.cols()*sizeof(scalar_t), cudaMemcpyHostToHost);
//       cudaError_t memcpy3 = cudaMemcpy(h_F21_, F21_.data(), F21_.rows()*F21_.cols()*sizeof(scalar_t), cudaMemcpyHostToHost);
//       cudaError_t memcpy4 = cudaMemcpy(h_F22_, F22_.data(), F22_.rows()*F22_.cols()*sizeof(scalar_t), cudaMemcpyHostToHost);
//       cudaError_t memset = cudaMemset(h_F11_piv, 0, minsize*sizeof(int));

//       assert(memcpy1 == cudaSuccess);
//       assert(memcpy2 == cudaSuccess);
//       assert(memcpy3 == cudaSuccess);
//       assert(memcpy4 == cudaSuccess);
//       assert(memset == cudaSuccess);

//       static std::size_t total_size_buff_dev_bytes = 0;
//       static double *buff_dev = nullptr;
//       std::size_t new_total_size_buff_dev_bytes = size*sizeof(scalar_t) + minsize*sizeof(int);

//       if (new_total_size_buff_dev_bytes > total_size_buff_dev_bytes) {
//         total_size_buff_dev_bytes = new_total_size_buff_dev_bytes;
//         if(buff_dev) cudaFree(buff_dev);
//         cudaError_t buff_dev_err = cudaMalloc((void **)&buff_dev, new_total_size_buff_dev_bytes);
//         assert(buff_dev_err == cudaSuccess);
//       }
      
//       double* d_F11_ = buff_dev;
//       double* d_F12_ = d_F11_ + F11_.rows()*F11_.cols();
//       double* d_F21_ = d_F12_ + F12_.rows()*F12_.cols();
//       double* d_F22_ = d_F21_ + F21_.rows()*F21_.cols();

//       F11_piv =(int *)(d_F22_ + F22_.rows()*F22_.cols());
//       F11_err = F11_piv + minsize*sizeof(int);

//       cublasStatus_t stat1 = cublasSetMatrixAsync(F11_.rows(), F11_.cols(),sizeof(scalar_t), h_F11_, F11_.ld(), d_F11_, F11_.ld(), F11_stream);
//       cublasStatus_t stat2 = cublasSetMatrixAsync(F12_.rows(), F12_.cols(),sizeof(scalar_t), h_F12_, F12_.ld(), d_F12_, F12_.ld(), schur_stream);
//       cublasStatus_t stat3 = cublasSetMatrixAsync(F21_.rows(), F21_.cols(),sizeof(scalar_t), h_F21_, F21_.ld(), d_F21_, F21_.ld(), schur_stream);
//       cublasStatus_t stat4 = cublasSetMatrixAsync(F22_.rows(), F22_.cols(),sizeof(scalar_t), h_F22_, F22_.ld(), d_F22_, F22_.ld(), schur_stream);

//       assert(stat1 == CUBLAS_STATUS_SUCCESS);
//       assert(stat2 == CUBLAS_STATUS_SUCCESS);
//       assert(stat3 == CUBLAS_STATUS_SUCCESS);
//       assert(stat4 == CUBLAS_STATUS_SUCCESS);

//       static std::size_t total_size_workspace = 0;
//       static double *workspace = nullptr;
//       cusolverDnDgetrf_bufferSize(F11_handle, F11_.rows(), F11_.cols(), d_F11_, F11_.ld(), &F11_worksize);
//       std::size_t new_total_size_workspace = F11_worksize;

//       if (new_total_size_workspace > total_size_workspace) {
//         total_size_workspace = new_total_size_workspace;
//         if (workspace) cudaFree(workspace);
//         cudaError_t workspace_err = cudaMalloc((void **)&workspace, new_total_size_workspace*sizeof(double));
//         assert(workspace_err == cudaSuccess); 
//       }

//       if (stat != CUBLAS_STATUS_SUCCESS) {
//         if (stat == CUBLAS_STATUS_NOT_INITIALIZED) {
//           std::cout << "Cuda Runtime Initialization failed" << std::endl;
//         }
//         if (stat == CUBLAS_STATUS_ALLOC_FAILED) {
//           std::cout << "Resources could not be allocated" << std::endl;
//         }
//       }
         
//       if (dim_sep()) {
         
//         LU_stat = cusolverDnDgetrf(F11_handle, F11_.rows(), F11_.cols(), d_F11_, F11_.ld(), workspace, F11_piv, F11_err);
//         assert(LU_stat == CUSOLVER_STATUS_SUCCESS);
//         cudaError_t eventerr1 = cudaEventRecord(LU_event, F11_stream);
//         assert(eventerr1 == cudaSuccess);

//         // TODO async?
//         stat1 = cublasGetMatrix(F11_.rows(), F11_.cols(), sizeof(scalar_t), d_F11_, F11_.ld(), h_F11_, F11_.ld());
//         assert(stat1 == CUBLAS_STATUS_SUCCESS);

//         cublasGetVector(minsize, sizeof(int), F11_piv, 1, h_F11_piv, 1); 
//         memcpy1 = cudaMemcpy(F11_.data(),h_F11_, F11_.rows()*F11_.cols()*sizeof(scalar_t), cudaMemcpyHostToHost);
//         assert(memcpy1 == cudaSuccess);

//         piv.resize(minsize);
//         cudaMemcpy(piv.data(), h_F11_piv, minsize*sizeof(int), cudaMemcpyHostToHost);

//  //       if (opts.replace_tiny_pivots()) {
//  //         // TODO consider other values for thresh
//  //         //  - sqrt(eps)*|A|_1 as in SuperLU ?
//  //         auto thresh = blas::lamch<real_t>('E') * A.size();
//  //         for (std::size_t i=0; i<F11_.rows(); i++)
//  //           if (std::abs(F11_(i,i)) < thresh)
//  //             F11_(i,i) = (std::real(F11_(i,i)) < 0) ? -thresh : thresh;
//  //       }
//         if (dim_upd()) {
//           cudaError_t eventerr2 = cudaStreamWaitEvent(schur_stream, LU_event, 0); 
//           assert(eventerr2 == cudaSuccess);

//           cudaStreamSynchronize(schur_stream);
//           cudaStreamSynchronize(F11_stream);
//           cusolverStatus_t getrs_err = cusolverDnDgetrs(F11_handle, CUBLAS_OP_N, F11_.rows(), F12_.cols(), d_F11_, F11_.ld(), F11_piv, d_F12_, F12_.ld(), F11_err); 
//           assert(getrs_err == CUSOLVER_STATUS_SUCCESS);
//           double alpha = -1.;
//           double beta = 1.;

//           stat = cublasDgemm(schur_handle, CUBLAS_OP_N, CUBLAS_OP_N, 
//                              F22_.rows(), F22_.cols(), F12_.rows(), 
//                              &alpha, d_F21_, F21_.ld(), 
//                              d_F12_, F12_.ld(), &beta, 
//                              d_F22_, F22_.ld());

// 	  // TODO async? in separate streams?
//           stat2 = cublasGetMatrix(F12_.rows(), F12_.cols(), sizeof(scalar_t), d_F12_, F12_.ld(), h_F12_, F12_.ld());
//           stat3 = cublasGetMatrix(F21_.rows(), F21_.cols(), sizeof(scalar_t), d_F21_, F21_.ld(), h_F21_, F21_.ld());
//           stat4 = cublasGetMatrix(F22_.rows(), F22_.cols(), sizeof(scalar_t), d_F22_, F22_.ld(), h_F22_, F22_.ld());



//           assert(stat2 == CUBLAS_STATUS_SUCCESS);
//           assert(stat3 == CUBLAS_STATUS_SUCCESS);
//           assert(stat4 == CUBLAS_STATUS_SUCCESS);


//           memcpy2 = cudaMemcpy(F12_.data(),h_F12_, F12_.rows()*F12_.cols()*sizeof(scalar_t), cudaMemcpyHostToHost);
//           memcpy3 = cudaMemcpy(F21_.data(),h_F21_, F21_.rows()*F21_.cols()*sizeof(scalar_t), cudaMemcpyHostToHost);
//           memcpy4 = cudaMemcpy(F22_.data(),h_F22_, F22_.rows()*F22_.cols()*sizeof(scalar_t), cudaMemcpyHostToHost);

//           assert(memcpy2 == cudaSuccess);
//           assert(memcpy3 == cudaSuccess);
//           assert(memcpy4 == cudaSuccess);

//         }
//     }
//     STRUMPACK_FULL_RANK_FLOPS
//       (LU_flops(F11_) +
//        gemm_flops(Trans::N, Trans::N, scalar_t(-1.), F21_, F12_, scalar_t(1.)) +
//        trsm_flops(Side::L, scalar_t(1.), F11_, F12_) +
//        trsm_flops(Side::R, scalar_t(1.), F11_, F21_));

//       cudaEventDestroy(LU_event);
//       stream_result = cudaStreamDestroy(F11_stream);
//       stream_result = cudaStreamDestroy(schur_stream);
//   }
// #endif

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDense<scalar_t,integer_t>::factor_phase2
  (const SpMat_t& A, const SPOptions<scalar_t>& opts,
   int etree_level, int task_depth) {
    if (dim_sep()) {
      piv = F11_.LU(task_depth);
      if (opts.replace_tiny_pivots()) {
        // TODO consider other values for thresh
        //  - sqrt(eps)*|A|_1 as in SuperLU ?
        auto thresh = blas::lamch<real_t>('E') * A.size();
        for (std::size_t i=0; i<F11_.rows(); i++)
          if (std::abs(F11_(i,i)) < thresh)
            F11_(i,i) = (std::real(F11_(i,i)) < 0) ? -thresh : thresh;
      }
      if (dim_upd()) {
        F11_.solve_LU_in_place(F12_, piv, 0);
        // F12_.laswp(piv, true);
        // trsm(Side::L, UpLo::L, Trans::N, Diag::U,
        //      scalar_t(1.), F11_, F12_, task_depth);
        // trsm(Side::R, UpLo::U, Trans::N, Diag::N,
        //      scalar_t(1.), F11_, F21_, task_depth);
        gemm(Trans::N, Trans::N, scalar_t(-1.), F21_, F12_,
             scalar_t(1.), F22_, task_depth);
      }
    }
    
    //std::cout << "F11 NORM IS = " << F11_.norm() << std::endl;
    STRUMPACK_FULL_RANK_FLOPS
      (LU_flops(F11_) +
       gemm_flops(Trans::N, Trans::N, scalar_t(-1.), F21_, F12_, scalar_t(1.)) +
       trsm_flops(Side::L, scalar_t(1.), F11_, F12_) +
       trsm_flops(Side::R, scalar_t(1.), F11_, F21_));
  }
  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDense<scalar_t,integer_t>::forward_multifrontal_solve
  (DenseM_t& b, DenseM_t* work, int etree_level, int task_depth) const {
    DenseMW_t bupd(dim_upd(), b.cols(), work[0], 0, 0);
    bupd.zero();
    if (task_depth == 0) {
      // tasking when calling the children
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single nowait
      this->fwd_solve_phase1(b, bupd, work, etree_level, task_depth);
      // no tasking for the root node computations, use system blas threading!
      fwd_solve_phase2(b, bupd, etree_level, params::task_recursion_cutoff_level);
    } else {
      this->fwd_solve_phase1(b, bupd, work, etree_level, task_depth);
      fwd_solve_phase2(b, bupd, etree_level, task_depth);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDense<scalar_t,integer_t>::fwd_solve_phase2
  (DenseM_t& b, DenseM_t& bupd, int etree_level, int task_depth) const {
    if (dim_sep()) {
      DenseMW_t bloc(dim_sep(), b.cols(), b, this->sep_begin_, 0);
      F11_.solve_LU_in_place(bloc, piv, task_depth);
      if (dim_upd())
        gemm(Trans::N, Trans::N, scalar_t(-1.), F21_, bloc,
             scalar_t(1.), bupd, task_depth);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDense<scalar_t,integer_t>::backward_multifrontal_solve
  (DenseM_t& y, DenseM_t* work, int etree_level, int task_depth) const {
    DenseMW_t yupd(dim_upd(), y.cols(), work[0], 0, 0);
    if (task_depth == 0) {
      // no tasking in blas routines, use system threaded blas instead
      bwd_solve_phase1
        (y, yupd, etree_level, params::task_recursion_cutoff_level);
#pragma omp parallel if(!omp_in_parallel())
#pragma omp single nowait
      // tasking when calling children
      this->bwd_solve_phase2(y, yupd, work, etree_level, task_depth);
    } else {
      bwd_solve_phase1(y, yupd, etree_level, task_depth);
      this->bwd_solve_phase2(y, yupd, work, etree_level, task_depth);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDense<scalar_t,integer_t>::bwd_solve_phase1
  (DenseM_t& y, DenseM_t& yupd, int etree_level, int task_depth) const {
    if (dim_sep()) {
      DenseMW_t yloc(dim_sep(), y.cols(), y, this->sep_begin_, 0);
      gemm(Trans::N, Trans::N, scalar_t(-1.), F12_, yupd,
           scalar_t(1.), yloc, task_depth);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDense<scalar_t,integer_t>::extract_CB_sub_matrix
  (const std::vector<std::size_t>& I, const std::vector<std::size_t>& J,
   DenseM_t& B, int task_depth) const {
    std::vector<std::size_t> lJ, oJ;
    this->find_upd_indices(J, lJ, oJ);
    if (lJ.empty()) return;
    std::vector<std::size_t> lI, oI;
    this->find_upd_indices(I, lI, oI);
    if (lI.empty()) return;
    for (std::size_t j=0; j<lJ.size(); j++)
      for (std::size_t i=0; i<lI.size(); i++)
        B(oI[i], oJ[j]) += F22_(lI[i], lJ[j]);
    STRUMPACK_FLOPS((is_complex<scalar_t>() ? 2 : 1) * lJ.size() * lI.size());
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDense<scalar_t,integer_t>::sample_CB
  (const SPOptions<scalar_t>& opts, const DenseM_t& R, DenseM_t& Sr,
   DenseM_t& Sc, F_t* pa, int task_depth) {
    auto I = this->upd_to_parent(pa);
    auto cR = R.extract_rows(I);
    DenseM_t cS(dim_upd(), R.cols());
    TIMER_TIME(TaskType::F22_MULT, 1, t_f22mult);
    gemm(Trans::N, Trans::N, scalar_t(1.), F22_, cR,
         scalar_t(0.), cS, task_depth);
    TIMER_STOP(t_f22mult);
    Sr.scatter_rows_add(I, cS, task_depth);
    TIMER_TIME(TaskType::F22_MULT, 1, t_f22mult2);
    gemm(Trans::C, Trans::N, scalar_t(1.), F22_, cR,
         scalar_t(0.), cS, task_depth);
    TIMER_STOP(t_f22mult2);
    Sc.scatter_rows_add(I, cS, task_depth);
    STRUMPACK_CB_SAMPLE_FLOPS
      (gemm_flops(Trans::N, Trans::N, scalar_t(1.), F22_, cR, scalar_t(0.)) +
       gemm_flops(Trans::C, Trans::N, scalar_t(1.), F22_, cR, scalar_t(0.)) +
       cS.rows()*cS.cols()*2); // for the skinny-extend add
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDense<scalar_t,integer_t>::sample_CB
  (Trans op, const DenseM_t& R, DenseM_t& S, F_t* pa, int task_depth) const {
    auto I = this->upd_to_parent(pa);
    auto cR = R.extract_rows(I);
    DenseM_t cS(dim_upd(), R.cols());
    TIMER_TIME(TaskType::F22_MULT, 1, t_f22mult);
    gemm(op, Trans::N, scalar_t(1.), F22_, cR,
         scalar_t(0.), cS, task_depth);
    TIMER_STOP(t_f22mult);
    S.scatter_rows_add(I, cS, task_depth);
    STRUMPACK_CB_SAMPLE_FLOPS
      (gemm_flops(op, Trans::N, scalar_t(1.), F22_, cR, scalar_t(0.)) +
       cS.rows()*cS.cols()); // for the skinny-extend add
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDense<scalar_t,integer_t>::sample_CB_to_F11
  (Trans op, const DenseM_t& R, DenseM_t& S, F_t* pa, int task_depth) const {
    const std::size_t dupd = dim_upd();
    if (!dupd) return;
    std::size_t u2s;
    auto Ir = this->upd_to_parent(pa, u2s);
    auto Rcols = R.cols();
    DenseM_t cR(u2s, Rcols);
    for (std::size_t c=0; c<Rcols; c++)
      for (std::size_t r=0; r<u2s; r++)
        cR(r,c) = R(Ir[r],c);
    DenseM_t cS(u2s, Rcols);
    DenseMW_t CB11(u2s, u2s, const_cast<DenseMW_t&>(F22_), 0, 0);
    gemm(op, Trans::N, scalar_t(1.), CB11, cR, scalar_t(0.), cS, task_depth);
    for (std::size_t c=0; c<Rcols; c++)
      for (std::size_t r=0; r<u2s; r++)
        S(Ir[r],c) += cS(r,c);
    STRUMPACK_CB_SAMPLE_FLOPS(u2s*Rcols);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDense<scalar_t,integer_t>::sample_CB_to_F12
  (Trans op, const DenseM_t& R, DenseM_t& S, F_t* pa, int task_depth) const {
    const std::size_t dupd = dim_upd();
    if (!dupd) return;
    std::size_t u2s;
    auto Ir = this->upd_to_parent(pa, u2s);
    auto pds = pa->dim_sep();
    auto Rcols = R.cols();
    DenseMW_t CB12(u2s, dupd-u2s, const_cast<DenseMW_t&>(F22_), 0, u2s);
    if (op == Trans::N) {
      DenseM_t cR(dupd-u2s, Rcols);
      for (std::size_t c=0; c<Rcols; c++)
        for (std::size_t r=u2s; r<dupd; r++)
          cR(r-u2s,c) = R(Ir[r]-pds,c);
      DenseM_t cS(u2s, Rcols);
      gemm(op, Trans::N, scalar_t(1.), CB12, cR, scalar_t(0.), cS, task_depth);
      for (std::size_t c=0; c<Rcols; c++)
        for (std::size_t r=0; r<u2s; r++)
          S(Ir[r],c) += cS(r,c);
      STRUMPACK_CB_SAMPLE_FLOPS(u2s*Rcols);
    } else {
      DenseM_t cR(u2s, Rcols);
      for (std::size_t c=0; c<Rcols; c++)
        for (std::size_t r=0; r<u2s; r++)
          cR(r,c) = R(Ir[r],c);
      DenseM_t cS(dupd-u2s, Rcols);
      gemm(op, Trans::N, scalar_t(1.), CB12, cR, scalar_t(0.), cS, task_depth);
      for (std::size_t c=0; c<Rcols; c++)
        for (std::size_t r=u2s; r<dupd; r++)
          S(Ir[r]-pds,c) += cS(r-u2s,c);
      STRUMPACK_CB_SAMPLE_FLOPS((dupd-u2s)*Rcols);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDense<scalar_t,integer_t>::sample_CB_to_F21
  (Trans op, const DenseM_t& R, DenseM_t& S, F_t* pa, int task_depth) const {
    const std::size_t dupd = dim_upd();
    if (!dupd) return;
    std::size_t u2s;
    auto Ir = this->upd_to_parent(pa, u2s);
    auto Rcols = R.cols();
    auto pds = pa->dim_sep();
    DenseMW_t CB21(dupd-u2s, u2s, const_cast<DenseMW_t&>(F22_), u2s, 0);
    if (op == Trans::N) {
      DenseM_t cR(u2s, Rcols);
      for (std::size_t c=0; c<Rcols; c++)
        for (std::size_t r=0; r<u2s; r++)
          cR(r,c) = R(Ir[r],c);
      DenseM_t cS(dupd-u2s, Rcols);
      gemm(op, Trans::N, scalar_t(1.), CB21, cR, scalar_t(0.), cS, task_depth);
      for (std::size_t c=0; c<Rcols; c++)
        for (std::size_t r=u2s; r<dupd; r++)
          S(Ir[r]-pds,c) += cS(r-u2s,c);
      STRUMPACK_CB_SAMPLE_FLOPS((dupd-u2s)*Rcols);
    } else {
      DenseM_t cR(dupd-u2s, Rcols);
      for (std::size_t c=0; c<Rcols; c++)
        for (std::size_t r=u2s; r<dupd; r++)
          cR(r-u2s,c) = R(Ir[r]-pds,c);
      DenseM_t cS(u2s, Rcols);
      gemm(op, Trans::N, scalar_t(1.), CB21, cR, scalar_t(0.), cS, task_depth);
      for (std::size_t c=0; c<Rcols; c++)
        for (std::size_t r=0; r<u2s; r++)
          S(Ir[r],c) += cS(r,c);
      STRUMPACK_CB_SAMPLE_FLOPS(u2s*Rcols);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDense<scalar_t,integer_t>::sample_CB_to_F22
  (Trans op, const DenseM_t& R, DenseM_t& S, F_t* pa, int task_depth) const {
    const std::size_t dupd = dim_upd();
    if (!dupd) return;
    std::size_t u2s;
    auto Ir = this->upd_to_parent(pa, u2s);
    auto pds = pa->dim_sep();
    auto Rcols = R.cols();
    DenseM_t cR(dupd-u2s, Rcols);
    for (std::size_t c=0; c<Rcols; c++)
      for (std::size_t r=u2s; r<dupd; r++)
        cR(r-u2s,c) = R(Ir[r]-pds,c);
    DenseM_t cS(dupd-u2s, Rcols);
    DenseMW_t CB22(dupd-u2s, dupd-u2s, const_cast<DenseMW_t&>(F22_), u2s, u2s);
    gemm(op, Trans::N, scalar_t(1.), CB22, cR, scalar_t(0.), cS, task_depth);
    for (std::size_t c=0; c<Rcols; c++)
      for (std::size_t r=u2s; r<dupd; r++)
        S(Ir[r]-pds,c) += cS(r-u2s,c);
    STRUMPACK_CB_SAMPLE_FLOPS((dupd-u2s)*Rcols);
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDense<scalar_t,integer_t>::setup_level
  (const SpMat_t& A, const SPOptions<scalar_t>& opts,
   LevelInfo_t& ldata, int etree_level, int l) {
    using FD_t = FrontalMatrixDense<scalar_t,integer_t>;
    if (l < etree_level) {
      if (lchild_) // child of dense should be dense!
        dynamic_cast<FD_t*>(lchild_.get())->
          setup_level(A, opts, ldata, etree_level, l+1);
      if (rchild_)
        dynamic_cast<FD_t*>(rchild_.get())->
          setup_level(A, opts, ldata, etree_level, l+1);
    } else if (etree_level == l) ldata.f.push_back(this);
  }

#if defined(STRUMPACK_USE_CUDA)
  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDense<scalar_t,integer_t>::factorization_by_level_cuda
  (const SpMat_t& A, const SPOptions<scalar_t>& opts) {
    int cutoff_size = opts.cuda_cutoff();
    using uniq_scalar_t = std::unique_ptr<scalar_t[],std::function<void(scalar_t*)>>;
    auto cuda_deleter = [](void* ptr) { cudaFree(ptr); };

    int device_id;
    cudaGetDevice(&device_id);
    const int max_streams = opts.cuda_streams();
    std::vector<cudaStream_t> streams(max_streams);
    std::vector<cublasHandle_t> blas_handle(max_streams);
    std::vector<cusolverDnHandle_t> solver_handle(max_streams);
    for (int i=0; i<max_streams; i++) {
      cudaStreamCreate(&streams[i]);
      cublasCreate(&blas_handle[i]);
      cusolverDnCreate(&solver_handle[i]);
      cusolverDnSetStream(solver_handle[i], streams[i]);
    }
    int lvls = this->levels();
    std::vector<LevelInfo_t> ldata(lvls);
    std::size_t max_level_work_mem_size = 0;
    for (int lvl=0; lvl<lvls; lvl++) {
      int l = lvls - lvl - 1;
      setup_level(A, opts, ldata[l], l);
      auto nnodes = ldata[l].f.size();
      std::size_t max_dsep = 0, node_max_dsep = 0;
      for (std::size_t n=0; n<nnodes; n++) {
        auto& f = *(ldata[l].f[n]);
        const auto dsep = f.dim_sep();
        const auto dupd = f.dim_upd();
        const auto size = dsep + dupd;
	ldata[l].factor_mem_size += dsep*dsep + 2*dsep*dupd;
        ldata[l].schur_mem_size += dupd*dupd;
	ldata[l].piv_mem_size += dsep;
        if (size < cutoff_size)
	  ldata[l].nnodes_small++;
        if (dsep > max_dsep) {
	  node_max_dsep = n;
	  max_dsep = dsep;
        }
      }
      cuda::cusolverDngetrf_bufferSize
	(solver_handle[0], max_dsep, max_dsep, 
	 /*f.F11_.data()*/ static_cast<scalar_t*>(nullptr),
	 max_dsep, &(ldata[l].getrf_work_size));
      ldata[l].total_work_mem_size = 
	sizeof(scalar_t) * (ldata[l].schur_mem_size + 
			    max_streams * ldata[l].getrf_work_size) +
	sizeof(int) * (ldata[l].piv_mem_size) +
	ldata[l].nnodes_small * 
	// pointers to F11, F12, F21, F22, piv, dsep, dupd for all small fronts
	(4*sizeof(scalar_t*) + sizeof(int*) + 2*sizeof(std::size_t)) +
	// pointers to all piv vectors
	nnodes*sizeof(scalar_t*) +
	// getrf error code
	sizeof(int) * max_streams;
      max_level_work_mem_size = std::max
	(max_level_work_mem_size, ldata[l].total_work_mem_size);
    }

    // std::cout << "max_level_work_mem_size=" << max_level_work_mem_size 
    // 	      << " max_level_work_mem_size % 8 = " << (max_level_work_mem_size % 8) << std::endl;
    max_level_work_mem_size += max_level_work_mem_size % 8;

    void* all_work_mem = nullptr;
    cudaMallocManaged(&all_work_mem, max_level_work_mem_size*2);
    void* work_mem[2] = {all_work_mem, (char*)all_work_mem + max_level_work_mem_size};

    for (int lvl=0; lvl<lvls; lvl++) {
      TaskTimer tl("");
      tl.start();
      int l = lvls - lvl - 1;
      auto nnodes = ldata[l].f.size();
      if (opts.verbose())
        std::cout << "#      level " << l << " of " << lvls
                  << " has " << ldata[l].f.size() << " nodes and "
                  << ldata[l].nnodes_small << " small nodes, needs "
                  << ldata[l].factor_mem_size * sizeof(scalar_t) / 1.e6
                  << " MB for factors, "
                  << ldata[l].schur_mem_size * sizeof(scalar_t) / 1.e6
                  << " MB for Schur complements"
                  << std::endl;

      scalar_t* fmem = nullptr;
      cudaMallocManaged(&fmem, ldata[l].factor_mem_size*sizeof(scalar_t));
      ldata[l].f[0]->factor_mem_ = uniq_scalar_t(fmem, cuda_deleter);
      auto wmem = work_mem[l % 2];

      for (std::size_t n=0; n<nnodes; n++) {
        auto& f = *(ldata[l].f[n]);
        const std::size_t dsep = f.dim_sep();
        const std::size_t dupd = f.dim_upd();
        f.F11_ = DenseMW_t(dsep, dsep, fmem, dsep); fmem += dsep*dsep;
        f.F12_ = DenseMW_t(dsep, dupd, fmem, dsep); fmem += dsep*dupd;
        f.F21_ = DenseMW_t(dupd, dsep, fmem, dupd); fmem += dupd*dsep;
        if (dupd) {
          f.F22_ = DenseMW_t(dupd, dupd, static_cast<scalar_t*>(wmem), dupd);
          wmem = static_cast<scalar_t*>(wmem) + dupd*dupd;
        }
      }

      for (std::size_t n=0; n<nnodes; n++) {
        auto& f = *(ldata[l].f[n]);
        f.F11_.zero();
        f.F12_.zero();
        f.F21_.zero();
        f.F22_.zero();
        A.extract_front
          (f.F11_, f.F12_, f.F21_, f.sep_begin_, f.sep_end_, f.upd_, 0);
        if (f.lchild_)
          f.lchild_->extend_add_to_dense
            (f.F11_, f.F12_, f.F21_, f.F22_, &f, 0);
        if (f.rchild_)
          f.rchild_->extend_add_to_dense
	    (f.F11_, f.F12_, f.F21_, f.F22_, &f, 0);
      }

      ldata[l].pF11 = static_cast<scalar_t**>(wmem);
      wmem = static_cast<scalar_t**>(wmem) + ldata[l].nnodes_small;
      ldata[l].pF12 = static_cast<scalar_t**>(wmem);    
      wmem = static_cast<scalar_t**>(wmem) + ldata[l].nnodes_small;
      ldata[l].pF21 = static_cast<scalar_t**>(wmem);
      wmem = static_cast<scalar_t**>(wmem) + ldata[l].nnodes_small;
      ldata[l].pF22 = static_cast<scalar_t**>(wmem);
      wmem = static_cast<scalar_t**>(wmem) + ldata[l].nnodes_small;

      ldata[l].dsep = static_cast<std::size_t*>(wmem);
      wmem = static_cast<std::size_t*>(wmem) + ldata[l].nnodes_small;
      ldata[l].dupd = static_cast<std::size_t*>(wmem);
      wmem = static_cast<std::size_t*>(wmem) + ldata[l].nnodes_small;

      ldata[l].ppiv = static_cast<int**>(wmem);
      wmem = static_cast<int**>(wmem) + nnodes;
      ldata[l].ppiv_small = static_cast<int**>(wmem);
      wmem = static_cast<int**>(wmem) + ldata[l].nnodes_small;

      ldata[l].getrf = static_cast<scalar_t*>(wmem);
      wmem = static_cast<scalar_t*>(wmem) + max_streams * ldata[l].getrf_work_size;
      ldata[l].getrf_err = static_cast<int*>(wmem);
      wmem = static_cast<int*>(wmem) + max_streams;

      for (std::size_t n=0, idx=0; n<nnodes; n++) {
        auto& f = *(ldata[l].f[n]);
        const auto dsep = f.dim_sep();
        const auto dupd = f.dim_upd();
        int size = dsep + dupd;
        if (size < cutoff_size) {
          //gets pointers to small fronts
          ldata[l].pF11[idx] = f.F11_.data();
          ldata[l].pF12[idx] = f.F12_.data();
          ldata[l].pF21[idx] = f.F21_.data(); 
          ldata[l].pF22[idx] = f.F22_.data();
          //gets sizes of small fronts
          ldata[l].dsep[idx] = dsep;
          ldata[l].dupd[idx] = dupd;
	  ldata[l].ppiv_small[idx] = static_cast<int*>(wmem);
          idx++;
        }
	ldata[l].ppiv[n] = static_cast<int*>(wmem);
	wmem = static_cast<int*>(wmem) + dsep;
      }

      //assert(wmem == work_mem[l % 2] + ldata[l].total_work_mem_size);

      // prefetch to GPU
      cudaMemPrefetchAsync
	(work_mem[l % 2], max_level_work_mem_size, device_id, 0);  
      cudaMemPrefetchAsync
	(ldata[l].f[0]->factor_mem_.get(),
	 ldata[l].factor_mem_size*sizeof(scalar_t), device_id, 0);

      long long level_flops = 0;
      for (std::size_t n=0; n<nnodes; n++) {
        auto& f = *(ldata[l].f[n]);
        auto stream = n % max_streams;
        const auto dsep = f.dim_sep(); 
	const auto dupd = f.dim_upd();
	const auto size = dsep + dupd;
        if (size >= cutoff_size) {
          if (dsep) {
            auto LU_stat = cuda::cusolverDngetrf
              (solver_handle[stream], dsep, dsep, f.F11_.data(), dsep,
	       ldata[l].getrf + stream * ldata[l].getrf_work_size,
	       ldata[l].ppiv[n], ldata[l].getrf_err + stream);

	    // TODO if (opts.replace_tiny_pivots()) { ...
            if (dupd) {
              auto LU_stat = cuda::cusolverDngetrs
		(solver_handle[stream], CUBLAS_OP_N, dsep, dupd, 
		 f.F11_.data(), dsep, ldata[l].ppiv[n], f.F12_.data(), dsep, 
		 ldata[l].getrf_err + stream);

              scalar_t alpha(-1.), beta(1.);
              auto stat = cuda::cublasgemm
                (blas_handle[stream], CUBLAS_OP_N, CUBLAS_OP_N,
                 dupd, dupd, dsep, &alpha, f.F21_.data(), dupd,
                 f.F12_.data(), dsep, &beta, f.F22_.data(), dupd);
            } 
          }
        }
	level_flops += LU_flops(f.F11_) +
	  gemm_flops(Trans::N, Trans::N, scalar_t(-1.), f.F21_, f.F12_, scalar_t(1.)) +
	  trsm_flops(Side::L, scalar_t(1.), f.F11_, f.F12_) +
	  trsm_flops(Side::R, scalar_t(1.), f.F11_, f.F21_);
      }
      STRUMPACK_FULL_RANK_FLOPS(level_flops);
      STRUMPACK_FLOPS(level_flops);
      if (ldata[l].nnodes_small) {
	dim3 bt(8, 8);
	cuda::partialLUWrapper
          (ldata[l].nnodes_small, bt, ldata[l].dsep, ldata[l].dupd, 
	   ldata[l].pF11, ldata[l].pF12, ldata[l].pF21, ldata[l].pF22, 
	   ldata[l].ppiv_small);
      }
      cudaDeviceSynchronize();

      // prefetch from device
      cudaMemPrefetchAsync
	(ldata[l].f[0]->factor_mem_.get(),
	 ldata[l].factor_mem_size*sizeof(scalar_t), cudaCpuDeviceId, 0);

      // copy pivot vectors back from the device
      for (std::size_t n=0, idx=0; n<nnodes; n++) {
        auto& f = *(ldata[l].f[n]);
        const auto dsep = f.dim_sep();
	f.piv.resize(dsep);
	std::copy(ldata[l].ppiv[n], ldata[l].ppiv[n]+dsep, f.piv.data());
      }
      if (opts.verbose()) {
	auto level_time = tl.elapsed();
	std::cout << "#  GPU Factorization complete, took: " 
		  << level_time << " seconds, " 
		  << level_flops / 1.e9 << " GFLOPS, "
		  << (float(level_flops) / level_time) / 1.e9 
		  << " GFLOP/s" << std::endl;
      }
    }
    cudaFree(all_work_mem);
    for (int i=0; i<max_streams; i++) {
      cudaStreamDestroy(streams[i]);
      cublasDestroy(blas_handle[i]);
      cusolverDnDestroy(solver_handle[i]);
    }
  }
#endif

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDense<scalar_t,integer_t>::factorization_by_level
  (const SpMat_t& A, const SPOptions<scalar_t>& opts) {
    using uniq_scalar_t = std::unique_ptr<scalar_t[],std::function<void(scalar_t*)>>;
    auto def_deleter = [](scalar_t* ptr) { delete[] ptr; };
    std::unique_ptr<scalar_t[]> schur_mem[2];
    int lvls = this->levels();
    for (int lvl=0; lvl<lvls; lvl++) {
      TaskTimer tl("");
      tl.start();
      int l = lvls - lvl - 1;
      LevelInfo_t ldata;
      setup_level(A, opts, ldata, l);
      auto nnodes = ldata.f.size();

      std::size_t factor_mem_size = 0, schur_mem_size = 0;
      for (std::size_t n=0; n<nnodes; n++) {
        auto& f = *(ldata.f[n]);
        const auto dsep = f.dim_sep();
        const auto dupd = f.dim_upd();
        factor_mem_size += dsep*dsep + 2*dsep*dupd;
        schur_mem_size += dupd*dupd;
      }
      if (opts.verbose())
        std::cout << "#      level " << l << " of " << lvls
                  << " has " << ldata.f.size() << " nodes, needs "
                  << factor_mem_size * sizeof(scalar_t) / 1.e6
                  << " MB for factors, "
                  << schur_mem_size * sizeof(scalar_t) / 1.e6
                  << " MB for Schur complements"
                  << std::endl;


      ldata.f[0]->factor_mem_ = uniq_scalar_t(new scalar_t[factor_mem_size], def_deleter);
      auto fmem = ldata.f[0]->factor_mem_.get();
      schur_mem[lvl % 2] = std::unique_ptr<scalar_t[]>(new scalar_t[schur_mem_size]);
      auto smem = schur_mem[lvl % 2].get();

      for (std::size_t n=0; n<nnodes; n++) {
        auto& f = *(ldata.f[n]);
	std::size_t dsep = f.dim_sep();
	std::size_t dupd = f.dim_upd();
        f.F11_ = DenseMW_t(dsep, dsep, fmem, dsep);  fmem += dsep*dsep;
        f.F12_ = DenseMW_t(dsep, dupd, fmem, dsep);  fmem += dsep*dupd;
	f.F21_ = DenseMW_t(dupd, dsep, fmem, dupd);  fmem += dupd*dsep;
        if (dupd) {
          f.F22_ = DenseMW_t(dupd, dupd, smem, dupd);
          smem += dupd*dupd;
        }
      }

      assert(smem == schur_mem[lvl%2].get() + schur_mem_size);
      assert(fmem == ldata.f[0]->factor_mem_.get() + factor_mem_size);

      long long level_flops = 0;
#pragma omp parallel for if(l!=0) reduction(+:level_flops)
      for (std::size_t n=0; n<nnodes; n++) {
        auto& f = *(ldata.f[n]);
        const auto dsep = f.dim_sep();
        const auto dupd = f.dim_upd();
        f.F11_.zero();
        f.F12_.zero();
        f.F21_.zero();
        f.F22_.zero();
        A.extract_front
          (f.F11_, f.F12_, f.F21_, f.sep_begin_, f.sep_end_, f.upd_, 0);
        if (f.lchild_)
          f.lchild_->extend_add_to_dense
            (f.F11_, f.F12_, f.F21_, f.F22_, &f, 0);
        if (f.rchild_)
          f.rchild_->extend_add_to_dense
            (f.F11_, f.F12_, f.F21_, f.F22_, &f, 0);
        if (dsep) {
          f.piv = f.F11_.LU(0);
          // TODO if (opts.replace_tiny_pivots()) { ...
          if (dupd) {
            f.F11_.solve_LU_in_place(f.F12_, f.piv, 0);
            gemm(Trans::N, Trans::N, scalar_t(-1.), f.F21_, f.F12_,
                 scalar_t(1.), f.F22_, 0);
          }
        }
	level_flops += 
	  LU_flops(f.F11_) +
	  gemm_flops(Trans::N, Trans::N, scalar_t(-1.), f.F21_, f.F12_, scalar_t(1.)) +
	  trsm_flops(Side::L, scalar_t(1.), f.F11_, f.F12_) +
	  trsm_flops(Side::R, scalar_t(1.), f.F11_, f.F21_);
      }
      STRUMPACK_FULL_RANK_FLOPS(level_flops);
      if (opts.verbose()) {
	auto level_time = tl.elapsed();
	std::cout << "#  CPU Factorization complete, took: " 
		  << level_time << " seconds, " 
		  << level_flops / 1.e9 << " GFLOPS, "
		  << (float(level_flops) / level_time) / 1.e9 
		  << " GFLOP/s" << std::endl;
      }
    }
  }

} // end namespace strumpack

#endif
