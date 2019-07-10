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
#ifndef CUDAWRAPPER_H
#define CUDAWRAPPER_H

#include <cmath>
#include <complex>
#include <iostream>
#include <cassert>
#include <memory>
#include "BLASLAPACKWrapper.hpp"
#include "StrumpackParameters.hpp"
#include "StrumpackFortranCInterface.h"
#if defined(STRUMPACK_USE_CUDA)
#include <cusolverDn.h>
#include "cublas_v2.h"
#include <cuda_runtime.h>
#endif

namespace strumpack {

  namespace cuda {
    ///////////////////////////////////////////////////////////
    ///////// CUBLAS //////////////////////////////////////////
    ///////////////////////////////////////////////////////////
    inline cublasStatus_t cublasgemm
    (cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
     int m, int n, int k, const float *alpha, const float *A, int lda,
     const float *B, int ldb, const float *beta, float *C, int ldc) {
      cublasStatus_t stat = cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B,ldb, beta, 
        C, ldc);
      STRUMPACK_FLOPS(blas::gemm_flops(m,n,k,*alpha,*beta));
      STRUMPACK_BYTES(4*blas::gemm_moves(m,n,k));
      return stat;
    }
    inline cublasStatus_t cublasgemm
    (cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
     int m, int n, int k, const double *alpha, const double *A, int lda,
     const double *B, int ldb, const double *beta, double *C, int ldc) {
      cublasStatus_t stat = cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda,  B, ldb, beta, C, ldc);
      STRUMPACK_FLOPS(blas::gemm_flops(m,n,k,*alpha,*beta));
      STRUMPACK_BYTES(8*blas::gemm_moves(m,n,k));
      return stat;
    }
    inline cublasStatus_t cublasgemm
    (cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
     int m, int n, int k, const std::complex<float> *alpha, const std::complex<float> *A, 
     int lda, const std::complex<float> *B, int ldb, const std::complex<float> *beta, 
     std::complex<float> *C, int ldc) {
      const cuComplex* c_alpha = reinterpret_cast<const cuComplex*>(alpha);
      const cuComplex* c_A = reinterpret_cast<const cuComplex*>(A);
      const cuComplex* c_B = reinterpret_cast<const cuComplex*>(B);
      const cuComplex* c_beta = reinterpret_cast<const cuComplex*>(beta);
      cuComplex* c_C = reinterpret_cast<cuComplex*>(C);
    
      cublasStatus_t stat = cublasCgemm(handle, transa, transb, m, n, k, c_alpha, c_A, lda, c_B, ldb, c_beta, c_C, ldc);
      STRUMPACK_FLOPS(4*blas::gemm_flops(m,n,k,*alpha,*beta));
      STRUMPACK_BYTES(2*4*blas::gemm_moves(m,n,k));
      return stat;
    }
    inline cublasStatus_t cublasgemm
    (cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
     int m, int n, int k, const std::complex<double> *alpha, const std::complex<double> *A, 
     int lda, const std::complex<double> *B, int ldb, const std::complex<double> *beta, 
     std::complex<double> *C, int ldc) {
      const cuDoubleComplex* c_alpha = reinterpret_cast<const cuDoubleComplex*>(alpha);
      const cuDoubleComplex* c_A = reinterpret_cast<const cuDoubleComplex*>(A);
      const cuDoubleComplex* c_B = reinterpret_cast<const cuDoubleComplex*>(B);
      const cuDoubleComplex* c_beta = reinterpret_cast<const cuDoubleComplex*>(beta);
      cuDoubleComplex* c_C = reinterpret_cast<cuDoubleComplex*>(C);
    
      cublasStatus_t stat = cublasZgemm(handle, transa, transb, m, n, k, c_alpha, c_A, lda, c_B, ldb, c_beta, 
        c_C, ldc);
      STRUMPACK_FLOPS(4*blas::gemm_flops(m,n,k,*alpha,*beta));
      STRUMPACK_BYTES(2*8*blas::gemm_moves(m,n,k));
      return stat;
    }


    ///////////////////////////////////////////////////////////
    ///////// CUSOLVER ////////////////////////////////////////
    ///////////////////////////////////////////////////////////
    inline cusolverStatus_t cusolverDngetrf_bufferSize
    (cusolverDnHandle_t handle, int m, int n, float *A, int lda, int *Lwork) {
      cusolverStatus_t stat = cusolverDnSgetrf_bufferSize(handle, m, n, A, lda, Lwork);
      return stat;
    }
    inline cusolverStatus_t cusolverDngetrf_bufferSize
    (cusolverDnHandle_t handle, int m, int n, double *A, int lda, int *Lwork) {
      cusolverStatus_t stat = cusolverDnDgetrf_bufferSize(handle, m, n, A, lda, Lwork);
      return stat;
    }
    inline cusolverStatus_t cusolverDngetrf_bufferSize
    (cusolverDnHandle_t handle, int m, int n, std::complex<float> *A, int lda, int *Lwork) {
      cuComplex* c_A = reinterpret_cast<cuComplex*>(A);
      cusolverStatus_t stat = cusolverDnCgetrf_bufferSize(handle, m, n, c_A, lda, Lwork);
      return stat;
    }
    inline cusolverStatus_t cusolverDngetrf_bufferSize
    (cusolverDnHandle_t handle, int m, int n, std::complex<double> *A, int lda, int *Lwork) {
      cuDoubleComplex* c_A = reinterpret_cast<cuDoubleComplex*>(A);
      cusolverStatus_t stat = cusolverDnZgetrf_bufferSize(handle, m, n, c_A, lda, Lwork);
      return stat;
    }


    inline long long getrf_flops(long long m, long long n) {
      // TODO check this
      if (m < n) return (m / 2 * (m * (n - m / 3 - 1) + n) + 2 * m / 3) +
                   (m / 2 * (m * (n - m / 3) - n) + m / 6);
      else return n * n * (m - n/3 - 1) / 2 + m + 2 * n / 3 +
             n * (n * (m - (1./3.) * n - 1) / 2 - m) + n / 6;
    }
    inline cusolverStatus_t cusolverDngetrf
    (cusolverDnHandle_t handle, int m, int n, float *A, int lda, float *Workspace, 
     int *devIpiv, int *devInfo) {
       cusolverStatus_t stat = cusolverDnSgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
       STRUMPACK_FLOPS(blas::getrf_flops(m,n));
       return stat;
    }
    inline cusolverStatus_t cusolverDngetrf
    (cusolverDnHandle_t handle, int m, int n, double *A, int lda, double *Workspace, 
     int *devIpiv, int *devInfo) {
       cusolverStatus_t stat = cusolverDnDgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
       STRUMPACK_FLOPS(blas::getrf_flops(m,n));
       return stat;
    }
    inline cusolverStatus_t cusolverDngetrf
    (cusolverDnHandle_t handle, int m, int n, std::complex<float> *A, int lda, std::complex<float> *Workspace, 
     int *devIpiv, int *devInfo) {
       cuComplex* c_A = reinterpret_cast<cuComplex*>(A);
       cuComplex* c_Workspace = reinterpret_cast<cuComplex*>(Workspace);
       cusolverStatus_t stat = cusolverDnCgetrf(handle, m, n, c_A, lda, c_Workspace, devIpiv, devInfo);
       STRUMPACK_FLOPS(4*blas::getrf_flops(m,n));
       return stat;
    }
    inline cusolverStatus_t cusolverDngetrf
    (cusolverDnHandle_t handle, int m, int n, std::complex<double> *A, int lda, std::complex<double> *Workspace, 
     int *devIpiv, int *devInfo) {
       cuDoubleComplex* c_A = reinterpret_cast<cuDoubleComplex*>(A);
       cuDoubleComplex* c_Workspace = reinterpret_cast<cuDoubleComplex*>(Workspace);
       cusolverStatus_t stat = cusolverDnZgetrf(handle, m, n, c_A, lda, c_Workspace, devIpiv, devInfo);
       STRUMPACK_FLOPS(4*blas::getrf_flops(m,n));
       return stat;
    }


    inline long long getrs_flops(long long n, long long nrhs) {
      return 2 * n * n * nrhs;
    }
    inline cusolverStatus_t cusolverDngetrs
    (cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const float *A, 
     int lda, const int *devIpiv, float *B, int ldb, int *devInfo) {
       cusolverStatus_t stat = cusolverDnSgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
       STRUMPACK_FLOPS(blas::getrs_flops(n,nrhs));
       return stat;
    }
    inline cusolverStatus_t cusolverDngetrs
    (cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const double *A, 
     int lda, const int *devIpiv, double *B, int ldb, int *devInfo) {
       cusolverStatus_t stat = cusolverDnDgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
       STRUMPACK_FLOPS(blas::getrs_flops(n,nrhs));
       return stat;
    }
    inline cusolverStatus_t cusolverDngetrs
    (cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const std::complex<float> *A, 
     int lda, const int *devIpiv, std::complex<float> *B, int ldb, int *devInfo) {
       const cuComplex* c_A = reinterpret_cast<const cuComplex*>(A); 
       cuComplex* c_B = reinterpret_cast<cuComplex*>(B); 
       cusolverStatus_t stat = cusolverDnCgetrs(handle, trans, n, nrhs, c_A, lda, devIpiv, c_B, ldb, devInfo);
       STRUMPACK_FLOPS(4*blas::getrs_flops(n,nrhs));
       return stat;
    }
    inline cusolverStatus_t cusolverDngetrs
    (cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const std::complex<double> *A, 
     int lda, const int *devIpiv, std::complex<double> *B, int ldb, int *devInfo) {
       const cuDoubleComplex* c_A = reinterpret_cast<const cuDoubleComplex*>(A); 
       cuDoubleComplex* c_B = reinterpret_cast<cuDoubleComplex*>(B); 
       cusolverStatus_t stat = cusolverDnZgetrs(handle, trans, n, nrhs, c_A, lda, devIpiv, c_B, ldb, devInfo);
       STRUMPACK_FLOPS(4*blas::getrs_flops(n,nrhs));
       return stat;
    }
  }
}
#endif
