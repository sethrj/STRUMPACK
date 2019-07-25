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
//#include "../dense/BLASLAPACKWrapper.hpp"
// #include "StrumpackParameters.hpp"
// #include "StrumpackFortranCInterface.h"
#include "FrontalMatrixDenseCudaWrapper.h"
//#include <cusolverDn.h>
//#include "cublas_v2.h"
//#include <cuda_runtime.h>
#define TILE_DIM 8
//#endif

namespace strumpack {

  namespace cuda {
    ///////////////////////////////////////////////////////////
    ///////// CUBLAS //////////////////////////////////////////
    ///////////////////////////////////////////////////////////
    __host__ inline cublasStatus_t cublasgemmwrapper
    (cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
     int m, int n, int k, const float *alpha, const float *A, int lda,
     const float *B, int ldb, const float *beta, float *C, int ldc) {
      cublasStatus_t stat = cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B,ldb, beta, 
        C, ldc);
      return stat;
    }
    __host__ inline cublasStatus_t cublasgemmwrapper
    (cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
     int m, int n, int k, const double *alpha, const double *A, int lda,
     const double *B, int ldb, const double *beta, double *C, int ldc) {
      cublasStatus_t stat = cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda,  B, ldb, beta, C, ldc);
      return stat;
    }
    __host__ inline cublasStatus_t cublasgemmwrapper
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
      return stat;
    }
    __host__ inline cublasStatus_t cublasgemmwrapper
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
      return stat;
    }
    cublasStatus_t cublasgemm
    (cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
     int m, int n, int k, const float *alpha, const float *A, int lda,
     const float *B, int ldb, const float *beta, float *C, int ldc) {
      cublasStatus_t stat = cublasgemmwrapper(handle, transa, transb, m, n, k, alpha, A, lda, 
                                              B, ldb, beta, C, ldc);
      return stat;
    }
    cublasStatus_t cublasgemm
    (cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
     int m, int n, int k, const double *alpha, const double *A, int lda,
     const double *B, int ldb, const double *beta, double *C, int ldc) {
       cublasStatus_t stat = cublasgemmwrapper(handle, transa, transb, m, n, k, alpha, A, lda, 
                                               B, ldb, beta, C, ldc);
       return stat;
    }
    cublasStatus_t cublasgemm
    (cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
     int m, int n, int k, const std::complex<float> *alpha, const std::complex<float> *A, 
     int lda, const std::complex<float> *B, int ldb, const std::complex<float> *beta, 
     std::complex<float> *C, int ldc) {
       cublasStatus_t stat = cublasgemmwrapper(handle, transa, transb, m, n, k, alpha, A, lda, 
                                               B, ldb, beta, C, ldc);
       return stat;
    }
    cublasStatus_t cublasgemm
    (cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
     int m, int n, int k, const std::complex<double> *alpha, const std::complex<double> *A, 
     int lda, const std::complex<double> *B, int ldb, const std::complex<double> *beta, 
     std::complex<double> *C, int ldc) {
       cublasStatus_t stat = cublasgemmwrapper(handle, transa, transb, m, n, k, alpha, A, lda, 
                                               B, ldb, beta, C, ldc);
       return stat;
    }

    ///////////////////////////////////////////////////////////
    ///////// CUSOLVER ////////////////////////////////////////
    ///////////////////////////////////////////////////////////
    __host__ inline cusolverStatus_t cusolverDngetrf_buffersizewrapper
    (cusolverDnHandle_t handle, int m, int n, float *A, int lda, int *Lwork) {
      cusolverStatus_t stat = cusolverDnSgetrf_bufferSize(handle, m, n, A, lda, Lwork);
      return stat;
    }
    __host__ inline cusolverStatus_t cusolverDngetrf_buffersizewrapper
    (cusolverDnHandle_t handle, int m, int n, double *A, int lda, int *Lwork) {
      cusolverStatus_t stat = cusolverDnDgetrf_bufferSize(handle, m, n, A, lda, Lwork);
      return stat;
    }
    __host__ inline cusolverStatus_t cusolverDngetrf_buffersizewrapper
    (cusolverDnHandle_t handle, int m, int n, std::complex<float> *A, int lda, int *Lwork) {
      cuComplex* c_A = reinterpret_cast<cuComplex*>(A);
      cusolverStatus_t stat = cusolverDnCgetrf_bufferSize(handle, m, n, c_A, lda, Lwork);
      return stat;
    }
    __host__ inline cusolverStatus_t cusolverDngetrf_buffersizewrapper
    (cusolverDnHandle_t handle, int m, int n, std::complex<double> *A, int lda, int *Lwork) {
      cuDoubleComplex* c_A = reinterpret_cast<cuDoubleComplex*>(A);
      cusolverStatus_t stat = cusolverDnZgetrf_bufferSize(handle, m, n, c_A, lda, Lwork);
      return stat;
    }

    cusolverStatus_t cusolverDngetrf_bufferSize
    (cusolverDnHandle_t handle, int m, int n, float *A, int lda, int *Lwork) {
      cusolverStatus_t stat = cusolverDngetrf_buffersizewrapper(handle, m, n, A, lda, Lwork);
      return stat;
    }
    cusolverStatus_t cusolverDngetrf_bufferSize
    (cusolverDnHandle_t handle, int m, int n, double *A, int lda, int *Lwork) {
      cusolverStatus_t stat = cusolverDngetrf_buffersizewrapper(handle, m, n, A, lda, Lwork);
      return stat;
    }
    cusolverStatus_t cusolverDngetrf_bufferSize
    (cusolverDnHandle_t handle, int m, int n, std::complex<float> *A, int lda, int *Lwork) {
      cusolverStatus_t stat = cusolverDngetrf_buffersizewrapper(handle, m, n, A, lda, Lwork);
      return stat;
    }
    cusolverStatus_t cusolverDngetrf_bufferSize
    (cusolverDnHandle_t handle, int m, int n, std::complex<double> *A, int lda, int *Lwork) {
      cusolverStatus_t stat = cusolverDngetrf_buffersizewrapper(handle, m, n, A, lda, Lwork);
      return stat;
    }

    __host__ inline cusolverStatus_t cusolverDngetrfwrapper
    (cusolverDnHandle_t handle, int m, int n, float *A, int lda, float *Workspace, 
     int *devIpiv, int *devInfo) {
       cusolverStatus_t stat = cusolverDnSgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
       return stat;
    }
    __host__ inline cusolverStatus_t cusolverDngetrfwrapper
    (cusolverDnHandle_t handle, int m, int n, double *A, int lda, double *Workspace, 
     int *devIpiv, int *devInfo) {
       cusolverStatus_t stat = cusolverDnDgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
       return stat;
    }
    __host__ inline cusolverStatus_t cusolverDngetrfwrapper
    (cusolverDnHandle_t handle, int m, int n, std::complex<float> *A, int lda, std::complex<float> *Workspace, 
     int *devIpiv, int *devInfo) {
       cuComplex* c_A = reinterpret_cast<cuComplex*>(A);
       cuComplex* c_Workspace = reinterpret_cast<cuComplex*>(Workspace);
       cusolverStatus_t stat = cusolverDnCgetrf(handle, m, n, c_A, lda, c_Workspace, devIpiv, devInfo);
       return stat;
    }
    __host__ inline cusolverStatus_t cusolverDngetrfwrapper
    (cusolverDnHandle_t handle, int m, int n, std::complex<double> *A, int lda, std::complex<double> *Workspace, 
     int *devIpiv, int *devInfo) {
       cuDoubleComplex* c_A = reinterpret_cast<cuDoubleComplex*>(A);
       cuDoubleComplex* c_Workspace = reinterpret_cast<cuDoubleComplex*>(Workspace);
       cusolverStatus_t stat = cusolverDnZgetrf(handle, m, n, c_A, lda, c_Workspace, devIpiv, devInfo);
       return stat;
    }

    cusolverStatus_t cusolverDngetrf
    (cusolverDnHandle_t handle, int m, int n, float *A, int lda, float *Workspace, 
     int *devIpiv, int *devInfo) {
       cusolverStatus_t stat = cusolverDngetrfwrapper(handle, m, n, A, lda, Workspace, 
                                                      devIpiv, devInfo);
       return stat;
    }
    cusolverStatus_t cusolverDngetrf
    (cusolverDnHandle_t handle, int m, int n, double *A, int lda, double *Workspace, 
     int *devIpiv, int *devInfo) {
       cusolverStatus_t stat = cusolverDngetrfwrapper(handle, m, n, A, lda, Workspace, 
                                                      devIpiv, devInfo);
       return stat;
    }
    cusolverStatus_t cusolverDngetrf
    (cusolverDnHandle_t handle, int m, int n, std::complex<float> *A, int lda, std::complex<float> *Workspace, 
     int *devIpiv, int *devInfo) {
       cusolverStatus_t stat = cusolverDngetrfwrapper(handle, m, n, A, lda, Workspace, 
                                                      devIpiv, devInfo);
       return stat;
    }
    cusolverStatus_t cusolverDngetrf
    (cusolverDnHandle_t handle, int m, int n, std::complex<double> *A, int lda, std::complex<double> *Workspace, 
     int *devIpiv, int *devInfo) {
       cusolverStatus_t stat = cusolverDngetrfwrapper(handle, m, n, A, lda, Workspace, 
                                                      devIpiv, devInfo);
       return stat;
    }

    __host__ inline cusolverStatus_t cusolverDngetrswrapper
    (cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const float *A, 
     int lda, const int *devIpiv, float *B, int ldb, int *devInfo) {
       cusolverStatus_t stat = cusolverDnSgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
       return stat;
    }
    __host__ inline cusolverStatus_t cusolverDngetrswrapper
    (cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const double *A, 
     int lda, const int *devIpiv, double *B, int ldb, int *devInfo) {
       cusolverStatus_t stat = cusolverDnDgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
       return stat;
    }
    __host__ inline cusolverStatus_t cusolverDngetrswrapper
    (cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const std::complex<float> *A, 
     int lda, const int *devIpiv, std::complex<float> *B, int ldb, int *devInfo) {
       const cuComplex* c_A = reinterpret_cast<const cuComplex*>(A); 
       cuComplex* c_B = reinterpret_cast<cuComplex*>(B); 
       cusolverStatus_t stat = cusolverDnCgetrs(handle, trans, n, nrhs, c_A, lda, devIpiv, c_B, ldb, devInfo);
       return stat;
    }
    __host__ inline cusolverStatus_t cusolverDngetrswrapper
    (cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const std::complex<double> *A, 
     int lda, const int *devIpiv, std::complex<double> *B, int ldb, int *devInfo) {
       const cuDoubleComplex* c_A = reinterpret_cast<const cuDoubleComplex*>(A); 
       cuDoubleComplex* c_B = reinterpret_cast<cuDoubleComplex*>(B); 
       cusolverStatus_t stat = cusolverDnZgetrs(handle, trans, n, nrhs, c_A, lda, devIpiv, c_B, ldb, devInfo);
       return stat;
    }

    cusolverStatus_t cusolverDngetrs
    (cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const float *A, 
     int lda, const int *devIpiv, float *B, int ldb, int *devInfo) {
      cusolverStatus_t stat = cusolverDngetrswrapper
        (handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
      return stat;
    }
    cusolverStatus_t cusolverDngetrs
    (cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const double *A, 
     int lda, const int *devIpiv, double *B, int ldb, int *devInfo) {
      cusolverStatus_t stat = cusolverDngetrswrapper
        (handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
      return stat;
    }
    cusolverStatus_t cusolverDngetrs
    (cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const std::complex<float> *A, 
     int lda, const int *devIpiv, std::complex<float> *B, int ldb, int *devInfo) {
      cusolverStatus_t stat = cusolverDngetrswrapper
        (handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
      return stat;
    }
    cusolverStatus_t cusolverDngetrs
    (cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const std::complex<double> *A, 
     int lda, const int *devIpiv, std::complex<double> *B, int ldb, int *devInfo) {
      cusolverStatus_t stat = cusolverDngetrswrapper
        (handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
      return stat;
    }
    ///////////////////////////////////////////////////////////
    ///////// KERNELS /////////////////////////////////////////
    ///////////////////////////////////////////////////////////

    __device__ void swap(size_t& a, size_t& b) {
      size_t tmp = a;
      a = b;
      b = tmp;
    }
    __device__ void swap(int& a, int& b) {
      int tmp = a;
      a = b;
      b = tmp;
    }
    __device__ void swap(double& a, double& b) {
      double& tmp = a;
      a = b;
      b = tmp;
    }

    __global__ void partialLU
    (size_t* l_n1, size_t* l_n2, double** l_A11, double** l_A12, double** l_A21, double** l_A22, int** l_piv) {

      size_t n1 = l_n1[blockIdx.x];
      size_t n2 = l_n2[blockIdx.x];
      double* A11 = l_A11[blockIdx.x];
      double* A12 = l_A12[blockIdx.x];
      double* A21 = l_A21[blockIdx.x];
      double* A22 = l_A22[blockIdx.x];
      int* piv = l_piv[blockIdx.x]; 


      for (int i=0; i<n1; i++) piv[i] = i+1; // fortran convention
      for (int j=0; j<n1; j++) {
        auto Amax = A11[j+j*n1];
        int imax = j;
        // find pivot element
        for (int i=j+1; i<n1; i++) {
          if (fabs(A11[i+j*n1]) > fabs(Amax)) {
            Amax = A11[i+j*n1];
            imax = i;
          }
        }
        //if (Amax == 0) return j;
        if (imax != j) {
          cuda::swap(piv[j], piv[imax]);
          for (int i=0; i<n1; i++)
            cuda::swap(A11[imax+i*n1], A11[j+i*n1]);
          for (int i=0; i<n2; i++)
            cuda::swap(A12[imax+i*n1], A12[j+i*n1]);
        }
        double one = 1.0;
        auto iAmax = one / Amax;
        for (int i=j+1; i<n1; i++)
          A11[i+j*n1] *= iAmax;
        for (int i=j+1; i<n1; i++)
          for (int k=j+1; k<n1; k++)
            A11[k+i*n1] -= A11[k+j*n1] * A11[j+i*n1];
      }
      //trsm with L
      for (int i=0; i<n1; i++)
        for (int k=0; k<n2; k++)
          for (int j=0; j<i; j++)
            A12[i+k*n1] -= A11[i+j*n1] * A12[j+k*n1];
      // trsm with U
      for (int i=n1-1; i>=0; i--)
        for (int k=0; k<n2; k++) {
          for (int j=i+1; j<n1; j++)
            A12[i+k*n1] -= A11[i+j*n1] * A12[j+k*n1];
          A12[i+k*n1] /= A11[i+i*n1];
        }
      // gemm
      for (int j=0; j<n2; j++)
        for (int i=0; i<n2; i++)
          for (int k=0; k<n1; k++)
            A22[i+j*n2] -= A21[i+k*n2] * A12[k+j*n1];
    }

    void partialLUWrapper
    (int num_blocks, int threads_per_block, size_t* l_n1, 
     size_t* l_n2, double** l_A11, double** l_A12, double** l_A21, double** l_A22, int** l_piv) {
       partialLU<<<num_blocks,threads_per_block>>>
         (l_n1, l_n2, l_A11, l_A12, l_A21, l_A22, l_piv);
    }
  }
}
#endif
