#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>

namespace strumpack {
  
  namespace cuda {

    ///////////////////////////////////////////////////////////
    ///////// CUBLAS //////////////////////////////////////////
    ///////////////////////////////////////////////////////////
    cublasStatus_t cublasgemm
    (cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
     int m, int n, int k, const float *alpha, const float *A, int lda,
     const float *B, int ldb, const float *beta, float *C, int ldc);

    cublasStatus_t cublasgemm
    (cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
     int m, int n, int k, const double *alpha, const double *A, int lda,
     const double *B, int ldb, const double *beta, double *C, int ldc);

    cublasStatus_t cublasgemm
    (cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
     int m, int n, int k, const std::complex<float> *alpha, const std::complex<float> *A, 
     int lda, const std::complex<float> *B, int ldb, const std::complex<float> *beta, 
     std::complex<float> *C, int ldc);
    
    cublasStatus_t cublasgemm
    (cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
     int m, int n, int k, const std::complex<double> *alpha, const std::complex<double> *A, 
     int lda, const std::complex<double> *B, int ldb, const std::complex<double> *beta, 
     std::complex<double> *C, int ldc);

    ///////////////////////////////////////////////////////////
    ///////// CUSOLVER ////////////////////////////////////////
    ///////////////////////////////////////////////////////////
    cusolverStatus_t cusolverDngetrf_bufferSize
    (cusolverDnHandle_t handle, int m, int n, float *A, int lda, int *Lwork);

    cusolverStatus_t cusolverDngetrf_bufferSize
    (cusolverDnHandle_t handle, int m, int n, double *A, int lda, int *Lwork);

    cusolverStatus_t cusolverDngetrf_bufferSize
    (cusolverDnHandle_t handle, int m, int n, std::complex<float> *A, int lda, int *Lwork);

    cusolverStatus_t cusolverDngetrf_bufferSize
    (cusolverDnHandle_t handle, int m, int n, std::complex<double> *A, int lda, int *Lwork);


    cusolverStatus_t cusolverDngetrf
    (cusolverDnHandle_t handle, int m, int n, float *A, int lda, float *Workspace, 
     int *devIpiv, int *devInfo);

    cusolverStatus_t cusolverDngetrf
    (cusolverDnHandle_t handle, int m, int n, double *A, int lda, double *Workspace, 
     int *devIpiv, int *devInfo); 
        
    cusolverStatus_t cusolverDngetrf
    (cusolverDnHandle_t handle, int m, int n, std::complex<float> *A, int lda, std::complex<float> *Workspace, 
     int *devIpiv, int *devInfo); 
        
    cusolverStatus_t cusolverDngetrf
    (cusolverDnHandle_t handle, int m, int n, std::complex<double> *A, int lda, std::complex<double> *Workspace, 
     int *devIpiv, int *devInfo); 


    cusolverStatus_t cusolverDngetrs
    (cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const float *A, 
     int lda, const int *devIpiv, float *B, int ldb, int *devInfo); 

    cusolverStatus_t cusolverDngetrs
    (cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const double *A, 
     int lda, const int *devIpiv, double *B, int ldb, int *devInfo);
        
    cusolverStatus_t cusolverDngetrs
    (cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const std::complex<float> *A, 
     int lda, const int *devIpiv, std::complex<float> *B, int ldb, int *devInfo);

    cusolverStatus_t cusolverDngetrs
    (cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const std::complex<double> *A, 
     int lda, const int *devIpiv, std::complex<double> *B, int ldb, int *devInfo);

    ///////////////////////////////////////////////////////////
    ///////// KERNELS /////////////////////////////////////////
    ///////////////////////////////////////////////////////////
    
    void partialLUWrapper
    (int num_blocks, dim3 threads_per_block, int* l_n1, 
     int* l_n2, double** l_A11, double** l_A12, double** l_A21, 
     double** l_A22, int** l_piv);
    void LUkernelWrapper
    (int num_blocks, dim3 threads_per_block, int* l_n1, 
     int* l_n2, double** l_A11, double** l_A12, double** l_A21, 
     double** l_A22, int** l_piv);
  }
}
