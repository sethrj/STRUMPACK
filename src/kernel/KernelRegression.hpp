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
 * Developers: Pieter Ghysels, Gustavo Chavez, Xiaoye S. Li.
 *             (Lawrence Berkeley National Lab, Computational Research
 *             Division).
 *
 */
/*!
 * \file Kernel.hpp
 *
 * \brief Definitions of several kernel functions, and helper
 * routines. Also provides driver routines for kernel ridge
 * regression.
 */
#ifndef STRUMPACK_KERNEL_REGRESSION_HPP
#define STRUMPACK_KERNEL_REGRESSION_HPP

#include "Kernel.hpp"
#include "HSS/HSSMatrix.hpp"
#include "sparse/GMRes.hpp"
#if defined(STRUMPACK_USE_MPI)
#include "HSS/HSSMatrixMPI.hpp"
#if defined(STRUMPACK_USE_BPACK)
#include "HODLR/HODLRMatrix.hpp"
#endif
#endif

#define ITERATIVE_REFINEMENT 0
#define DIRECT_SOLVE 1

extern std::size_t memory_counter;
extern std::size_t peak_dense_mem;

namespace strumpack {

  namespace kernel {

    template<typename scalar_t>
    DenseMatrix<scalar_t> Kernel<scalar_t>::fit_HSS
    (std::vector<scalar_t>& labels, const HSS::HSSOptions<scalar_t>& opts) {
      TaskTimer timer("compression");
      if (opts.verbose())
        std::cout << "# starting HSS compression..." << std::endl;
      std::vector<int> perm;
      timer.start();
      HSS::HSSMatrix<scalar_t> H(*this, perm, opts);
      std::cout << "# HSS_compression_time = "
                << timer.elapsed() << std::endl;
      DenseMW_t B(1, n(), labels.data(), 1);
      B.lapmt(perm, true);
      perm.clear();
      // draw(H,"plot_");
      if (H.is_compressed())
        std::cout << "# created HSS matrix of dimension "
                  << H.rows() << " x " << H.cols()
                  << " with " << H.levels() << " levels" << std::endl
                  << "# compression succeeded!" << std::endl;
      else std::cout << "# compression failed!!!" << std::endl;
      std::cout << "# rank_H = " << H.rank() << std::endl
                << "# HSS memory_H = "
                << H.memory() / 1e6 << " MB" << std::endl;
      // std::cout << "# HSS memory_counter_H = "
      //           << memory_counter / 1e6 << " MB" << std::endl;

      // // Computing error against dense matrix
      // if ( n()<= 1000 ){
      //   DenseM_t Kdense(n(),n());
      //   for(int j = 0; j < n(); j++){
      //     for(int i = 0; i < n(); i++){
      //       Kdense(i,j) = eval(i,j);
      //     }
      //   }
      //   auto HSSd = H.dense();
      //   HSSd.scaled_add(-1., Kdense);
      //   std::cout << "# Compression rel error = ||HSSd-Hd||_F/||Hd||_F = " <<
      //   HSSd.normF() / Kdense.normF() << std::endl;
      // }
      // params::print_dense_counter("AFTER COMP");

      std::cout << "# factorization started..." << std::endl;
      timer.start();
      auto ULV = H.factor();
      if (1)
        std::cout << "# ULV_factorization_time = "
                  << timer.elapsed()
                  << std::endl
                  << "# ULV_memory_MB = "
                  << ULV.memory()/1.e6
                  << std::endl;
      // params::print_dense_counter("AFTER FACT");


      std::cout << "# solution started..." << std::endl;
      DenseM_t weights(n(), 1, labels.data(), n());
      H.solve(ULV, weights);

      #if 0
      #if ITERATIVE_REFINEMENT == 1
        DenseMW_t rhs(n(), 1, labels.data(), n());
        DenseM_t weights(rhs), residual(n(), 1);
        H.solve(ULV, weights);
        auto rhs_normF = rhs.normF();
        using real_t = typename RealType<scalar_t>::value_type;
        for (int ref=0; ref<3; ref++) {
          auto residual = H.apply(weights);
          residual.scaled_add(scalar_t(-1.), rhs);
          auto rres = residual.normF() / rhs_normF;
          if (opts.verbose())
            std::cout << "||H*weights - labels||_2/||labels||_2 = "
                      << rres << std::endl;
          if (rres < 10*blas::lamch<real_t>('E')) break;
          H.solve(ULV, residual);
          weights.scaled_add(scalar_t(-1.), residual);
        }
      #else // no iterative refinement
        DenseM_t weights(n(), 1, labels.data(), n());
        #if DIRECT_SOLVE == 1
          H.solve(ULV, weights);
        #else // no direct solve
          const DenseMatrix<scalar_t> z_gmres(weights); // copy to pass to GMRES (x)
          DenseMatrix<scalar_t> weights_gmres(weights); // copy to pass to GMRES (b)
          auto Kdense = // Needs the original dense kernel matrix

          // Lambda spmv
          std::function<void(const scalar_t*, scalar_t*)>
          spmv = [&](const scalar_t* x, scalar_t* b) {
            auto cdmw_x = ConstDenseMatrixWrapperPtr<scalar_t> (n(), size_t(1), x, n());
            DenseMatrixWrapper<scalar_t> dmw_b(n(), size_t(1), b, n());
            // Operation: b = KMM*x
            gemv(
              Trans::N,
              scalar_t(1.0),
              Kdense,          // DenseMatrix
              *cdmw_x,            // ConstDenseMatrixWrapperPtr
              scalar_t(0.0),
              dmw_b,              // DenseMatrix
              0                   // OMP task depth
            );
          };

          // Lambda for HSS as preconditioner
          auto HSS_solve = [&](scalar_t* bp) {
            DenseMatrixWrapper<scalar_t> dmw_bp(n(), size_t(1), bp, n());
            H.solve(ULV, dmw_bp);
          };

          // // Lambda for dense LU as preconditioner
          // auto LU_solve = [&](scalar_t* bp) {
          //   DenseMatrixWrapper<scalar_t> dmw_bp(n(), size_t(1), bp, n());
          //   auto x = Kdense_LU.solve(dmw_bp, piv, 0);
          //   dmw_bp.copy(x);
          // };

          // Lambda for GMRes
          auto iter_wrapper = [&](const std::function<void(scalar_t*)>& prec) {
            scalar_t opts_rel_tol       = 1e-4;
            scalar_t opts_abs_tol       = 1e-20;
            int      opts_maxit         = 100;
            int      opts_gmres_restart = 30;
            int      totit;
            GMRes
            (
              spmv,
              prec,
              n(),
              weights_gmres.data(),         // scalar_t*
              z_gmres.data(),               // const scalar_t*
              opts_rel_tol,
              opts_abs_tol,
              totit,
              opts_maxit,
              opts_gmres_restart,
              GramSchmidtType::MODIFIED,
              false,                          // non_zero_guess?
              true//,                         // verbose?
              //hss_opts.rel_tol()            // to name output file
            );
            std::cout << "## Krylov iterations: " << totit << std::endl;
          };

          // Execute GMRES
          iter_wrapper(HSS_solve);
          // iter_wrapper(LU_solve);

          // Copy result to vector <weights>
          weights = weights_gmres;
        #endif

      #endif // iterative or direct solve
      #endif

      if (opts.verbose())
        std::cout << "# HSS_solve_time = " << timer.elapsed() << std::endl;

      // DenseM_t weights(1, 1);
      return weights;
    }

    template<typename scalar_t>
    std::vector<scalar_t> Kernel<scalar_t>::predict
    (const DenseM_t& test, const DenseM_t& weights) const {
      assert(test.rows() == d());
      std::vector<scalar_t> prediction(test.cols());
      #pragma omp parallel for
      for (std::size_t c=0; c<test.cols(); c++)
        for (std::size_t r=0; r<n(); r++)
          prediction[c] += weights(r, 0) *
            eval_kernel_function(data_.ptr(0, r), test.ptr(0, c));
      return prediction;
    }

    template<typename scalar_t>
    DenseMatrix<scalar_t> Kernel<scalar_t>::fit_HSS_multiple
    (std::vector<scalar_t>& labels, const HSS::HSSOptions<scalar_t>& opts,
      std::vector<scalar_t> lambda_vec) {
      TaskTimer timer("compression");
      if (opts.verbose())
        std::cout << "# starting HSS compression..." << std::endl;
      std::vector<int> perm;
      timer.start();
      HSS::HSSMatrix<scalar_t> H(*this, perm, opts);
      std::cout << "# HSS_compression_time = "
                << timer.elapsed() << std::endl;
      DenseMW_t B(1, n(), labels.data(), 1);
      B.lapmt(perm, true);
      perm.clear();
      if (H.is_compressed())
        std::cout << "# created HSS matrix of dimension "
                  << H.rows() << " x " << H.cols()
                  << " with " << H.levels() << " levels" << std::endl
                  << "# compression succeeded!" << std::endl;
      else std::cout << "# compression failed!!!" << std::endl;
      std::cout << "# rank_H = " << H.rank() << std::endl
                << "# HSS memory_H = "
                << H.memory() / 1e6 << " MB" << std::endl;

      // Computing error against dense matrix
      if ( n()<= 500 ){
        DenseM_t Kdense(n(),n());
        for(int j = 0; j < n(); j++){
          for(int i = 0; i < n(); i++){
            Kdense(i,j) = eval(i,j);
          }
        }
        auto HSSd = H.dense();
        HSSd.scaled_add(-1., Kdense);
        std::cout << "# Compression rel error = ||HSSd-Hd||_F/||Hd||_F = " <<
        HSSd.normF() / Kdense.normF() << std::endl;
      }

      std::cout << std::endl;
      int number_weights = lambda_vec.size();
      std::vector<std::vector<scalar_t>> vov_weights;
      H.shift(-lambda_); // Substract original lambda
      for(auto ilambda: lambda_vec){
        std::cout << "Solving for lambda = " << ilambda << std::endl;
        H.shift(ilambda);

        std::cout << "# factorization started..." << std::endl;
        timer.start();
        auto ULV = H.factor();
        std::cout << "# ULV_factorization_time = "
                  << timer.elapsed()
                  << std::endl
                  << "# ULV_memory_MB = "
                  << ULV.memory()/1.e6
                  << std::endl;

        std::cout << "# solution started..." << std::endl;
        DenseM_t weight(n(), 1, labels.data(), n());
        H.solve(ULV, weight);

        if (opts.verbose())
          std::cout << "# HSS_solve_time = " << timer.elapsed() << std::endl;

        std::vector<scalar_t> tmp(weight.data(), weight.data() + n());
        vov_weights.push_back(tmp);
        H.shift(-ilambda);
        std::cout << std::endl;
      }

      // Combine vector of vector, of weights
      std::vector<scalar_t> allc;
      allc.reserve(number_weights*n());
      for (auto& items: vov_weights)
          std::move(items.begin(), items.end(), std::back_inserter(allc));
      DenseMatrix<scalar_t> weights(n(), number_weights, allc.data(), n());

      // DenseM_t weights(1, 1);
      return weights;
    }

    template<typename scalar_t>
    DenseMatrix<scalar_t> Kernel<scalar_t>::predict_multiple // shared memory here
    (const DenseM_t& test, const DenseM_t& weights) const {
      assert(test.rows() == d());
      int m = test.cols();
      int numw = weights.cols();
      // weights.print("weights_shared",true,10);
      // WARNING: test is read transposed
      // std::cout << "n = " << n() << std::endl;
      // std::cout << "m = " << m << std::endl;
      // std::cout << "w = " << numw << std::endl;
      // std::cout << "test_points(" << test.rows() <<    "," << test.cols()    << ")" << std::endl;
      // std::cout << "weights    (" << weights.rows() << "," << weights.cols() << ")" << std::endl;
      DenseMatrix<scalar_t> prediction(m, numw);
      prediction.zero();

      // DenseMatrix<scalar_t> Kp(n(),m);
      for(int w = 0; w < numw; w++){
        #pragma omp parallel for
        for (std::size_t c=0; c<test.cols(); c++){
          for (std::size_t r=0; r<n(); r++){
            prediction(c,w) += weights(r, w) *
              eval_kernel_function(data_.ptr(0, r), test.ptr(0, c));
            // Kp(r,c) = eval_kernel_function(data_.ptr(0, r), test.ptr(0, c));
            // Kp(r,c) = eval_kernel_function(data_.ptr(0, r), test.ptr(0, 0));
          }
        }
      }

      // DenseMatrix<scalar_t>  Kpt = Kp.transpose();
      // Kpt.print("Kpt", true, 11);
      // prediction.print("prediction", true, 12);
      return prediction;
    }


#if defined(STRUMPACK_USE_MPI)
    template<typename scalar_t>
    DistributedMatrix<scalar_t> Kernel<scalar_t>::fit_HSS
    (const BLACSGrid& grid, std::vector<scalar_t>& labels,
     const HSS::HSSOptions<scalar_t>& opts) {
      TaskTimer timer("HSScompression");
      auto& c = grid.Comm();
      bool verb = opts.verbose() && c.is_root();
      if (verb) std::cout << "# Compression started..." << std::endl;
      timer.start();
      std::vector<int> perm;
      HSS::HSSMatrixMPI<scalar_t> H(*this, &grid, perm, opts);
      DenseMW_t B(1, n(), labels.data(), 1);
      B.lapmt(perm, true);
      perm.clear();
      if (1) {
        const auto lvls = H.max_levels();
        const auto rank = H.max_rank();
        const auto mem = H.total_memory();
        if (c.is_root()) {
          std::cout << "# HSS compression time = "
                    << timer.elapsed() << std::endl;
          if (H.is_compressed())
            std::cout << "# created HSS matrix of dimension "
                      << H.rows() << " x " << H.cols()
                      << " with " << lvls << " levels" << std::endl
                      << "# compression succeeded!" << std::endl;
          else std::cout << "# compression failed!!!" << std::endl;
          std::cout << "# rank_H = " << rank << std::endl
                    << "# HSS_memory_MB = " << mem / 1e6 << std::endl;
        }
      }
      // // Computing error against dense matrix
      // if ( n()<= 1000 ){
      //   DenseM_t Kdense(n(),n());
      //   for(int j = 0; j < n(); j++){
      //     for(int i = 0; i < n(); i++){
      //       Kdense(i,j) = eval(i,j);
      //     }
      //   }
      //   auto HSSd_dist = H.dense();
      //   auto HSSd = HSSd_dist.all_gather();
      //   HSSd.scaled_add(-1., Kdense);
      //   if (c.is_root())
      //     std::cout << "# Compression rel error = ||HSSd-Hd||_F/||Hd||_F = " <<
      //     HSSd.normF() / Kdense.normF() << std::endl;
      // }

      if (verb)
        std::cout << "# factorization started..." << std::endl;
      timer.start();
      auto ULV = H.factor();
      if (verb)
        std::cout << "# factorization time = "
                  << timer.elapsed() << std::endl;

      if (verb)
        std::cout << "# solve started..." << std::endl;
      DenseMW_t cB(n(), 1, labels.data(), n());
      DistM_t weights(&grid, n(), 1);
      weights.scatter(cB);
      #if ITERATIVE_REFINEMENT == 1
        DistM_t rhs(weights), residual(&grid, n(), 1);
        H.solve(ULV, weights);
        auto rhs_normF = rhs.normF();
        using real_t = typename RealType<scalar_t>::value_type;
        for (int ref=0; ref<3; ref++) {
          auto residual = H.apply(weights);
          residual.scaled_add(scalar_t(-1.), rhs);
          auto rres = residual.normF() / rhs_normF;
          if (verb)
            std::cout << "||H*weights - labels||_2/||labels||_2 = "
                      << rres << std::endl;
          if (rres < 10*blas::lamch<real_t>('E')) break;
          H.solve(ULV, residual);
          weights.scaled_add(scalar_t(-1.), residual);
        }
      #else // no iterative refinement
        H.solve(ULV, weights);
      #endif
      if (verb)
        std::cout << "# HSS_solve_time = " << timer.elapsed() << std::endl;

      // DistM_t weights(&grid, 1, 1);
      return weights;
    }

    template<typename scalar_t>
    std::vector<scalar_t> Kernel<scalar_t>::predict
    (const DenseM_t& test, const DistM_t& weights) const {
      std::vector<scalar_t> prediction(test.cols());
      if (weights.active() && weights.lcols()) {
        #pragma omp parallel for
        for (std::size_t c=0; c<test.cols(); c++)
          for (std::size_t r=0; r<weights.lrows(); r++) {
            prediction[c] += weights(r, 0) *
              eval_kernel_function
              (data_.ptr(0, weights.rowl2g(r)), test.ptr(0, c));
          }
      }
      // reduce the local sums to the global vector
      weights.Comm().all_reduce
        (prediction.data(), prediction.size(), MPI_SUM);
      return prediction;
    }

    template<typename scalar_t>
    DistributedMatrix<scalar_t> Kernel<scalar_t>::fit_HSS_multiple
    (const BLACSGrid& grid, std::vector<scalar_t>& labels,
     const HSS::HSSOptions<scalar_t>& opts, std::vector<scalar_t> lambda_vec) {
      TaskTimer timer("HSScompression");
      auto& c = grid.Comm();
      bool verb = opts.verbose() && c.is_root();
      if (verb) std::cout << "# Compression started..." << std::endl;
      timer.start();
      std::vector<int> perm;
      HSS::HSSMatrixMPI<scalar_t> H(*this, &grid, perm, opts);
      DenseMW_t B(1, n(), labels.data(), 1);
      B.lapmt(perm, true);
      perm.clear();
      const auto lvls = H.max_levels();
      const auto rank = H.max_rank();
      const auto mem = H.total_memory();
      if (c.is_root()) {
        std::cout << "# HSS compression time = "
                  << timer.elapsed() << std::endl;
        if (H.is_compressed())
          std::cout << "# created HSS matrix of dimension "
                    << H.rows() << " x " << H.cols()
                    << " with " << lvls << " levels" << std::endl
                    << "# compression succeeded!" << std::endl;
        else std::cout << "# compression failed!!!" << std::endl;
        std::cout << "# rank_H = " << rank << std::endl
                  << "# HSS_memory_MB = " << mem / 1e6 << std::endl;
      }
      // Computing error against dense matrix
      if ( n()<= 500 ){
        DenseM_t Kdense(n(),n());
        for(int j = 0; j < n(); j++){
          for(int i = 0; i < n(); i++){
            Kdense(i,j) = eval(i,j);
          }
        }
        auto HSSd_dist = H.dense();
        auto HSSd = HSSd_dist.all_gather();
        HSSd.scaled_add(-1., Kdense);
        if (c.is_root())
          std::cout << "# Compression rel error = ||HSSd-Hd||_F/||Hd||_F = " <<
          HSSd.normF() / Kdense.normF() << std::endl << std::endl;
      }

      DistM_t weights(&grid, n(), 0);
      H.shift(-lambda_); // Substract original lambda
      for(auto ilambda: lambda_vec){
        if (c.is_root()){
          std::cout << std::endl;
          std::cout << "Solving for lambda = " << ilambda << std::endl;
        }
        H.shift(ilambda);

        if (c.is_root())
          std::cout << "# factorization started..." << std::endl;
        timer.start();
        auto ULV = H.factor();
        if (c.is_root())
          std::cout << "# factorization time = " << timer.elapsed() << std::endl;

        if (c.is_root())
          std::cout << "# solve started..." << std::endl;
        DenseMW_t cB(n(), 1, labels.data(), n());
        DistM_t weight(&grid, n(), 1);
        weight.scatter(cB);
        H.solve(ULV, weight);
        if (c.is_root())
          std::cout << "# HSS_solve_time = " << timer.elapsed()
                    << std::endl  << std::endl;

        weights.hconcat(weight);
        H.shift(-ilambda);
      }

      return weights;
    }

    template<typename scalar_t>
    std::vector<int> Kernel<scalar_t>::getBlockRange(int n, int size, int rank)
    const {
      std::vector<int> range;
      int lastBlock = int(std::ceil( scalar_t(n)/ scalar_t(size)));
      int sub_len;
      int sub_start;
      if ((rank == lastBlock-1) && (n % size)){ // LAST
        sub_start = rank * size;
        sub_len   = n % size;
      } else { // REST
        sub_start = rank * size;
        sub_len   = size;
      }
      range.push_back(sub_start);
      range.push_back(sub_start + sub_len);
      return range;
    }

    template<typename scalar_t>
    void Kernel<scalar_t>::getBlock(DistributedMatrix<scalar_t> &_D,
    const DenseMatrix<scalar_t> &test,
    std::vector<int>& vec_rows, std::vector<int>& vec_cols,
    MPIComm c, BLACSGrid *grid) const {
      using DistM_t = DistributedMatrix<scalar_t>;

      // MPI Element extraction
      auto Aelem = [&]
        (const std::vector<std::size_t>& I, const std::vector<std::size_t>& J,
        DistM_t& B, std::size_t rlo, std::size_t ,
        MPI_Comm comm) {
        std::vector<std::size_t> lI, lJ;
        lI.reserve(B.lrows());
        lJ.reserve(B.lcols());
        for (size_t j=0; j<J.size(); j++)
          if (B.colg2p(j) == B.pcol())
            lJ.push_back(J[j]);
        for (size_t i=0; i<I.size(); i++)
          if (B.rowg2p(i) == B.prow())
            lI.push_back(I[i]);
        auto lB = B.dense_wrapper();
        // K->eval_vec(lI, lJ, lB); // operator call

        #pragma omp parallel for collapse(2) num_threads(32)
        for (auto i=0; i<lI.size(); i++) {  // here
          for (auto j=0; j<lJ.size(); j++){
            lB(i, j) = eval_kernel_function(data_.ptr(0, lJ[j]), test.ptr(0, lI[i]));
          }
        }
      };

      int numRows = vec_rows[1] - vec_rows[0];
      int numCols = vec_cols[1] - vec_cols[0];
      std::vector<std::size_t> I, J;
      I.reserve(numRows);
      J.reserve(numCols);
      for (std::size_t i=vec_rows[0]; i<vec_rows[1]; i++)
        I.push_back(i);
      for (std::size_t j=vec_cols[0]; j<vec_cols[1]; j++)
        J.push_back(j);

      _D = DistM_t(grid, numRows, numCols);
      // Aelem(I, J, _D, 0, 0, comm);
      Aelem(I, J, _D, vec_rows[0], vec_cols[0], c.comm());
    }

    template<typename scalar_t>
    DistributedMatrix<scalar_t> Kernel<scalar_t>::predict_multiple // MPI here
    (const DenseM_t& test, DistM_t& weights,
    MPIComm c, BLACSGrid *grid) const {
      int m = test.cols();
      int LB = weights.cols();
      int NB = std::min(int(n())/2, int(n())); // m or 50
      int numb_rows = int(std::ceil( scalar_t(m)/scalar_t(LB)));
      int numb_cols = int(std::ceil( scalar_t(n())/ scalar_t(NB)));

      // WARNING: test is read transposed. test = [d x m]
      if(c.is_root()){
        std::cout <<  "       m  = " << m << std::endl;
        std::cout <<  "       n  = " << n() << std::endl;
        std::cout <<  "       LB = " << LB  << std::endl;
        std::cout <<  "       NB = " << NB << std::endl;
        std::cout <<  "numb_rows = " << numb_rows << std::endl;
        std::cout <<  "numb_cols = " << numb_cols << std::endl;
      }

      // omp_set_num_threads(32);
      // if(c.is_root()){
      //   std::cout << "omp_get_max_threads() = "
      //             << omp_get_max_threads() << std::endl;
      // }

      // Complete prediction matrix
      DistributedMatrix<scalar_t> matP(grid, m, LB);
      matP.zero();
      for(int ib = 0; ib < numb_rows; ib++){
        std::vector<int> rowRange = getBlockRange(int(m), int(LB), int(ib));
        // if (c.is_root()) std::cout << c.rank() << "_rowRange = [" << rowRange[0] << ", " << rowRange[1] << "]" << std::endl;
        // Wrapping prediction matrix
        DistributedMatrixWrapper<scalar_t> bP(rowRange[1]-rowRange[0], LB, matP, rowRange[0], 0);
        for(int jb = 0; jb < numb_cols; jb++){
          std::vector<int> colRange = getBlockRange( int(n()), int(NB), int(jb));
          // if (c.is_root()) std::cout << c.rank() << "_colRange = [" << colRange[0] << ", " << colRange[1] << "]" << std::endl;
          // Wrapping weights matrix
          DistributedMatrixWrapper<scalar_t> bW(colRange[1]-colRange[0], LB, weights, colRange[0], 0);
          // Forming block of Kp matrix (expensive step)
          DistributedMatrix<scalar_t> bKp;
          getBlock(bKp, test, rowRange, colRange, c.comm(), grid);
          // bKp.print("bKp", 11);
          // bW.print("bW",10);
          // Perform multiplication
          gemm(Trans::N, Trans::N, scalar_t(1.0), bKp, bW, scalar_t(1.0), bP);
        }
        // break;
        if(c.is_root()) std::cout << "row " << ib+1 << "/" << numb_rows << std::endl;
      }

      return matP;
    }

#if defined(STRUMPACK_USE_BPACK)
    template<typename scalar_t>
    DenseMatrix<scalar_t> Kernel<scalar_t>::fit_HODLR
    (const MPIComm& c, std::vector<scalar_t>& labels,
     const HODLR::HODLROptions<scalar_t>& opts) {
      TaskTimer timer("HODLRcompression");
      bool verb = opts.verbose() && c.is_root();
      if (verb) std::cout << "# starting HODLR compression..." << std::endl;
      timer.start();
      std::vector<int> perm;
      HODLR::HODLRMatrix<scalar_t> H(c, *this, perm, opts);
      DenseMW_t B(1, n(), labels.data(), 1);
      B.lapmt(perm, true);
      perm.clear();
      if (verb)
        std::cout << "# HODLR compression time = "
                  << timer.elapsed() << std::endl;
      timer.start();
      H.factor();
      if (verb)
        std::cout << "# factorization time = "
                  << timer.elapsed() << std::endl
                  << "# ULV_memory_MV = "
                  << H.memory()
                  << "# solution start..." << std::endl;
      int lrows = H.lrows();
      DenseMW_t lB(lrows, 1, &labels[H.begin_row()], lrows);
      DenseM_t lw(lB);
      H.solve(lB, lw);
      std::vector<int> rcnts(c.size()), displ(c.size());
      MPI_Allgather(&lrows, 1, mpi_type<int>(),
                    rcnts.data(), 1, mpi_type<int>(), c.comm());
      for (std::size_t i=1; i<displ.size(); i++)
        displ[i] = displ[i-1] + rcnts[i-1];
      DenseM_t weights(n(), 1);
      MPI_Allgatherv
        (lw.data(), lrows, mpi_type<scalar_t>(), weights.data(),
         rcnts.data(), displ.data(), mpi_type<scalar_t>(), c.comm());
      if (verb)
        std::cout << "# solve time = " << timer.elapsed() << std::endl;
      return weights;
    }
#endif
#endif

  } // end namespace kernel

} // end namespace strumpack


#endif // STRUMPACK_KERNEL_REGRESSION_HPP
