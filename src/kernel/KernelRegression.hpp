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
      std::cout << "## HSS_compression_time = "
                << timer.elapsed() << std::endl;
      DenseMW_t B(1, n(), labels.data(), 1);
      B.lapmt(perm, true);
      perm.clear();
      if (opts.verbose()) {
        // draw(H,"plot_");
        if (H.is_compressed())
          std::cout << "# created HSS matrix of dimension "
                    << H.rows() << " x " << H.cols()
                    << " with " << H.levels() << " levels" << std::endl
                    << "# compression succeeded!" << std::endl;
        else std::cout << "# compression failed!!!" << std::endl;
        std::cout << "# rank_H = " << H.rank() << std::endl
                  << "# HSS memory_H = "
                  << H.memory() / 1e6 << " MB" << std::endl
                  << "# HSS memory_counter_H = "
                  << memory_counter / 1e6 << " MB" << std::endl << std::endl
                  << "# factorization start" << std::endl;
      }
      // Computing error against dense matrix
      if ( n()<= 1000 ){
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
      timer.start();
      auto ULV = H.factor();
      if (opts.verbose())
        std::cout << "# factorization time = "
                  << timer.elapsed() << std::endl
                  << "# solution start..." << std::endl;
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
    #else
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

#endif
      if (opts.verbose())
        std::cout << "# solve time = " << timer.elapsed() << std::endl;
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


#if defined(STRUMPACK_USE_MPI)
    template<typename scalar_t>
    DistributedMatrix<scalar_t> Kernel<scalar_t>::fit_HSS
    (const BLACSGrid& grid, std::vector<scalar_t>& labels,
     const HSS::HSSOptions<scalar_t>& opts) {
      TaskTimer timer("HSScompression");
      auto& c = grid.Comm();
      bool verb = opts.verbose() && c.is_root();
      if (verb) std::cout << "# starting HSS compression..." << std::endl;
      timer.start();
      std::vector<int> perm;
      HSS::HSSMatrixMPI<scalar_t> H(*this, &grid, perm, opts);
      DenseMW_t B(1, n(), labels.data(), 1);
      B.lapmt(perm, true);
      perm.clear();
      if (opts.verbose()) {
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
                    << "# HSS memory(H) = " << mem / 1e6
                    << " MB " << std::endl << std::endl
                    << "# factorization start" << std::endl;
        }
      }
      // Computing error against dense matrix
      if ( n()<= 1000 ){
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
          HSSd.normF() / Kdense.normF() << std::endl;
      }
      timer.start();
      auto ULV = H.factor();
      if (verb)
        std::cout << "# factorization time = "
                  << timer.elapsed() << std::endl
                  << "# solution start..." << std::endl;
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
        std::cout << "# solve time = " << timer.elapsed() << std::endl;
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
