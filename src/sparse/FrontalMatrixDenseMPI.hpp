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
 * five (5) year renewals, the U.S. Government igs granted for itself
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
#ifndef FRONTAL_MATRIX_DENSE_MPI_HPP
#define FRONTAL_MATRIX_DENSE_MPI_HPP

#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include "misc/TaskTimer.hpp"
#include "misc/MPIWrapper.hpp"
#include "dense/DistributedMatrix.hpp"
#include "CompressedSparseMatrix.hpp"
#include "MatrixReordering.hpp"
#include "FrontalMatrixMPI.hpp"
#include "FrontalMatrixDense.hpp"
#include "FrontalMatrixBLR.hpp"

namespace strumpack {

  template<typename scalar_t,typename integer_t> class ExtractFront;

  template<typename scalar_t,typename integer_t>
  class FrontalMatrixDenseMPI : public FrontalMatrixMPI<scalar_t,integer_t> {
    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;
    using DenseM_t = DenseMatrix<scalar_t>;
    using DistM_t = DistributedMatrix<scalar_t>;
    using DistMW_t = DistributedMatrixWrapper<scalar_t>;
    using FMPI_t = FrontalMatrixMPI<scalar_t,integer_t>;
    using FDMPI_t = FrontalMatrixDenseMPI<scalar_t,integer_t>;
    using F_t = FrontalMatrix<scalar_t,integer_t>;
    using ExtAdd = ExtendAdd<scalar_t,integer_t>;
    template<typename _scalar_t,typename _integer_t> friend class ExtendAdd;

  public:
    FrontalMatrixDenseMPI
    (integer_t _sep, integer_t _sep_begin, integer_t _sep_end,
     std::vector<integer_t>& _upd, const MPIComm& comm, int P);
    FrontalMatrixDenseMPI(const FDMPI_t&) = delete;
    FrontalMatrixDenseMPI& operator=(FDMPI_t const&) = delete;
    ~FrontalMatrixDenseMPI() {}

    void release_work_memory() override;
    void build_front(const SpMat_t& A);
    void partial_factorization();

    void extend_add();
    void extend_add_copy_to_buffers
    (std::vector<std::vector<scalar_t>>& sbuf,
     const FMPI_t* pa) const override;

    void sample_CB
    (const SPOptions<scalar_t>& opts, const DistM_t& R, DistM_t& Sr,
     DistM_t& Sc, F_t* pa) const;

    void multifrontal_factorization
    (const SpMat_t& A, const SPOptions<scalar_t>& opts,
     int etree_level=0, int task_depth=0) override;

    void forward_multifrontal_solve
    (DenseM_t& bloc, DistM_t* bdist, DistM_t& bupd, DenseM_t& seqbupd,
     int etree_level=0) const override;
    void backward_multifrontal_solve
    (DenseM_t& yloc, DistM_t* ydist, DistM_t& yupd, DenseM_t& seqyupd,
     int etree_level=0) const override;

    void extract_CB_sub_matrix_2d
    (const std::vector<std::size_t>& I, const std::vector<std::size_t>& J,
     DistM_t& B) const override;

    std::string type() const override { return "FrontalMatrixDenseMPI"; }

  private:
    DistM_t F11_, F12_, F21_, F22_;
    std::vector<int> piv;
    using FrontalMatrixMPI<scalar_t,integer_t>::Comm;

    long long node_factor_nonzeros() const override;
  };

  template<typename scalar_t,typename integer_t>
  FrontalMatrixDenseMPI<scalar_t,integer_t>::FrontalMatrixDenseMPI
  (integer_t _sep, integer_t _sep_begin, integer_t _sep_end,
   std::vector<integer_t>& _upd, const MPIComm& comm, int P)
    : FrontalMatrixMPI<scalar_t,integer_t>
    (_sep, _sep_begin, _sep_end, _upd, comm, P) {}


  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDenseMPI<scalar_t,integer_t>::release_work_memory() {
    F22_.clear(); // remove the update block
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDenseMPI<scalar_t,integer_t>::extend_add() {
    if (!this->lchild && !this->rchild) return;
    std::vector<std::vector<scalar_t>> sbuf(this->P());
    for (auto ch : {this->lchild, this->rchild}) {
      if (ch && Comm().is_root()) {
        STRUMPACK_FLOPS
          (static_cast<long long int>(ch->dim_upd())*ch->dim_upd());
      }
      if (!this->visit(ch)) continue;
      ch->extend_add_copy_to_buffers(sbuf, this);
    }
    scalar_t *rbuf = nullptr, **pbuf = nullptr;
    all_to_all_v(sbuf, rbuf, pbuf, Comm().comm());
    for (auto ch : {this->lchild, this->rchild}) {
      if (!ch) continue;
      ch->extend_add_copy_from_buffers
        (F11_, F12_, F21_, F22_, pbuf+this->child_master(ch), this);
    }
    delete[] pbuf;
    delete[] rbuf;
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDenseMPI<scalar_t,integer_t>::extend_add_copy_to_buffers
  (std::vector<std::vector<scalar_t>>& sbuf, const FMPI_t* pa) const {
    ExtAdd::extend_add_copy_to_buffers
      (F22_, sbuf, pa, this->upd_to_parent(pa));
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDenseMPI<scalar_t,integer_t>::build_front
  (const SpMat_t& A) {
    const auto dupd = this->dim_upd();
    const auto dsep = this->dim_sep();
    if (dsep) {
      F11_ = DistM_t(this->grid(), dsep, dsep);
      using ExFront = ExtractFront<scalar_t,integer_t>;
      ExFront::extract_F11(F11_, A, this->sep_begin, dsep);
      if (this->dim_upd()) {
        F12_ = DistM_t(this->grid(), dsep, dupd);
        ExFront::extract_F12
          (F12_, A, this->sep_begin, this->sep_end, this->upd);
        F21_ = DistM_t(this->grid(), dupd, dsep);
        ExFront::extract_F21
          (F21_, A, this->sep_end, this->sep_begin, this->upd);
      }
    }
    if (dupd) {
      F22_ = DistM_t(this->grid(), dupd, dupd);
      F22_.zero();
    }
    extend_add();
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDenseMPI<scalar_t,integer_t>::partial_factorization() {
    if (this->dim_sep() && this->grid()->active()) {
#if defined(WRITE_ROOT)
      if (etree_level == 0) {
        if (Comm().is_root())
          std::cout << "Writing root node to file..." << std::endl;
        F11_.MPI_binary_write();
        if (Comm().is_root())
          std::cout << "Done. Early abort." << std::endl;
        MPI_Finalize();
        exit(0);
      }
#endif
      piv = F11_.LU();
      STRUMPACK_FULL_RANK_FLOPS(LU_flops(F11_))
      if (this->dim_upd()) {
        F12_.laswp(piv, true);
        trsm(Side::L, UpLo::L, Trans::N, Diag::U, scalar_t(1.), F11_, F12_);
        trsm(Side::R, UpLo::U, Trans::N, Diag::N, scalar_t(1.), F11_, F21_);
        gemm(Trans::N, Trans::N, scalar_t(-1.), F21_, F12_, scalar_t(1.), F22_);
        STRUMPACK_FULL_RANK_FLOPS
          (gemm_flops(Trans::N, Trans::N, scalar_t(-1.), F21_, F12_, scalar_t(1.)) +
           trsm_flops(Side::L, scalar_t(1.), F11_, F12_) +
           trsm_flops(Side::R, scalar_t(1.), F11_, F21_));
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDenseMPI<scalar_t,integer_t>::multifrontal_factorization
  (const SpMat_t& A, const SPOptions<scalar_t>& opts,
   int etree_level, int task_depth) {
    if (this->visit(this->lchild))
      this->lchild->multifrontal_factorization
        (A, opts, etree_level+1, task_depth);
    if (this->visit(this->rchild))
      this->rchild->multifrontal_factorization
        (A, opts, etree_level+1, task_depth);
    build_front(A);
    if (this->lchild) this->lchild->release_work_memory();
    if (this->rchild) this->rchild->release_work_memory();
    partial_factorization();
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDenseMPI<scalar_t,integer_t>::forward_multifrontal_solve
  (DenseM_t& bloc, DistM_t* bdist, DistM_t& bupd, DenseM_t& seqbupd,
   int etree_level) const {
    DistM_t CBl, CBr;
    DenseM_t seqCBl, seqCBr;
    if (this->visit(this->lchild))
      this->lchild->forward_multifrontal_solve
        (bloc, bdist, CBl, seqCBl, etree_level);
    if (this->visit(this->rchild))
      this->rchild->forward_multifrontal_solve
        (bloc, bdist, CBr, seqCBr, etree_level);
    DistM_t& b = bdist[this->sep];
    bupd = DistM_t(this->grid(), this->dim_upd(), b.cols());
    bupd.zero();
    this->extend_add_b(b, bupd, CBl, CBr, seqCBl, seqCBr);
    if (this->dim_sep()) {
      TIMER_TIME(TaskType::SOLVE_LOWER, 0, t_s);
      b.laswp(piv, true);
      if (b.cols() == 1) {
        trsv(UpLo::L, Trans::N, Diag::U, F11_, b);
        if (this->dim_upd())
          gemv(Trans::N, scalar_t(-1.), F21_, b, scalar_t(1.), bupd);
      } else {
        trsm(Side::L, UpLo::L, Trans::N, Diag::U, scalar_t(1.), F11_, b);
        if (this->dim_upd())
          gemm(Trans::N, Trans::N, scalar_t(-1.), F21_, b, scalar_t(1.), bupd);
      }
      TIMER_STOP(t_s);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDenseMPI<scalar_t,integer_t>::backward_multifrontal_solve
  (DenseM_t& yloc, DistM_t* ydist, DistM_t& yupd, DenseM_t& seqyupd,
   int etree_level) const {
    DistM_t& y = ydist[this->sep];
    if (this->dim_sep()) {
      TIMER_TIME(TaskType::SOLVE_UPPER, 0, t_s);
      if (y.cols() == 1) {
        if (this->dim_upd())
          gemv(Trans::N, scalar_t(-1.), F12_, yupd, scalar_t(1.), y);
        trsv(UpLo::U, Trans::N, Diag::N, F11_, y);
      } else {
        if (this->dim_upd())
          gemm(Trans::N, Trans::N, scalar_t(-1.), F12_, yupd, scalar_t(1.), y);
        trsm(Side::L, UpLo::U, Trans::N, Diag::N, scalar_t(1.), F11_, y);
      }
      TIMER_STOP(t_s);
    }
    DistM_t CBl, CBr;
    DenseM_t seqCBl, seqCBr;
    this->extract_b(y, yupd, CBl, CBr, seqCBl, seqCBr);
    if (this->visit(this->lchild))
      this->lchild->backward_multifrontal_solve
        (yloc, ydist, CBl, seqCBl, etree_level);
    if (this->visit(this->rchild))
      this->rchild->backward_multifrontal_solve
        (yloc, ydist, CBr, seqCBr, etree_level);
  }

  /**
   * Note that B should be defined on the same context as used in this
   * front. This simplifies communication.
   */
  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDenseMPI<scalar_t,integer_t>::extract_CB_sub_matrix_2d
  (const std::vector<std::size_t>& I,
   const std::vector<std::size_t>& J, DistM_t& B) const {
    if (Comm().is_null()) return;
    std::vector<std::size_t> lJ, oJ, lI, oI;
    this->find_upd_indices(J, lJ, oJ);
    this->find_upd_indices(I, lI, oI);
    std::vector<std::vector<scalar_t>> sbuf(this->P());
    ExtAdd::extract_copy_to_buffers(F22_, lI, lJ, oI, oJ, B, sbuf);
    scalar_t *rbuf = nullptr, **pbuf = nullptr;
    all_to_all_v(sbuf, rbuf, pbuf, Comm().comm());
    ExtAdd::extract_copy_from_buffers(B, lI, lJ, oI, oJ, F22_, pbuf);
    delete[] rbuf;
    delete[] pbuf;
  }

  /**
   *  Sr = F22 * R
   *  Sc = F22^* * R
   */
  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDenseMPI<scalar_t,integer_t>::sample_CB
  (const SPOptions<scalar_t>& opts, const DistM_t& R, DistM_t& Sr, DistM_t& Sc,
   FrontalMatrix<scalar_t,integer_t>* pa) const {
    if (F11_.active() || F22_.active()) {
      auto b = R.cols();
      Sr = DistM_t(this->grid(), this->dim_upd(), b);
      Sc = DistM_t(this->grid(), this->dim_upd(), b);
      gemm(Trans::N, Trans::N, scalar_t(1.), F22_, R, scalar_t(0.), Sr);
      gemm(Trans::C, Trans::N, scalar_t(1.), F22_, R, scalar_t(0.), Sc);
      STRUMPACK_CB_SAMPLE_FLOPS
        (gemm_flops(Trans::N, Trans::N, scalar_t(1.), F22_, R, scalar_t(0.)) +
         gemm_flops(Trans::C, Trans::N, scalar_t(1.), F22_, R, scalar_t(0.)));
    }
  }

  template<typename scalar_t,typename integer_t> long long
  FrontalMatrixDenseMPI<scalar_t,integer_t>::node_factor_nonzeros() const {
    return F11_.nonzeros() + F12_.nonzeros() + F21_.nonzeros();
  }

} // end namespace strumpack

#endif //FRONTAL_MATRIX_DENSE_MPI_HPP
