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

#if defined(STRUMPACK_USE_SLATE_SCALAPACK)
#include <slate.hh>
#endif

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
    (integer_t sep, integer_t sep_begin, integer_t sep_end,
     std::vector<integer_t>& upd, const MPIComm& comm, int P);
    FrontalMatrixDenseMPI(const FDMPI_t&) = delete;
    FrontalMatrixDenseMPI& operator=(FDMPI_t const&) = delete;

    void release_work_memory() override;
    void build_front(const SpMat_t& A);
    void partial_factorization();

    void extend_add();
    void extend_add_copy_to_buffers
    (std::vector<std::vector<scalar_t>>& sbuf, const FMPI_t* pa) const override {
      ExtAdd::extend_add_copy_to_buffers(F22_, sbuf, pa, this->upd_to_parent(pa));
    }

    void sample_CB
    (const DistM_t& R, DistM_t& Sr, DistM_t& Sc, F_t* pa) const override;
    void sample_CB
    (Trans op, const DistM_t& R, DistM_t& S,
     FrontalMatrix<scalar_t,integer_t>* pa) const override;

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
#if defined(STRUMPACK_USE_SLATE_SCALAPACK)
    slate::Matrix<scalar_t> slateF11_, slateF12_, slateF21_, slateF22_;
    slate::Pivots slate_piv_;
    std::map<slate::Option, slate::Value> slate_opts_;
#endif

    long long node_factor_nonzeros() const override;

    using FrontalMatrix<scalar_t,integer_t>::lchild_;
    using FrontalMatrix<scalar_t,integer_t>::rchild_;
    using FrontalMatrixMPI<scalar_t,integer_t>::visit;
    using FrontalMatrixMPI<scalar_t,integer_t>::grid;
    using FrontalMatrixMPI<scalar_t,integer_t>::Comm;
  };

  template<typename scalar_t,typename integer_t>
  FrontalMatrixDenseMPI<scalar_t,integer_t>::FrontalMatrixDenseMPI
  (integer_t sep, integer_t sep_begin, integer_t sep_end,
   std::vector<integer_t>& upd, const MPIComm& comm, int P)
    : FrontalMatrixMPI<scalar_t,integer_t>
    (sep, sep_begin, sep_end, upd, comm, P) {}


  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDenseMPI<scalar_t,integer_t>::release_work_memory() {
    F22_.clear(); // remove the update block
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDenseMPI<scalar_t,integer_t>::extend_add() {
    if (!lchild_ && !rchild_) return;
    std::vector<std::vector<scalar_t>> sbuf(this->P());
    for (auto& ch : {lchild_.get(), rchild_.get()}) {
      if (ch && Comm().is_root()) {
        STRUMPACK_FLOPS
          (static_cast<long long int>(ch->dim_upd())*ch->dim_upd());
      }
      if (!visit(ch)) continue;
      ch->extend_add_copy_to_buffers(sbuf, this);
    }
    std::vector<scalar_t> rbuf;
    std::vector<scalar_t*> pbuf;
    Comm().all_to_all_v(sbuf, rbuf, pbuf);
    for (auto& ch : {lchild_.get(), rchild_.get()}) {
      if (!ch) continue;
      ch->extend_add_copy_from_buffers
        (F11_, F12_, F21_, F22_, pbuf.data()+this->master(ch), this);
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDenseMPI<scalar_t,integer_t>::build_front
  (const SpMat_t& A) {
    const auto dupd = this->dim_upd();
    const auto dsep = this->dim_sep();
    if (dsep) {
      F11_ = DistM_t(grid(), dsep, dsep);
      using ExFront = ExtractFront<scalar_t,integer_t>;
      ExFront::extract_F11(F11_, A, this->sep_begin_, dsep);
      if (this->dim_upd()) {
        F12_ = DistM_t(grid(), dsep, dupd);
        ExFront::extract_F12
          (F12_, A, this->sep_begin_, this->sep_end_, this->upd_);
        F21_ = DistM_t(grid(), dupd, dsep);
        ExFront::extract_F21
          (F21_, A, this->sep_end_, this->sep_begin_, this->upd_);
      }
    }
    if (dupd) {
      F22_ = DistM_t(grid(), dupd, dupd);
      F22_.zero();
    }
    extend_add();
#if defined(STRUMPACK_USE_SLATE_SCALAPACK)
    if (dsep) {
      slateF11_ = slate::Matrix<scalar_t>::fromScaLAPACK
        (F11_.rows(), F11_.cols(), F11_.data(), F11_.ld(),
         F11_.MB(), F11_.nprows(), F11_.npcols(), F11_.comm());
      if (dupd) {
        slateF12_ = slate::Matrix<scalar_t>::fromScaLAPACK
          (F12_.rows(), F12_.cols(), F12_.data(), F12_.ld(),
           F12_.MB(), F12_.nprows(), F12_.npcols(), F12_.comm());
        slateF21_ = slate::Matrix<scalar_t>::fromScaLAPACK
          (F21_.rows(), F21_.cols(), F21_.data(), F21_.ld(),
           F21_.MB(), F21_.nprows(), F21_.npcols(), F21_.comm());
      }
    }
    if (dupd)
      slateF22_ = slate::Matrix<scalar_t>::fromScaLAPACK
        (F22_.rows(), F22_.cols(), F22_.data(), F22_.ld(),
         F22_.MB(), F22_.nprows(), F22_.npcols(), F22_.comm());
#endif
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDenseMPI<scalar_t,integer_t>::partial_factorization() {
    if (this->dim_sep() && grid()->active()) {
#if defined(STRUMPACK_USE_SLATE_SCALAPACK)
      std::map<slate::Option, slate::Value> slate_opts = {{slate::Option::Target, slate::Target::Devices}};
      slate::getrf(slateF11_, slate_piv_, slate_opts);
#else
      piv = F11_.LU();
#endif
      STRUMPACK_FULL_RANK_FLOPS(LU_flops(F11_));
      if (this->dim_upd()) {
#if defined(STRUMPACK_USE_SLATE_SCALAPACK)
        // F12_.laswp(piv, true);

        // slate::TriangularMatrix<scalar_t> F11_L
        //   (slate::Uplo::Lower, slate::Diag::Unit, slateF11_);
        // slate::trsm(slate::Side::Left, scalar_t(1.), F11_L, slateF12_, slate_opts);
        // slate::TriangularMatrix<scalar_t> F11_U
        //   (slate::Uplo::Upper, slate::Diag::NonUnit, slateF11_);
        // slate::trsm(slate::Side::Right, scalar_t(1.), F11_U, slateF21_, slate_opts);

	slate::getrs(slateF11_, slate_piv_, slateF12_);
	slate::gemm(scalar_t(-1.), slateF21_, slateF12_,
                    scalar_t(1.), slateF22_, slate_opts);
#else
        F12_.laswp(piv, true);
        trsm(Side::L, UpLo::L, Trans::N, Diag::U, scalar_t(1.), F11_, F12_);
        trsm(Side::R, UpLo::U, Trans::N, Diag::N, scalar_t(1.), F11_, F21_);
        gemm(Trans::N, Trans::N, scalar_t(-1.), F21_, F12_, scalar_t(1.), F22_);
#endif
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
    if (visit(lchild_))
      lchild_->multifrontal_factorization(A, opts, etree_level+1, task_depth);
    if (visit(rchild_))
      rchild_->multifrontal_factorization(A, opts, etree_level+1, task_depth);
    build_front(A);
    if (etree_level == 0 && opts.write_root_front()) {
      F11_.print_to_files("Froot");
      Comm().barrier();
    }
    if (lchild_) lchild_->release_work_memory();
    if (rchild_) rchild_->release_work_memory();
    partial_factorization();
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDenseMPI<scalar_t,integer_t>::forward_multifrontal_solve
  (DenseM_t& bloc, DistM_t* bdist, DistM_t& bupd, DenseM_t& seqbupd,
   int etree_level) const {
    DistM_t CBl, CBr;
    DenseM_t seqCBl, seqCBr;
    if (visit(lchild_))
      lchild_->forward_multifrontal_solve
        (bloc, bdist, CBl, seqCBl, etree_level);
    if (visit(rchild_))
      rchild_->forward_multifrontal_solve
        (bloc, bdist, CBr, seqCBr, etree_level);
    DistM_t& b = bdist[this->sep_];
    bupd = DistM_t(grid(), this->dim_upd(), b.cols());
    bupd.zero();
    this->extend_add_b(b, bupd, CBl, CBr, seqCBl, seqCBr);
#if defined(STRUMPACK_USE_SLATE_SCALAPACK)
    if (this->dim_sep()) {
      auto sbloc = slate::Matrix<scalar_t>::fromScaLAPACK
        (b.rows(), b.cols(), b.data(), b.ld(),
         b.MB(), b.nprows(), b.npcols(), b.comm());
      slate::getrs(const_cast<slate::Matrix<scalar_t>&>(slateF11_),
                   const_cast<slate::Pivots&>(slate_piv_), sbloc, slate_opts_);
      if (this->dim_upd()) {
        auto sbupd = slate::Matrix<scalar_t>::fromScaLAPACK
          (bupd.rows(), bupd.cols(), bupd.data(), bupd.ld(),
           bupd.MB(), bupd.nprows(), bupd.npcols(), bupd.comm());
	slate::gemm
          (scalar_t(-1.), const_cast<slate::Matrix<scalar_t>&>(slateF21_),
           sbloc, scalar_t(1.), sbupd, slate_opts_);
      }
    }
#else
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
#endif
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDenseMPI<scalar_t,integer_t>::backward_multifrontal_solve
  (DenseM_t& yloc, DistM_t* ydist, DistM_t& yupd, DenseM_t& seqyupd,
   int etree_level) const {
    DistM_t& y = ydist[this->sep_];
#if defined(STRUMPACK_USE_SLATE_SCALAPACK)
    if (this->dim_sep() && this->dim_upd()) {
      auto sy = slate::Matrix<scalar_t>::fromScaLAPACK
        (y.rows(), y.cols(), y.data(), y.ld(), y.MB(),
         y.nprows(), y.npcols(), y.comm());
      auto syupd = slate::Matrix<scalar_t>::fromScaLAPACK
        (yupd.rows(), yupd.cols(), yupd.data(), yupd.ld(), yupd.MB(),
         yupd.nprows(), yupd.npcols(), yupd.comm());
      slate::gemm
        (scalar_t(-1.), const_cast<slate::Matrix<scalar_t>&>(slateF12_),
         syupd, scalar_t(1.), sy, slate_opts_);
    }
#else
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
#endif
    DistM_t CBl, CBr;
    DenseM_t seqCBl, seqCBr;
    this->extract_b(y, yupd, CBl, CBr, seqCBl, seqCBr);
    if (visit(lchild_))
      lchild_->backward_multifrontal_solve
        (yloc, ydist, CBl, seqCBl, etree_level);
    if (visit(rchild_))
      rchild_->backward_multifrontal_solve
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
    std::vector<scalar_t> rbuf;
    std::vector<scalar_t*> pbuf;
    Comm().all_to_all_v(sbuf, rbuf, pbuf);
    ExtAdd::extract_copy_from_buffers(B, lI, lJ, oI, oJ, F22_, pbuf);
  }

  /**
   *  Sr = F22 * R
   *  Sc = F22^* * R
   */
  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDenseMPI<scalar_t,integer_t>::sample_CB
  (const DistM_t& R, DistM_t& Sr, DistM_t& Sc,
   FrontalMatrix<scalar_t,integer_t>* pa) const {
    if (F11_.active() || F22_.active()) {
      auto b = R.cols();
      Sr = DistM_t(grid(), this->dim_upd(), b);
      Sc = DistM_t(grid(), this->dim_upd(), b);
      gemm(Trans::N, Trans::N, scalar_t(1.), F22_, R, scalar_t(0.), Sr);
      gemm(Trans::C, Trans::N, scalar_t(1.), F22_, R, scalar_t(0.), Sc);
      STRUMPACK_CB_SAMPLE_FLOPS
        (gemm_flops(Trans::N, Trans::N, scalar_t(1.), F22_, R, scalar_t(0.)) +
         gemm_flops(Trans::C, Trans::N, scalar_t(1.), F22_, R, scalar_t(0.)));
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontalMatrixDenseMPI<scalar_t,integer_t>::sample_CB
  (Trans op, const DistM_t& R, DistM_t& S,
   FrontalMatrix<scalar_t,integer_t>* pa) const {
    if (F11_.active() || F22_.active()) {
      auto b = R.cols();
      S = DistM_t(grid(), this->dim_upd(), b);
      TIMER_TIME(TaskType::F22_MULT, 1, t_f22mult);
      gemm(op, Trans::N, scalar_t(1.), F22_, R, scalar_t(0.), S);
      TIMER_STOP(t_f22mult);
      STRUMPACK_CB_SAMPLE_FLOPS
        (gemm_flops(op, Trans::N, scalar_t(1.), F22_, R, scalar_t(0.)));
    }
  }

  template<typename scalar_t,typename integer_t> long long
  FrontalMatrixDenseMPI<scalar_t,integer_t>::node_factor_nonzeros() const {
    return F11_.nonzeros() + F12_.nonzeros() + F21_.nonzeros();
  }

} // end namespace strumpack

#endif //FRONTAL_MATRIX_DENSE_MPI_HPP
