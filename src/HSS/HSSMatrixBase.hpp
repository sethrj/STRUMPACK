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
/**
 * \file HSSMatrixBase.hpp
 * \brief This file contains the HSSMatrixBase class definition, an
 * abstract class for HSS matrix representation.
 */
#ifndef HSS_MATRIX_BASE_HPP
#define HSS_MATRIX_BASE_HPP

#include <cassert>
#include <iostream>

#include "dense/DenseMatrix.hpp"
#include "HSSOptions.hpp"
#if defined(STRUMPACK_USE_MPI)
#include "dense/DistributedMatrix.hpp"
#include "HSSExtraMPI.hpp"
#include "DistSamples.hpp"
#include "HSSMatrixMPI.hpp"
#include "DistElemMult.hpp"
#endif

namespace strumpack {
  namespace HSS {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
    template<typename scalar_t> class HSSMatrix;
#if defined(STRUMPACK_USE_MPI)
    template<typename scalar_t> class HSSMatrixMPI;
    template<typename scalar_t> class DistSubLeaf;
    template<typename scalar_t> class DistSamples;
#endif //defined(STRUMPACK_USE_MPI)
#endif //DOXYGEN_SHOULD_SKIP_THIS


    /**
     * \class HSSMatrixBase
     *
     * \brief Abstract base class for Hierarchically Semi-Separable
     * (HSS) matrices.
     *
     * This is for non-symmetric HSS matrices, but can be used with
     * symmetric matrices as well.
     *
     * \tparam scalar_t Can be float, double, std:complex<float> or
     * std::complex<double>.
     *
     * \see HSSMatrix, HSSMatrixMPI
     */
    template<typename scalar_t> class HSSMatrixBase {
      using real_t = typename RealType<scalar_t>::value_type;
      using DenseM_t = DenseMatrix<scalar_t>;
      using DenseMW_t = DenseMatrixWrapper<scalar_t>;
      using elem_t = typename std::function
        <void(const std::vector<std::size_t>& I,
              const std::vector<std::size_t>& J, DenseM_t& B)>;
      using opts_t = HSSOptions<scalar_t>;
#if defined(STRUMPACK_USE_MPI)
      using DistM_t = DistributedMatrix<scalar_t>;
      using DistMW_t = DistributedMatrixWrapper<scalar_t>;
      using delem_t = typename std::function
        <void(const std::vector<std::size_t>& I,
              const std::vector<std::size_t>& J, DistM_t& B)>;
#endif //defined(STRUMPACK_USE_MPI)

    public:
      /**
       * Construct an m x n HSS matrix, not initialized.
       *
       * \param m number of rows
       * \param n number of columns
       * \param active Denote if this matrix (node) is active on this
       * process.
       */
      HSSMatrixBase(std::size_t m, std::size_t n, bool active);

      /**
       * Default virtual destructor.
       */
      virtual ~HSSMatrixBase() = default;

      /**
       * Copy constructor.
       * \param other HSS matrix to copy.
       */
      HSSMatrixBase(const HSSMatrixBase<scalar_t>& other);

      /**
       * Copy assignment operator, makes a deep copy.
       * \param other HSS matrix to copy.
       * \return reference to this HSS matrix.
       */
      HSSMatrixBase<scalar_t>& operator=(const HSSMatrixBase<scalar_t>& other);

      /**
       * Move constructor.
       * \param h HSS matrix to move from, h will be emptied.
       */
      HSSMatrixBase(HSSMatrixBase&& h) = default;

      /**
       * Move assignment operator.
       * \param h HSS matrix to move from, h will be emptied.
       * \return reference to this HSS matrix.
       */
      HSSMatrixBase& operator=(HSSMatrixBase&& h) = default;

      /**
       * Clone this HSS matrix.
       * TODO delete this??
       *
       * \return std::unique_ptr to a clone of this HSS matrix.
       */
      virtual std::unique_ptr<HSSMatrixBase<scalar_t>> clone() const = 0;

      /**
       * Returns the dimensions of this HSS matrix, as a pair.
       *
       * \return pair with number of rows and columns of this HSS
       * matrix.
       */
      std::pair<std::size_t,std::size_t> dims() const {
        return std::make_pair(_rows, _cols);
      }

      /**
       * Return the number of rows in this HSS matrix.
       * \return number of rows
       */
      std::size_t rows() const { return _rows; }

      /**
       * Return the number of columns in this HSS matrix.
       * \return number of columns
       */
      std::size_t cols() const { return _cols; }

      /**
       * Check whether this node of the HSS tree is a leaf.
       * \return true if this node is a leaf, false otherwise.
       */
      bool leaf() const { return _ch.empty(); }

      /**
       * Return a const reference to the child (0, or 1) of this HSS
       * matrix. This is only valid when !this->leaf(). It is assumed
       * that a non-leaf node always has exactly 2 children.
       *
       * \param c Number of the child, should be 0 or 1, for the left
       * or the right child.
       * \return Const reference to the child (HSSMatrixBase).
       */
      const HSSMatrixBase<scalar_t>& child(int c) const {
        assert(c>=0 && c<_ch.size()); return *(_ch[c]);
      }

      /**
       * Return a reference to the child (0, or 1) of this HSS
       * matrix. This is only valid when !this->leaf(). It is assumed
       * that a non-leaf node always has exactly 2 children.
       *
       * \param c Number of the child, should be 0 or 1, for the left
       * or the right child.
       * \return Reference to the child (HSSMatrixBase).
       */
      HSSMatrixBase<scalar_t>& child(int c) {
        assert(c>=0 && c<_ch.size()); return *(_ch[c]);
      }

      /**
       * Check whether the HSS matrix was compressed.
       *
       * \return True if this HSS matrix was succesfully compressed,
       * false otherwise.
       *
       * \see is_untouched
       */
      bool is_compressed() const {
        return _U_state == State::COMPRESSED &&
          _V_state == State::COMPRESSED;
      }

      /**
       * Check whether the HSS compression was started for this
       * matrix.
       *
       * \return True if HSS compression was not started yet, false
       * otherwise. False may mean that compression was started but
       * failed, or that compression succeeded.
       *
       * \see is_compressed
       */
      bool is_untouched() const {
        return _U_state == State::UNTOUCHED &&
          _V_state == State::UNTOUCHED;
      }

      /**
       * Check if this HSS matrix (or node in the HSS tree) is active
       * on this rank.
       *
       * \return True if this node is active, false otherwise.
       */
      bool active() const { return _active; };

      /**
       * Return the maximum rank of this HSS matrix over all
       * off-diagonal blocks, and all levels.
       *
       * \return Maximum HSS rank.
       */
      virtual std::size_t rank() const = 0;

      /**
       * Return the total amount of memory used by this HSS matrix, in
       * bytes. This counts the memory for the low rank basis matrices
       * as well as some of the memory used for the data structures.
       *
       * \return Memory usage in bytes.
       * \see nonzeros
       */
      virtual std::size_t memory() const = 0;

      /**
       * Return the total number of nonzeros stored in the HSS
       * generator matrices (D,U,V at leafs U,B,V at non-leafs).
       *
       * \return Nonzeros in the HSS representation.
       * \see memory
       */
      virtual std::size_t nonzeros() const = 0;

      /**
       * Return the number of levels in the HSS matrix.
       *
       * \return Number of HSS levels (>= 1).
       */
      virtual std::size_t levels() const = 0;

      /**
       * Print info about this HSS matrix, such as tree info, ranks,
       * etc.
       *
       * \param out Stream to print to, defaults to std::cout
       * \param roff Row offset of top left corner, defaults to
       * 0. This is used to recursively print the tree, you can leave
       * this at the default.
       * \param coff Column offset of top left corner, defaults to
       * 0. This is used to recursively print the tree, you can leave
       * this at the default.
       */
      virtual void print_info
      (std::ostream &out=std::cout,
       std::size_t roff=0, std::size_t coff=0) const = 0;

      /**
       * Set the depth of openmp nested tasks. This can be used to
       * limit the number of tasks to spawn in the HSS routines, which
       * is can reduce task creation overhead.  This is used in the
       * sparse solver when multiple HSS matrices are created from
       * within multiple openmp tasks. The HSS routines all use openmp
       * tasking to traverse the HSS tree and for parallelism within
       * the HSS nodes as well.
       */
      void set_openmp_task_depth(int depth) { _openmp_task_depth = depth; }

#ifndef DOXYGEN_SHOULD_SKIP_THIS
      virtual void delete_trailing_block() { if (_ch.size()==2) _ch.resize(1); }
      virtual void reset() {
        _U_state = _V_state = State::UNTOUCHED;
        _U_rank = _U_rows = _V_rank = _V_rows = 0;
        for (auto& c : _ch) c->reset();
      }
#endif

      /**
       * APply a shift to the diagonal of this matrix. Ie, this +=
       * sigma * I, with I the identity matrix. Call this after
       * compression.
       *
       * \param sigma Shift to be applied to the diagonal.
       */
      virtual void shift(scalar_t sigma) = 0;

      /**
       * Internal routine to draw this HSS matrix. Do not use this
       * directly. Use HSS::draw.
       *
       * \see HSS::draw
       */
      virtual void draw
      (std::ostream& of, std::size_t rlo, std::size_t clo) const {};

#if defined(STRUMPACK_USE_MPI)
      virtual void forward_solve
      (const HSSFactorsMPI<scalar_t>& ULV, WorkSolveMPI<scalar_t>& w,
       const DistM_t& b, bool partial) const;
      virtual void backward_solve
      (const HSSFactorsMPI<scalar_t>& ULV, WorkSolveMPI<scalar_t>& w,
       DistM_t& x) const;

      virtual const BLACSGrid* grid() const { return nullptr; }
      virtual const BLACSGrid* grid(const BLACSGrid* local_grid) const {
        return active() ? local_grid : nullptr;
      }
      virtual const BLACSGrid* grid_local() const { return nullptr; }
      virtual int Ptotal() const { return 1; }
      virtual int Pactive() const { return 1; }

      virtual void to_block_row
      (const DistM_t& A, DenseM_t& sub_A, DistM_t& leaf_A) const;
      virtual void allocate_block_row
      (int d, DenseM_t& sub_A, DistM_t& leaf_A) const;
      virtual void from_block_row
      (DistM_t& A, const DenseM_t& sub_A, const DistM_t& leaf_A,
       const BLACSGrid* lg) const;
#endif //defined(STRUMPACK_USE_MPI)

    protected:
      std::size_t _rows, _cols;

      // TODO store children array in the sub-class???
      std::vector<std::unique_ptr<HSSMatrixBase<scalar_t>>> _ch;
      State _U_state, _V_state;
      int _openmp_task_depth;
      bool _active;

      int _U_rank = 0, _U_rows = 0, _V_rank = 0, _V_rows = 0;
      virtual std::size_t U_rank() const { return _U_rank; }
      virtual std::size_t V_rank() const { return _V_rank; }
      virtual std::size_t U_rows() const { return _U_rows; };
      virtual std::size_t V_rows() const { return _V_rows; };

      // Used to redistribute the original 2D block cyclic matrix
      // according to the HSS tree
      DenseM_t _Asub;

      virtual void compress_recursive_original
      (DenseM_t& Rr, DenseM_t& Rc, DenseM_t& Sr, DenseM_t& Sc,
       const elem_t& Aelem, const opts_t& opts, WorkCompress<scalar_t>& w,
       int dd, int depth) {};
      virtual void compress_recursive_stable
      (DenseM_t& Rr, DenseM_t& Rc, DenseM_t& Sr, DenseM_t& Sc,
       const elem_t& Aelem, const opts_t& opts, WorkCompress<scalar_t>& w,
       int d, int dd, int depth) {};
      virtual void compress_level_original
      (DenseM_t& Rr, DenseM_t& Rc, DenseM_t& Sr, DenseM_t& Sc,
       const opts_t& opts, WorkCompress<scalar_t>& w,
       int dd, int lvl, int depth) {}
      virtual void compress_level_stable
      (DenseM_t& Rr, DenseM_t& Rc, DenseM_t& Sr, DenseM_t& Sc,
       const opts_t& opts, WorkCompress<scalar_t>& w,
       int d, int dd, int lvl, int depth) {}
      virtual void compress_recursive_ann
      (DenseMatrix<std::uint32_t>& ann, DenseMatrix<real_t>& scores,
       const elem_t& Aelem, const opts_t& opts,
       WorkCompressANN<scalar_t>& w, int depth) {}

      virtual void get_extraction_indices
      (std::vector<std::vector<std::size_t>>& I,
       std::vector<std::vector<std::size_t>>& J,
       const std::pair<std::size_t,std::size_t>& off,
       WorkCompress<scalar_t>& w, int& self, int lvl) {}

      virtual void get_extraction_indices
      (std::vector<std::vector<std::size_t>>& I,
       std::vector<std::vector<std::size_t>>& J,
       std::vector<DenseM_t*>& B,
       const std::pair<std::size_t,std::size_t>& off,
       WorkCompress<scalar_t>& w, int& self, int lvl) {}
      virtual void extract_D_B
      (const elem_t& Aelem, const opts_t& opts,
       WorkCompress<scalar_t>& w, int lvl) {}

      virtual real_t update_orthogonal_basis
      (DenseM_t& S, int d, int dd, int depth) { return real_t(0.); }

      virtual void factor_recursive
      (HSSFactors<scalar_t>& ULV, WorkFactor<scalar_t>& w,
       bool isroot, bool partial, int depth) const {};

      virtual void apply_fwd
      (const DenseM_t& b, WorkApply<scalar_t>& w,
       bool isroot, int depth, std::atomic<long long int>& flops) const {};
      virtual void apply_bwd
      (const DenseM_t& b, scalar_t beta, DenseM_t& c, WorkApply<scalar_t>& w,
       bool isroot, int depth, std::atomic<long long int>& flops) const {};
      virtual void applyT_fwd
      (const DenseM_t& b, WorkApply<scalar_t>& w, bool isroot,
       int depth, std::atomic<long long int>& flops) const {};
      virtual void applyT_bwd
      (const DenseM_t& b, scalar_t beta, DenseM_t& c, WorkApply<scalar_t>& w,
       bool isroot, int depth, std::atomic<long long int>& flops) const {};

      virtual void forward_solve
      (const HSSFactors<scalar_t>& ULV, WorkSolve<scalar_t>& w,
       const DenseMatrix<scalar_t>& b, bool partial) const {};
      virtual void backward_solve
      (const HSSFactors<scalar_t>& ULV, WorkSolve<scalar_t>& w,
       DenseMatrix<scalar_t>& b) const {};
      virtual void solve_fwd
      (const HSSFactors<scalar_t>& ULV, const DenseM_t& b,
       WorkSolve<scalar_t>& w, bool partial, bool isroot, int depth) const {};
      virtual void solve_bwd
      (const HSSFactors<scalar_t>& ULV, DenseM_t& x, WorkSolve<scalar_t>& w,
       bool isroot, int depth) const {};

      virtual void extract_fwd
      (WorkExtract<scalar_t>& w, bool odiag, int depth) const {};
      virtual void extract_bwd
      (DenseMatrix<scalar_t>& B, WorkExtract<scalar_t>& w,
       int depth) const {};
      virtual void extract_bwd
      (std::vector<Triplet<scalar_t>>& triplets,
       WorkExtract<scalar_t>& w, int depth) const {};

      virtual void apply_UV_big
      (DenseM_t& Theta, DenseM_t& Uop, DenseM_t& Phi, DenseM_t& Vop,
       const std::pair<std::size_t, std::size_t>& offset, int depth,
       std::atomic<long long int>& flops) const {};
      virtual void apply_UtVt_big
      (const DenseM_t& A, DenseM_t& UtA, DenseM_t& VtA,
       const std::pair<std::size_t, std::size_t>& offset,
       int depth, std::atomic<long long int>& flops) const {};

      virtual void dense_recursive
      (DenseM_t& A, WorkDense<scalar_t>& w, bool isroot, int depth) const {};

      friend class HSSMatrix<scalar_t>;

#if defined(STRUMPACK_USE_MPI)
      using delemw_t = typename std::function
        <void(const std::vector<std::size_t>& I,
              const std::vector<std::size_t>& J,
              DistM_t& B, DistM_t& A,
              std::size_t rlo, std::size_t clo,
              MPI_Comm comm)>;

      virtual void compress_recursive_original
      (DistSamples<scalar_t>& RS, const delemw_t& Aelem,
       const opts_t& opts, WorkCompressMPI<scalar_t>& w, int dd);
      virtual void compress_recursive_stable
      (DistSamples<scalar_t>& RS, const delemw_t& Aelem,
       const opts_t& opts, WorkCompressMPI<scalar_t>& w, int d, int dd);
      virtual void compress_level_original
      (DistSamples<scalar_t>& RS, const opts_t& opts,
       WorkCompressMPI<scalar_t>& w, int dd, int lvl);
      virtual void compress_level_stable
      (DistSamples<scalar_t>& RS, const opts_t& opts,
       WorkCompressMPI<scalar_t>& w, int d, int dd, int lvl);
      virtual void compress_recursive_ann
      (DenseMatrix<std::uint32_t>& ann, DenseMatrix<real_t>& scores,
       const delemw_t& Aelem, WorkCompressMPIANN<scalar_t>& w,
       const opts_t& opts, const BLACSGrid* lg);

      virtual void get_extraction_indices
      (std::vector<std::vector<std::size_t>>& I,
       std::vector<std::vector<std::size_t>>& J,
       WorkCompressMPI<scalar_t>& w, int& self, int lvl);
      virtual void get_extraction_indices
      (std::vector<std::vector<std::size_t>>& I,
       std::vector<std::vector<std::size_t>>& J, std::vector<DistMW_t>& B,
       const BLACSGrid* lg, WorkCompressMPI<scalar_t>& w, int& self, int lvl);
      virtual void extract_D_B
      (const delemw_t& Aelem, const BLACSGrid* lg, const opts_t& opts,
       WorkCompressMPI<scalar_t>& w, int lvl);

      virtual void apply_fwd
      (const DistSubLeaf<scalar_t>& B, WorkApplyMPI<scalar_t>& w,
       bool isroot, long long int flops) const;
      virtual void apply_bwd
      (const DistSubLeaf<scalar_t>& B, scalar_t beta,
       DistSubLeaf<scalar_t>& C, WorkApplyMPI<scalar_t>& w,
       bool isroot, long long int flops) const;
      virtual void applyT_fwd
      (const DistSubLeaf<scalar_t>& B, WorkApplyMPI<scalar_t>& w,
       bool isroot, long long int flops) const;
      virtual void applyT_bwd
      (const DistSubLeaf<scalar_t>& B, scalar_t beta,
       DistSubLeaf<scalar_t>& C, WorkApplyMPI<scalar_t>& w,
       bool isroot, long long int flops) const;

      virtual void factor_recursive
      (HSSFactorsMPI<scalar_t>& ULV, WorkFactorMPI<scalar_t>& w,
       const BLACSGrid* lg, bool isroot, bool partial) const;

      virtual void solve_fwd
      (const HSSFactorsMPI<scalar_t>& ULV, const DistSubLeaf<scalar_t>& b,
       WorkSolveMPI<scalar_t>& w, bool partial, bool isroot) const;
      virtual void solve_bwd
      (const HSSFactorsMPI<scalar_t>& ULV, DistSubLeaf<scalar_t>& x,
       WorkSolveMPI<scalar_t>& w, bool isroot) const;

      virtual void extract_fwd
      (WorkExtractMPI<scalar_t>& w, const BLACSGrid* lg, bool odiag) const;
      virtual void extract_bwd
      (std::vector<Triplet<scalar_t>>& triplets,
       const BLACSGrid* lg, WorkExtractMPI<scalar_t>& w) const;
      virtual void extract_fwd
      (WorkExtractBlocksMPI<scalar_t>& w, const BLACSGrid* lg,
       std::vector<bool>& odiag) const;
      virtual void extract_bwd
      (std::vector<std::vector<Triplet<scalar_t>>>& triplets,
       const BLACSGrid* lg, WorkExtractBlocksMPI<scalar_t>& w) const;

      virtual void apply_UV_big
      (DistSubLeaf<scalar_t>& Theta, DistM_t& Uop,
       DistSubLeaf<scalar_t>& Phi, DistM_t& Vop,
       long long int& flops) const;

      virtual void redistribute_to_tree_to_buffers
      (const DistM_t& A, std::size_t Arlo, std::size_t Aclo,
       std::vector<std::vector<scalar_t>>& sbuf, int dest);
      virtual void redistribute_to_tree_from_buffers
      (const DistM_t& A, std::size_t Arlo, std::size_t Aclo,
       std::vector<scalar_t*>& pbuf);
      virtual void delete_redistributed_input();

      friend class HSSMatrixMPI<scalar_t>;
#endif //defined(STRUMPACK_USE_MPI)
    };

    template<typename scalar_t>
    HSSMatrixBase<scalar_t>::HSSMatrixBase
    (std::size_t m, std::size_t n, bool active)
      : _rows(m), _cols(n), _U_state(State::UNTOUCHED),
        _V_state(State::UNTOUCHED),
        _openmp_task_depth(0), _active(active) { }

    template<typename scalar_t>
    HSSMatrixBase<scalar_t>::HSSMatrixBase
    (const HSSMatrixBase<scalar_t>& other) {
      *this = other;
    }

    template<typename scalar_t> HSSMatrixBase<scalar_t>&
    HSSMatrixBase<scalar_t>::operator=
    (const HSSMatrixBase<scalar_t>& other) {
      _rows = other._rows;
      _cols = other._cols;
      for (auto& c : other._ch)
        _ch.emplace_back(c->clone());
      _U_state = other._U_state;
      _V_state = other._V_state;
      _openmp_task_depth = other._openmp_task_depth;
      _active = other._active;
      _U_rank = other._U_rank;
      _U_rows = other._U_rows;
      _V_rank = other._V_rank;
      _V_rows = other._V_rows;
      return *this;
    }

#if defined(STRUMPACK_USE_MPI)
    template<typename scalar_t> void HSSMatrixBase<scalar_t>::forward_solve
    (const HSSFactorsMPI<scalar_t>& ULV, WorkSolveMPI<scalar_t>& w,
     const DistM_t& b, bool partial) const {
      if (!this->active()) return;
      if (!w.w_seq)
        w.w_seq = std::unique_ptr<WorkSolve<scalar_t>>
          (new WorkSolve<scalar_t>());
#pragma omp parallel
#pragma omp single nowait
      forward_solve
        (*(ULV._factors_seq), *(w.w_seq),
         const_cast<DistM_t&>(b).dense_wrapper(), partial);
      w.z = DistM_t(b.grid(), std::move(w.w_seq->z));
      w.ft1 = DistM_t(b.grid(), std::move(w.w_seq->ft1));
      w.y = DistM_t(b.grid(), std::move(w.w_seq->y));
      w.x = DistM_t(b.grid(), std::move(w.w_seq->x));
      w.reduced_rhs = DistM_t(b.grid(), std::move(w.w_seq->reduced_rhs));
    }

    template<typename scalar_t> void HSSMatrixBase<scalar_t>::backward_solve
    (const HSSFactorsMPI<scalar_t>& ULV, WorkSolveMPI<scalar_t>& w,
     DistM_t& x) const {
      if (!this->active()) return;
      DenseM_t lx(x.rows(), x.cols());
      w.w_seq->x = w.x.dense_and_clear();
#pragma omp parallel
#pragma omp single nowait
      backward_solve(*(ULV._factors_seq), *(w.w_seq), lx);
      x = DistM_t(x.grid(), std::move(lx));
    }

    template<typename scalar_t> void HSSMatrixBase<scalar_t>::to_block_row
    (const DistM_t& dist, DenseM_t& sub, DistM_t& leaf) const {
      if (!this->active()) return;
      assert(std::size_t(dist.rows())==this->cols());
      sub = dist.dense();
    }

    template<typename scalar_t> void
    HSSMatrixBase<scalar_t>::allocate_block_row
    (int d, DenseM_t& sub, DistM_t& leaf) const {
      if (!this->active()) return;
      sub = DenseM_t(rows(), d);
    }

    template<typename scalar_t> void HSSMatrixBase<scalar_t>::from_block_row
    (DistM_t& dist, const DenseM_t& sub, const DistM_t& leaf,
     const BLACSGrid* lg) const {
      if (!this->active()) return;
      dist = DistM_t(lg, sub);
    }

    /**
     * This switches from distributed compression to sequential/
     * threaded compression on the subtree.
     */
    template<typename scalar_t> void
    HSSMatrixBase<scalar_t>::compress_recursive_original
    (DistSamples<scalar_t>& RS, const delemw_t& Aelem, const opts_t& opts,
     WorkCompressMPI<scalar_t>& w_mpi, int dd) {
      if (!active()) return;
      auto lg = RS.HSS().grid_local();
      std::pair<std::size_t,std::size_t> offset;
      LocalElemMult<scalar_t> lAelem
        (Aelem, (w_mpi.lvl==0) ? offset : w_mpi.offset, lg, _Asub);
      w_mpi.create_sequential();
      WorkCompress<scalar_t>& w = *(w_mpi.w_seq);
      if (w.lvl == 0) w.offset = w_mpi.offset;
      std::swap(w.Ir, w_mpi.Ir); std::swap(w.Ic, w_mpi.Ic);
      std::swap(w.Jr, w_mpi.Jr); std::swap(w.Jc, w_mpi.Jc);
      bool was_not_compressed = !is_compressed();
#pragma omp parallel
#pragma omp single nowait
      compress_recursive_original
        (RS.sub_Rr, RS.sub_Rc, RS.sub_Sr, RS.sub_Sc,
         lAelem, opts, w, dd, _openmp_task_depth);
      if (is_compressed()) {
        auto lg = RS.HSS().grid_local();
        auto d = RS.sub_Rr.cols();
        if (was_not_compressed) {
          w_mpi.Rr = DistM_t
            (lg, DenseMW_t(V_rank(), d, RS.sub_Rr, w.offset.second, 0));
          w_mpi.Rc = DistM_t
            (lg, DenseMW_t(U_rank(), d, RS.sub_Rc, w.offset.second, 0));
          w_mpi.Sr = DistM_t
            (lg, DenseMW_t(U_rows(), d, RS.sub_Sr, w.offset.second, 0));
          w_mpi.Sc = DistM_t
            (lg, DenseMW_t(V_rows(), d, RS.sub_Sc, w.offset.second, 0));
        } else {
          auto d_old = w_mpi.Rr.cols();
          w_mpi.Rr.resize(V_rank(), d_old+dd);
          w_mpi.Rc.resize(U_rank(), d_old+dd);
          copy(V_rank(), dd,
               DenseMW_t(V_rank(), dd, RS.sub_Rr, w.offset.second, d-dd),
               0, w_mpi.Rr, 0, d_old, lg->ctxt());
          copy(U_rank(), dd,
               DenseMW_t(U_rank(), dd, RS.sub_Rc, w.offset.second, d-dd),
               0, w_mpi.Rc, 0, d_old, lg->ctxt());
          d_old = w_mpi.Sr.cols();
          w_mpi.Sr.resize(U_rows(), d_old+dd);
          w_mpi.Sc.resize(V_rows(), d_old+dd);
          copy(U_rows(), dd,
               DenseMW_t(U_rows(), dd, RS.sub_Sr, w.offset.second, d-dd),
               0, w_mpi.Sr, 0, d_old, lg->ctxt());
          copy(V_rows(), dd,
               DenseMW_t(V_rows(), dd, RS.sub_Sc, w.offset.second, d-dd),
               0, w_mpi.Sc, 0, d_old, lg->ctxt());
        }
      }
      std::swap(w.Ir, w_mpi.Ir); std::swap(w.Ic, w_mpi.Ic);
      std::swap(w.Jr, w_mpi.Jr); std::swap(w.Jc, w_mpi.Jc);
      if (w.lvl != 0 && was_not_compressed && is_compressed()) {
        for (auto& i : w_mpi.Ir) i += w_mpi.offset.first;
        for (auto& j : w_mpi.Ic) j += w_mpi.offset.second;
      }
    }

    /**
     * This switches from distributed compression to sequential/
     * threaded compression on the subtree.
     */
    template<typename scalar_t> void
    HSSMatrixBase<scalar_t>::compress_level_original
    (DistSamples<scalar_t>& RS, const opts_t& opts,
     WorkCompressMPI<scalar_t>& w_mpi, int dd, int lvl) {
      if (!active()) return;
      WorkCompress<scalar_t>& w = *(w_mpi.w_seq);
      if (w.lvl == 0) w.offset = w_mpi.offset;
      if (w.lvl == lvl) {
        std::swap(w.Ir, w_mpi.Ir); std::swap(w.Ic, w_mpi.Ic);
        std::swap(w.Jr, w_mpi.Jr); std::swap(w.Jc, w_mpi.Jc);
      }
      bool was_not_compressed = !is_compressed();
#pragma omp parallel
#pragma omp single nowait
      compress_level_original
        (RS.sub_Rr, RS.sub_Rc, RS.sub_Sr, RS.sub_Sc,
         opts, w, dd, lvl, _openmp_task_depth);
      if (w.lvl == lvl) {
        if (is_compressed()) {
          auto lg = RS.HSS().grid_local();
          auto d = RS.sub_Rr.cols();
          if (was_not_compressed) {
            w_mpi.Rr = DistM_t
              (lg, DenseMW_t(V_rank(), d, RS.sub_Rr, w.offset.second, 0));
            w_mpi.Rc = DistM_t
              (lg, DenseMW_t(U_rank(), d, RS.sub_Rc, w.offset.second, 0));
            w_mpi.Sr = DistM_t
              (lg, DenseMW_t(U_rows(), d, RS.sub_Sr, w.offset.second, 0));
            w_mpi.Sc = DistM_t
              (lg, DenseMW_t(V_rows(), d, RS.sub_Sc, w.offset.second, 0));
          } else {
            auto d_old = w_mpi.Rr.cols();
            w_mpi.Rr.resize(V_rank(), d_old+dd);
            w_mpi.Rc.resize(U_rank(), d_old+dd);
            copy(V_rank(), dd,
                 DenseMW_t(V_rank(), dd, RS.sub_Rr, w.offset.second, d-dd),
                 0, w_mpi.Rr, 0, d_old, lg->ctxt());
            copy(U_rank(), dd,
                 DenseMW_t(U_rank(), dd, RS.sub_Rc, w.offset.second, d-dd),
                 0, w_mpi.Rc, 0, d_old, lg->ctxt());
            d_old = w_mpi.Sr.cols();
            w_mpi.Sr.resize(U_rows(), d_old+dd);
            w_mpi.Sc.resize(V_rows(), d_old+dd);
            copy(U_rows(), dd,
                 DenseMW_t(U_rows(), dd, RS.sub_Sr, w.offset.second, d-dd),
                 0, w_mpi.Sr, 0, d_old, lg->ctxt());
            copy(V_rows(), dd,
                 DenseMW_t(V_rows(), dd, RS.sub_Sc, w.offset.second, d-dd),
                 0, w_mpi.Sc, 0, d_old, lg->ctxt());
          }
        }
        std::swap(w.Ir, w_mpi.Ir); std::swap(w.Ic, w_mpi.Ic);
        std::swap(w.Jr, w_mpi.Jr); std::swap(w.Jc, w_mpi.Jc);
        if (w.lvl != 0 && was_not_compressed && is_compressed()) {
          for (auto& i : w_mpi.Ir) i += w_mpi.offset.first;
          for (auto& j : w_mpi.Ic) j += w_mpi.offset.second;
        }
      }
    }

    template<typename scalar_t> void
    HSSMatrixBase<scalar_t>::compress_recursive_stable
    (DistSamples<scalar_t>& RS, const delemw_t& Aelem, const opts_t& opts,
     WorkCompressMPI<scalar_t>& w_mpi, int d, int dd) {
      if (!active()) return;
      auto lg = RS.HSS().grid_local();
      std::pair<std::size_t,std::size_t> offset;
      LocalElemMult<scalar_t> lAelem
        (Aelem, (w_mpi.lvl==0) ? offset : w_mpi.offset, lg, _Asub);
      w_mpi.create_sequential();
      WorkCompress<scalar_t>& w = *(w_mpi.w_seq);
      if (w.lvl == 0) w.offset = w_mpi.offset;
      std::swap(w.Ir, w_mpi.Ir); std::swap(w.Ic, w_mpi.Ic);
      std::swap(w.Jr, w_mpi.Jr); std::swap(w.Jc, w_mpi.Jc);
      bool was_not_compressed = !is_compressed();
#pragma omp parallel
#pragma omp single nowait
      compress_recursive_stable
        (RS.sub_Rr, RS.sub_Rc, RS.sub_Sr, RS.sub_Sc,
         lAelem, opts, w, d, dd, _openmp_task_depth);
      if (is_compressed()) {
        //auto lg = RS.HSS().grid_local();
        auto c = RS.sub_Rr.cols();
        if (was_not_compressed) {
          w_mpi.Rr = DistM_t
            (lg, DenseMW_t(V_rank(), c, RS.sub_Rr, w.offset.second, 0));
          w_mpi.Rc = DistM_t
            (lg, DenseMW_t(U_rank(), c, RS.sub_Rc, w.offset.second, 0));
          w_mpi.Sr = DistM_t
            (lg, DenseMW_t(U_rows(), c, RS.sub_Sr, w.offset.second, 0));
          w_mpi.Sc = DistM_t
            (lg, DenseMW_t(V_rows(), c, RS.sub_Sc, w.offset.second, 0));
        } else {
          auto d_old = w_mpi.Rr.cols();
          w_mpi.Rr.resize(V_rank(), d_old+dd);
          w_mpi.Rc.resize(U_rank(), d_old+dd);
          copy(V_rank(), dd,
               DenseMW_t(V_rank(), dd, RS.sub_Rr, w.offset.second, c-dd),
               0, w_mpi.Rr, 0, d_old, lg->ctxt());
          copy(U_rank(), dd,
               DenseMW_t(U_rank(), dd, RS.sub_Rc, w.offset.second, c-dd),
               0, w_mpi.Rc, 0, d_old, lg->ctxt());
          d_old = w_mpi.Sr.cols();
          w_mpi.Sr.resize(U_rows(), d_old+dd);
          w_mpi.Sc.resize(V_rows(), d_old+dd);
          copy(U_rows(), dd,
               DenseMW_t(U_rows(), dd, RS.sub_Sr, w.offset.second, c-dd),
               0, w_mpi.Sr, 0, d_old, lg->ctxt());
          copy(V_rows(), dd,
               DenseMW_t(V_rows(), dd, RS.sub_Sc, w.offset.second, c-dd),
               0, w_mpi.Sc, 0, d_old, lg->ctxt());
        }
      }
      std::swap(w.Ir, w_mpi.Ir); std::swap(w.Ic, w_mpi.Ic);
      std::swap(w.Jr, w_mpi.Jr); std::swap(w.Jc, w_mpi.Jc);
      if (w.lvl != 0 && was_not_compressed && is_compressed()) {
        for (auto& i : w_mpi.Ir) i += w_mpi.offset.first;
        for (auto& j : w_mpi.Ic) j += w_mpi.offset.second;
      }
    }

    template<typename scalar_t> void
    HSSMatrixBase<scalar_t>::compress_level_stable
    (DistSamples<scalar_t>& RS, const opts_t& opts,
     WorkCompressMPI<scalar_t>& w_mpi, int d, int dd, int lvl) {
      if (!active()) return;
      WorkCompress<scalar_t>& w = *(w_mpi.w_seq);
      if (w.lvl == 0) w.offset = w_mpi.offset;
      if (w.lvl == lvl) {
        std::swap(w.Ir, w_mpi.Ir); std::swap(w.Ic, w_mpi.Ic);
        std::swap(w.Jr, w_mpi.Jr); std::swap(w.Jc, w_mpi.Jc);
      }
      bool was_not_compressed = !is_compressed();
#pragma omp parallel
#pragma omp single nowait
      compress_level_stable
        (RS.sub_Rr, RS.sub_Rc, RS.sub_Sr, RS.sub_Sc,
         opts, w, d, dd, lvl, _openmp_task_depth);
      if (w.lvl == lvl) {
        if (is_compressed()) {
          auto lg = RS.HSS().grid_local();
          auto c = RS.sub_Rr.cols();
          if (was_not_compressed) {
            w_mpi.Rr = DistM_t
              (lg, DenseMW_t(V_rank(), c, RS.sub_Rr, w.offset.second, 0));
            w_mpi.Rc = DistM_t
              (lg, DenseMW_t(U_rank(), c, RS.sub_Rc, w.offset.second, 0));
            w_mpi.Sr = DistM_t
              (lg, DenseMW_t(U_rows(), c, RS.sub_Sr, w.offset.second, 0));
            w_mpi.Sc = DistM_t
              (lg, DenseMW_t(V_rows(), c, RS.sub_Sc, w.offset.second, 0));
          } else {
            auto d_old = w_mpi.Rr.cols();
            w_mpi.Rr.resize(V_rank(), d_old+dd);
            w_mpi.Rc.resize(U_rank(), d_old+dd);
            copy(V_rank(), dd,
                 DenseMW_t(V_rank(), dd, RS.sub_Rr, w.offset.second, c-dd),
                 0, w_mpi.Rr, 0, d_old, lg->ctxt());
            copy(U_rank(), dd,
                 DenseMW_t(U_rank(), dd, RS.sub_Rc, w.offset.second, c-dd),
                 0, w_mpi.Rc, 0, d_old, lg->ctxt());
            d_old = w_mpi.Sr.cols();
            w_mpi.Sr.resize(U_rows(), d_old+dd);
            w_mpi.Sc.resize(V_rows(), d_old+dd);
            copy(U_rows(), dd,
                 DenseMW_t(U_rows(), dd, RS.sub_Sr, w.offset.second, c-dd),
                 0, w_mpi.Sr, 0, d_old, lg->ctxt());
            copy(V_rows(), dd,
                 DenseMW_t(V_rows(), dd, RS.sub_Sc, w.offset.second, c-dd),
                 0, w_mpi.Sc, 0, d_old, lg->ctxt());
          }
        }
        std::swap(w.Ir, w_mpi.Ir); std::swap(w.Ic, w_mpi.Ic);
        std::swap(w.Jr, w_mpi.Jr); std::swap(w.Jc, w_mpi.Jc);
        if (w.lvl != 0 && was_not_compressed && is_compressed()) {
          for (auto& i : w_mpi.Ir) i += w_mpi.offset.first;
          for (auto& j : w_mpi.Ic) j += w_mpi.offset.second;
        }
      }
    }

    template<typename scalar_t> void
    HSSMatrixBase<scalar_t>::compress_recursive_ann
    (DenseMatrix<std::uint32_t>& ann, DenseMatrix<real_t>& scores,
     const delemw_t& Aelem, WorkCompressMPIANN<scalar_t>& w_mpi,
     const opts_t& opts, const BLACSGrid* lg) {
      if (!active()) return;
      std::pair<std::size_t,std::size_t> offset;
      LocalElemMult<scalar_t> lAelem
        (Aelem, offset, lg, _Asub);
      w_mpi.create_sequential();
      WorkCompressANN<scalar_t>& w = *(w_mpi.w_seq);
      w.offset = w_mpi.offset;
#pragma omp parallel
#pragma omp single nowait
      compress_recursive_ann
        (ann, scores, lAelem, opts, w, _openmp_task_depth);
      w_mpi.S = DistM_t(lg, std::move(w.S));
      std::swap(w.Ir, w_mpi.Ir);
      std::swap(w.Ic, w_mpi.Ic);
      std::swap(w.Jr, w_mpi.Jr);
      std::swap(w.Jc, w_mpi.Jc);
      std::swap(w.ids_scores, w_mpi.ids_scores);
    }


    template<typename scalar_t> void
    HSSMatrixBase<scalar_t>::get_extraction_indices
    (std::vector<std::vector<std::size_t>>& I,
     std::vector<std::vector<std::size_t>>& J,
     WorkCompressMPI<scalar_t>& w, int& self, int lvl) {
      if (!this->active()) return;
      w.create_sequential();
      get_extraction_indices(I, J, w.offset, *w.w_seq, self, lvl);
    }

    template<typename scalar_t> void
    HSSMatrixBase<scalar_t>::get_extraction_indices
    (std::vector<std::vector<std::size_t>>& I,
     std::vector<std::vector<std::size_t>>& J, std::vector<DistMW_t>& B,
     const BLACSGrid* lg, WorkCompressMPI<scalar_t>& w, int& self, int lvl) {
      if (!this->active()) return;
      w.create_sequential();
      std::vector<DenseM_t*> Bdense;
      get_extraction_indices(I, J, Bdense, w.offset, *w.w_seq, self, lvl);
      for (auto& Bd : Bdense)
        B.emplace_back(lg, Bd->rows(), Bd->cols(), *Bd);
    }

    template<typename scalar_t> void HSSMatrixBase<scalar_t>::extract_D_B
    (const delemw_t& Aelem, const BLACSGrid* lg, const opts_t& opts,
     WorkCompressMPI<scalar_t>& w, int lvl) {
      if (!this->active()) return;
      LocalElemMult<scalar_t> lAelem(Aelem, w.offset, lg, _Asub);
      extract_D_B(lAelem, opts, *w.w_seq, lvl);
    }

    template<typename scalar_t> void HSSMatrixBase<scalar_t>::apply_fwd
    (const DistSubLeaf<scalar_t>& B, WorkApplyMPI<scalar_t>& w,
     bool isroot, long long int flops) const {
      if (!active()) return;
      if (!w.w_seq)
        w.w_seq = std::unique_ptr<WorkApply<scalar_t>>
          (new WorkApply<scalar_t>());
      if (isroot) w.w_seq->offset = w.offset;
      std::atomic<long long int> lflops(0);
#pragma omp parallel
#pragma omp single nowait
      apply_fwd(B.sub, *(w.w_seq), isroot, _openmp_task_depth, lflops);
      flops += lflops.load();
      w.tmp1 = DistM_t(B.grid_local(), std::move(w.w_seq->tmp1));
    }

    template<typename scalar_t> void HSSMatrixBase<scalar_t>::apply_bwd
    (const DistSubLeaf<scalar_t>& B, scalar_t beta, DistSubLeaf<scalar_t>& C,
     WorkApplyMPI<scalar_t>& w, bool isroot, long long int flops) const {
      if (!active()) return;
      w.w_seq->tmp2 = w.tmp2.dense_and_clear();
      std::atomic<long long int> lflops(0);
#pragma omp parallel
#pragma omp single nowait
      apply_bwd
        (B.sub, beta, C.sub, *(w.w_seq), isroot,
         _openmp_task_depth, lflops);
      flops += lflops.load();
    }

    template<typename scalar_t> void HSSMatrixBase<scalar_t>::applyT_fwd
    (const DistSubLeaf<scalar_t>& B, WorkApplyMPI<scalar_t>& w,
     bool isroot, long long int flops) const {
      if (!active()) return;
      if (!w.w_seq)
        w.w_seq = std::unique_ptr<WorkApply<scalar_t>>
          (new WorkApply<scalar_t>());
      if (isroot) w.w_seq->offset = w.offset;
      std::atomic<long long int> lflops(0);
#pragma omp parallel
#pragma omp single nowait
      applyT_fwd(B.sub, *(w.w_seq), isroot, _openmp_task_depth, lflops);
      flops += lflops.load();
      w.tmp1 = DistM_t(B.grid_local(), std::move(w.w_seq->tmp1));
    }

    template<typename scalar_t> void HSSMatrixBase<scalar_t>::applyT_bwd
    (const DistSubLeaf<scalar_t>& B, scalar_t beta, DistSubLeaf<scalar_t>& C,
     WorkApplyMPI<scalar_t>& w, bool isroot, long long int flops) const {
      if (!active()) return;
      w.w_seq->tmp2 = w.tmp2.dense_and_clear();
      std::atomic<long long int> lflops(0);
#pragma omp parallel
#pragma omp single nowait
      applyT_bwd
        (B.sub, beta, C.sub, *(w.w_seq), isroot,
         _openmp_task_depth, lflops);
      flops += lflops.load();
    }

    template<typename scalar_t> void HSSMatrixBase<scalar_t>::factor_recursive
    (HSSFactorsMPI<scalar_t>& ULV, WorkFactorMPI<scalar_t>& w,
     const BLACSGrid* lg, bool isroot, bool partial) const {
      if (!active()) return;
      if (!ULV._factors_seq)
        ULV._factors_seq = std::unique_ptr<HSSFactors<scalar_t>>
          (new HSSFactors<scalar_t>());
      if (!w.w_seq)
        w.w_seq = std::unique_ptr<WorkFactor<scalar_t>>
          (new WorkFactor<scalar_t>());
#pragma omp parallel
#pragma omp single nowait
      factor_recursive
        (*(ULV._factors_seq), *(w.w_seq), isroot, partial, _openmp_task_depth);
      if (isroot) {
        if (partial) ULV._Vt0 = DistM_t(lg, ULV._factors_seq->_Vt0);
        ULV._D = DistM_t(lg, ULV._factors_seq->_D);
        ULV._piv.resize(ULV._D.lrows() + ULV._D.MB());
        std::copy(ULV._factors_seq->_piv.begin(),
                  ULV._factors_seq->_piv.end(), ULV._piv.begin());
      } else {
        w.Dt = DistM_t(lg, std::move(w.w_seq->Dt));
        w.Vt1 = DistM_t(lg, std::move(w.w_seq->Vt1));
        ULV._Q = DistM_t(lg, std::move(ULV._factors_seq->_Q));
        ULV._W1 = DistM_t(lg, std::move(ULV._factors_seq->_W1));
      }
    }

    template<typename scalar_t> void HSSMatrixBase<scalar_t>::solve_fwd
    (const HSSFactorsMPI<scalar_t>& ULV, const DistSubLeaf<scalar_t>& b,
     WorkSolveMPI<scalar_t>& w, bool partial, bool isroot) const {
      if (!active()) return;
      if (!w.w_seq)
        w.w_seq = std::unique_ptr<WorkSolve<scalar_t>>
          (new WorkSolve<scalar_t>());
#pragma omp parallel
#pragma omp single nowait
      solve_fwd(*(ULV._factors_seq), b.sub, *(w.w_seq),
                partial, isroot, _openmp_task_depth);
      w.z = DistM_t(b.grid_local(), std::move(w.w_seq->z));
      w.ft1 = DistM_t(b.grid_local(), std::move(w.w_seq->ft1));
      w.y = DistM_t(b.grid_local(), std::move(w.w_seq->y));
    }

    template<typename scalar_t> void HSSMatrixBase<scalar_t>::solve_bwd
    (const HSSFactorsMPI<scalar_t>& ULV, DistSubLeaf<scalar_t>& x,
     WorkSolveMPI<scalar_t>& w, bool isroot) const {
      if (!active()) return;
      w.w_seq->x = w.x.dense_and_clear();
#pragma omp parallel
#pragma omp single nowait
      solve_bwd(*(ULV._factors_seq), x.sub, *(w.w_seq),
                isroot, _openmp_task_depth);
    }

    template<typename scalar_t> void HSSMatrixBase<scalar_t>::extract_fwd
    (WorkExtractMPI<scalar_t>& w, const BLACSGrid* lg, bool odiag) const {
      if (!active()) return;
      if (!w.w_seq)
        w.w_seq = std::unique_ptr<WorkExtract<scalar_t>>
          (new WorkExtract<scalar_t>());
      WorkExtract<scalar_t>& w_seq = *(w.w_seq);
      std::swap(w.I, w_seq.I);
      std::swap(w.J, w_seq.J);
      std::swap(w.ycols, w_seq.ycols);
#pragma omp parallel
#pragma omp single nowait
      extract_fwd(*(w.w_seq), odiag, _openmp_task_depth);
      std::swap(w.I, w_seq.I);
      std::swap(w.J, w_seq.J);
      std::swap(w.ycols, w_seq.ycols);
      w.y = DistM_t(lg, std::move(w.w_seq->y));
    }

    template<typename scalar_t> void HSSMatrixBase<scalar_t>::extract_fwd
    (WorkExtractBlocksMPI<scalar_t>& w, const BLACSGrid* lg,
     std::vector<bool>& odiag) const {
      if (!active()) return;
      const auto nb = w.I.size();
      w.w_seq.resize(nb);
      for (std::size_t k=0; k<nb; k++) {
        w.w_seq[k] = std::unique_ptr<WorkExtract<scalar_t>>
          (new WorkExtract<scalar_t>());
        WorkExtract<scalar_t>& w_seq = *w.w_seq[k];
        std::swap(w.I[k], w_seq.I);
        std::swap(w.J[k], w_seq.J);
        std::swap(w.ycols[k], w_seq.ycols);
#pragma omp parallel
#pragma omp single nowait
        extract_fwd(w_seq, odiag[k], _openmp_task_depth);
        std::swap(w.I[k], w_seq.I);
        std::swap(w.J[k], w_seq.J);
        std::swap(w.ycols[k], w_seq.ycols);
        w.y[k] = DistM_t(lg, std::move(w_seq.y));
      }
    }

    template<typename scalar_t> void HSSMatrixBase<scalar_t>::extract_bwd
    (std::vector<Triplet<scalar_t>>& triplets, const BLACSGrid* lg,
     WorkExtractMPI<scalar_t>& w) const {
      if (!active()) return;
      if (!w.w_seq)
        w.w_seq = std::unique_ptr<WorkExtract<scalar_t>>
          (new WorkExtract<scalar_t>());
      WorkExtract<scalar_t>& w_seq = *(w.w_seq);
      std::swap(w.I, w_seq.I);
      std::swap(w.J, w_seq.J);
      std::swap(w.ycols, w_seq.ycols);
      std::swap(w.zcols, w_seq.zcols);
      std::swap(w.rl2g, w_seq.rl2g);
      std::swap(w.cl2g, w_seq.cl2g);
      w.w_seq->z = w.z.dense_and_clear();
#pragma omp parallel
#pragma omp single nowait
      extract_bwd(triplets, w_seq, _openmp_task_depth);
    }

    template<typename scalar_t> void HSSMatrixBase<scalar_t>::extract_bwd
    (std::vector<std::vector<Triplet<scalar_t>>>& triplets,
     const BLACSGrid* lg, WorkExtractBlocksMPI<scalar_t>& w) const {
      if (!active()) return;
      const auto nb = w.I.size();
      w.w_seq.resize(nb);
      for (std::size_t k=0; k<nb; k++) {
        if (!w.w_seq[k])
          w.w_seq[k] = std::unique_ptr<WorkExtract<scalar_t>>
            (new WorkExtract<scalar_t>());
        WorkExtract<scalar_t>& w_seq = *w.w_seq[k];
        // move instead??
        std::swap(w.I[k], w_seq.I);
        std::swap(w.J[k], w_seq.J);
        std::swap(w.ycols[k], w_seq.ycols);
        std::swap(w.zcols[k], w_seq.zcols);
        std::swap(w.rl2g[k], w_seq.rl2g);
        std::swap(w.cl2g[k], w_seq.cl2g);
        w_seq.z = w.z[k].dense_and_clear();
#pragma omp parallel
#pragma omp single nowait
        extract_bwd(triplets[k], w_seq, _openmp_task_depth);
      }
    }

    template<typename scalar_t> void HSSMatrixBase<scalar_t>::apply_UV_big
    (DistSubLeaf<scalar_t>& Theta, DistM_t& Uop, DistSubLeaf<scalar_t>& Phi,
     DistM_t& Vop, long long int& flops) const {
      if (!active()) return;
      auto sUop = Uop.dense_and_clear();
      auto sVop = Vop.dense_and_clear();
      const std::pair<std::size_t, std::size_t> offset;
      std::atomic<long long int> UVflops(0);
#pragma omp parallel
#pragma omp single nowait
      apply_UV_big
        (Theta.sub, sUop, Phi.sub, sVop, offset, _openmp_task_depth, UVflops);
      flops += UVflops.load();
    }

    template<typename scalar_t> void
    HSSMatrixBase<scalar_t>::redistribute_to_tree_to_buffers
    (const DistM_t& A, std::size_t Arlo, std::size_t Aclo,
     std::vector<std::vector<scalar_t>>& sbuf, int dest) {
      const DistMW_t Ad
        (rows(), cols(), const_cast<DistM_t&>(A), Arlo, Aclo);
      int rlo, rhi, clo, chi;
      Ad.lranges(rlo, rhi, clo, chi);
      sbuf.reserve(sbuf.size()+(chi-clo)*(rhi-rlo));
      for (int c=clo; c<chi; c++)
        for (int r=rlo; r<rhi; r++)
          sbuf[dest].push_back(Ad(r,c));
    }

    template<typename scalar_t> void
    HSSMatrixBase<scalar_t>::redistribute_to_tree_from_buffers
    (const DistM_t& A, std::size_t Arlo, std::size_t Aclo,
     std::vector<scalar_t*>& pbuf) {
      if (!this->active()) return;
      _Asub = DenseM_t(rows(), cols());
      const auto B = DistM_t::default_MB;
      const auto Aprows = A.grid()->nprows();
      const auto Apcols = A.grid()->npcols();
      std::vector<std::size_t> srcr(rows());
      for (std::size_t r=0; r<rows(); r++)
        srcr[r] = ((r + Arlo) / B) % Aprows;
      for (std::size_t c=0; c<cols(); c++)
        for (std::size_t srcc=(((c+Aclo)/B)%Apcols)*Aprows,
               r=0; r<cols(); r++)
          _Asub(r,c) = *(pbuf[srcr[r] + srcc]++);
    }

    template<typename scalar_t> void
    HSSMatrixBase<scalar_t>::delete_redistributed_input() {
      _Asub.clear();
    }
#endif //defined(STRUMPACK_USE_MPI)

  } // end namespace HSS
} // end namespace strumpack


#endif // HSS_MATRIX_BASE_HPP
