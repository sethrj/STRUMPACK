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
 * works, and perform publicly and display publicly. Beginning five
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
#define COUNT_FLOPS 1
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <atomic>

#include "HSS/HSSMatrixMPI.hpp"
#include "misc/TaskTimer.hpp"
#include "zC_HODLR_wrapper.h"
#include "C_EMCURV_wrapper.h"

using namespace std;
using namespace strumpack;
using namespace strumpack::HSS;

typedef std::complex<double> dcomplex;
#define myscalar dcomplex
#define cscalar _Complex double


// #define myscalar double
// #define cscalar double


// FAST_H_SAMPLING= 0: N^2 sampling followed by HSS factor-solve 
// FAST_H_SAMPLING= 2: HODLR sampling followed by HSS factor-solve
// FAST_H_SAMPLING= 3: HODLR sampling followed by HODLR factor-solve
#define FAST_H_SAMPLING 3

#if defined(_OPENMP)
#include <omp.h>
#endif


#define ERROR_TOLERANCE 1e2

random_device rd;
double r;
//mt19937 generator(rd());
mt19937 generator; // make sure to use the same ordering on all processes






inline double dist2(double *x, double *y, int d) {
  double k = 0.;
  for (int i = 0; i < d; i++)
    k += pow(x[i] - y[i], 2.);
  return k;
}

inline double dist(double *x, double *y, int d) { return sqrt(dist2(x, y, d)); }


void kd_partition(double *p, int n, int d, int *nc) {
  // find coordinate of the most spread
  double *maxes = new double[d];
  double *mins = new double[d];

  for (int j = 0; j < d; ++j) {
    maxes[j] = p[j];
    mins[j] = p[j];
  }

  for (int i = 1; i < n; ++i) {
    for (int j = 0; j < d; ++j) {
      if (p[i * d + j] > maxes[j]) {
        maxes[j] = p[i * d + j];
      }
      if (p[i * d + j] > mins[j]) {
        mins[j] = p[i * d + j];
      }
    }
  }
  double max_var = maxes[0] - mins[0];
  int dim = 0;
  for (int j = 0; j < d; ++j) {
    if (maxes[j] - mins[j] > max_var) {
      max_var = maxes[j] - mins[j];
      dim = j;
    }
  }

   
  
  std::vector<std::pair<double, int>> dat_MD;
  dat_MD.resize(n);
  for (int i=0;i<n;++i){
	dat_MD.data()[i].first = p[i * d + dim];
	dat_MD.data()[i].second = i;
  } 
  std::sort(dat_MD.begin(), dat_MD.end());  
  double *p_perm = new double[n * d];
  for (int row=0; row<n; ++row){
  for (int l = 0; l < d; l++)
    p_perm[l + row * d] = p[l + dat_MD.data()[row].second * d]; 
  }  
  dat_MD.clear();
  
#if 1  // split by median
  nc[0] = floor((double)n/2.0);
  nc[1] = n - nc[0];  
#else // split by mean  
  // find the mean
  double mean_value = 0.;
  for (int i = 0; i < n; ++i) {
    mean_value += p_perm[i * d + dim];
  }
  mean_value /= n;  

  nc[0] = 0;
  nc[1] = 0;
  for (int i = 0; i < n; ++i) {
    if (p_perm[d * i + dim] > mean_value) {
      nc[1] += 1;
    } else {
      nc[0] += 1;
    }
  }

#endif  
  
  
  copy(p_perm, p_perm + n * d, p);
  delete[] p_perm;
  delete[] maxes;
  delete[] mins;

}


void cobble_partition(double *p, int n, int d, int *nc) {

  // find centroid
  double centroid[d];

  for (int i = 0; i < d; i++) {
    centroid[i] = 0;
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < d; j++) {
      centroid[j] += p[i * d + j];
    }
  }

  for (int j = 0; j < d; j++)
    centroid[j] /= n;

  // find farthest point from centroid
  int first_index = 0;
  double max_dist = -1;

  for (int i = 0; i < n; i++) {
    double dd = dist(&p[i * d], centroid, d);
    if (dd > max_dist) {
      max_dist = dd;
      first_index = i;
    }
  }

  // compute and sort distance from the firsth point
  std::vector<double> dists;
  dists.resize(n);
  for (int i = 0; i < n; i++) {
    dists[i] = dist(&p[i * d], &p[first_index * d], d);
  }
  
  std::vector<std::pair<double, int>> dat_MD;
  dat_MD.resize(n);
  for (int i=0;i<n;++i){
	dat_MD.data()[i].first = dists[i];
	dat_MD.data()[i].second = i;
  } 
  std::sort(dat_MD.begin(), dat_MD.end());  
  
  double *p_perm = new double[n * d];
  for (int row=0; row<n; ++row){
  // cout<<dat_MD.data()[row].first<<""<<dat_MD.data()[row].second<<endl;
  for (int l = 0; l < d; l++)
    p_perm[l + row * d] = p[l + dat_MD.data()[row].second * d]; 
  }  
  dat_MD.clear();
  dists.clear();
  
  nc[0] = floor((double)n/2.0);
  nc[1] = n - nc[0];    
  
  copy(p_perm, p_perm + n * d, p);
  delete[] p_perm;

}




void recursive_cobble(double *p, int n, int d, int cluster_size,
                  HSSPartitionTree &tree) {
  if (n < cluster_size)
    return;
  auto nc = new int[2];
  cobble_partition(p, n, d, nc);
  if (nc[0] == 0 || nc[1] == 0)
    return;
  tree.c.resize(2);
  tree.c[0].size = nc[0];
  tree.c[1].size = nc[1];
  recursive_cobble(p, nc[0], d, cluster_size, tree.c[0]);
  recursive_cobble(p + nc[0] * d, nc[1], d, cluster_size, tree.c[1]);
  delete[] nc;
}



void recursive_kd(double *p, int n, int d, int cluster_size,
                  HSSPartitionTree &tree) {
  if (n < cluster_size)
    return;
  auto nc = new int[2];
  kd_partition(p, n, d, nc);
  if (nc[0] == 0 || nc[1] == 0)
    return;
  tree.c.resize(2);
  tree.c[0].size = nc[0];
  tree.c[1].size = nc[1];
  recursive_kd(p, nc[0], d, cluster_size, tree.c[0]);
  recursive_kd(p + nc[0] * d, nc[1], d, cluster_size, tree.c[1]);
  delete[] nc;
}




// get maximum tree level
inline void get_treelevel(HSSPartitionTree &tree, int &level, int level_ind) {
	level = std::max(level,level_ind);
	if (!tree.c.empty()) {
		get_treelevel(tree.c[0], level, level_ind+1);
		get_treelevel(tree.c[1], level, level_ind+1);
	}
}

// get leaf node sizes of a tree
inline void get_leafsizes(HSSPartitionTree &tree, int &leaf_ind, int* leaf_sizes) {
	if (!tree.c.empty()) {
		get_leafsizes(tree.c[0], leaf_ind, leaf_sizes);
		get_leafsizes(tree.c[1], leaf_ind, leaf_sizes);
	}else{
		leaf_sizes[leaf_ind++] = tree.size;
	}
} 


//convert from blacs local indices to global indices
inline int l2g(int il, int p, int n,int np,int nb){
	int ilm1 = il-1;
	return ((floor(ilm1/nb) * np) + p)*nb + ilm1%nb + 1;
} 



class CompressSetup {
  using DenseM_t = DenseMatrix<myscalar>;
  using DenseMW_t = DenseMatrixWrapper<myscalar>;
  using DistM_t = DistributedMatrix<myscalar>;
  using DistMW_t = DistributedMatrixWrapper<myscalar>;

public:
  // DenseMatrix<myscalar>* _Aseq=nullptr;	
  vector<double> _data;
  int _n = 0;
  string _reorder;
  int _dim = 2;
  int _model2d = 10; // Model type (1=strip; 2=corner reflector; 3=two opposite strips; 4=CR with RRS; 5=cylinder; 6=Rectangle Cavity); 7=half cylinder; 8=corrugated half cylinder; 9=corrugated corner reflector; 10=open polygon; 11=taller open polygon 
  double _wavelength = 0.08; 
  // double _h = 0.;
  // double _l = 0.;
  int _nprows = -1;
  int _npcols = -1;
  std::vector<int> _Hperm;
  std::vector<int> _iHperm;
  int _Hrows = 0;
  int _cluster_size = 500; // finest leafsize
  int _nlevel = 0; // 0: tree level, nonzero if a tree is provided 
  int* _tree; //user provided array containing size of each leaf node, not used if _nlevel=0  
  int _com_opt=2;			 
  vector<int> _dist;
  F2Cptr quant_ptr;
  F2Cptr ho_bf;  //HODLR returned by Fortran code 
  F2Cptr option;     //option structure returned by Fortran code 
  F2Cptr stats;      //statistics structure returned by Fortran code
  F2Cptr msh;		   //d_mesh structure returned by Fortran code
  F2Cptr kerquant;   //kernel quantities structure returned by Fortran code
  F2Cptr ptree;      //process tree returned by Fortran code
  MPI_Fint Fcomm;  // the fortran MPI communicator
  
  
 void times(DenseM_t &R, DistM_t &S, int Rprow) {
    const auto B = S.MB();
    const auto Bc = S.lcols();
	cscalar val;
    DenseM_t Asub(B, B);
#pragma omp parallel for firstprivate(Asub) schedule(dynamic)
    for (int lr = 0; lr < S.lrows(); lr += B) {
      const size_t Br = std::min(B, S.lrows() - lr);
      const int Ar = S.rowl2g(lr);
      for (int k = 0, Ac = Rprow*B; Ac < _n; k += B) {
        const size_t Bk = std::min(B, _n - Ac);
        // construct a block of A
        for (size_t j = 0; j < Bk; j++) {
          for (size_t i = 0; i < Br; i++) {
			int m,n;
			m = Ar + i;
			n = Ac + j;
			c_emcurv_sample(&m,&n,&val,quant_ptr);
            Asub(i, j) = (myscalar)val;
          }
        }
        DenseMW_t Ablock(Br, Bk, Asub, 0, 0);
        DenseMW_t Sblock(Br, Bc, &S(lr, 0), S.ld());
        DenseMW_t Rblock(Bk, Bc, &R(k, 0), R.ld());
        // multiply block of A with a row-block of Rr and add result to Sr
        gemm(Trans::N, Trans::N, (myscalar)1., Ablock, Rblock, (myscalar)1., Sblock);
        Ac += S.nprows() * B;
      }
    }
  }

  
  void densemult(DistM_t &R, DistM_t &Sr) {
    Sr.zero();
    int maxlocrows = R.MB() * (R.rows() / R.MB());
    if (R.rows() % R.MB()) maxlocrows += R.MB();
    int maxloccols = R.MB() * (R.cols() / R.MB());
    if (R.cols() % R.MB()) maxloccols += R.MB();
    DenseM_t tmp(maxlocrows, maxloccols);
    // each processor broadcasts his/her local part of R to all
    // processes in the same column of the BLACS grid, one after the
    // other
    for (int p=0; p<R.nprows(); p++) {
      if (p == R.prow()) {
        strumpack::scalapack::gebs2d
          (R.ctxt(), 'C', ' ', R.lrows(), R.lcols(), R.data(), R.ld());
        DenseMW_t Rdense(R.lrows(), R.lcols(), R.data(), R.ld());
        strumpack::copy(Rdense, tmp, 0, 0);
      } else {
        int recvrows = strumpack::scalapack::numroc
          (R.rows(), R.MB(), p, 0, R.nprows());
        strumpack::scalapack::gebr2d
          (R.ctxt(), 'C', ' ', recvrows, R.lcols(),
           tmp.data(), tmp.ld(), p, R.pcol());
      }
      times(tmp, Sr, p);
    }
  }  
    
  ~CompressSetup(){
#if FAST_H_SAMPLING == 2 || FAST_H_SAMPLING == 3
	z_c_hodlr_deletestats(&stats);
	z_c_hodlr_deleteproctree(&ptree);
	z_c_hodlr_deletemesh(&msh);
	z_c_hodlr_deletekernelquant(&kerquant);
	z_c_hodlr_deletehobf(&ho_bf);
	z_c_hodlr_deleteoption(&option);	
#endif	
	_data.resize(0);
	_Hperm.resize(0);
	_iHperm.resize(0);
	_dist.resize(0);
  }    
  
  CompressSetup() = default;
  CompressSetup(int n, HSSOptions<myscalar>& opts, 
	    int nprow, int npcol, int aca, int cluster_size, string reorder, double wavelength)
    : _n(n), _nprows(nprow), _npcols(npcol),_cluster_size(cluster_size),_com_opt(aca), _reorder(reorder),_wavelength(wavelength) {
	

	
  _data.resize(_n*_dim);
  Fcomm = MPI_Comm_c2f(MPI_COMM_WORLD); 
  c_emcurv_init(&_n, _data.data(), &quant_ptr, &_model2d, &_wavelength, &Fcomm);
	
	
	
  HSSPartitionTree cluster_tree;
  cluster_tree.size = _n;
  if (_reorder == "cob") {
    recursive_cobble(_data.data(), _n, _dim, _cluster_size, cluster_tree);  
  } else if (_reorder == "kd") {
    recursive_kd(_data.data(), _n, _dim, _cluster_size, cluster_tree);
  } 

  get_treelevel(cluster_tree, _nlevel, 0); 
  _tree = new int[(int)pow(2,_nlevel)]; //user provided array containing size of each leaf node, not used
  int leaf_ind=0;  
  get_leafsizes(cluster_tree, leaf_ind, _tree);
	
	
#if FAST_H_SAMPLING == 2	|| FAST_H_SAMPLING == 3
	   
    int P = mpi_nprocs();
    const auto rank = mpi_rank();
    if (!rank)
      cout << "# Called <FAST_H_SAMPLING> Started matrix construction..." << endl;
    auto starttime = MPI_Wtime();		   
		
	int* groups;
	int ker = 1;
	int _d =1;
	int nogeo=0;
	// int Nmin = 500;    // finest leafsize
	
	
#if FAST_H_SAMPLING == 2  // set slightly higher accuracy in hodlr than hss if hodlr is used as matvec	
	double tol = opts.rel_tol()*1e-1; // compression tolerance
#else
	double tol = opts.rel_tol(); // compression tolerance
#endif	
	
	// int com_opt=2;    //compression option 1:SVD 2:RRQR 3:ACA 4:BACA  
	int sort_opt=0; //0:natural order 1:kd-tree 2:cobble-like ordering 3:gram distance-based cobble-like ordering
	int checkerr = 0; //1: check compression quality 
	int batch = 100; //batch size for BACA	
	int myseg=0;     // local number of unknowns
	int LRlevel=100;   //the top LRlevel levels in hodlr are butterfly
	

    _Hperm.resize(_n);   
	groups = new int[P];
	for (int i = 0; i < P; i++)groups[i]=i;

	// create hodlr data structures
	z_c_hodlr_createptree(&P, groups, &Fcomm, &ptree);
	z_c_hodlr_createoption(&option);	
	z_c_hodlr_createstats(&stats);		
	
	// set hodlr options
	z_c_hodlr_set_D_option(&option, "tol_comp", tol);
	z_c_hodlr_set_I_option(&option, "nogeo", nogeo);
	z_c_hodlr_set_I_option(&option, "Nmin_leaf", _cluster_size); 
	z_c_hodlr_set_I_option(&option, "RecLR_leaf", _com_opt); 
	z_c_hodlr_set_I_option(&option, "xyzsort", sort_opt); 
	z_c_hodlr_set_I_option(&option, "ErrFillFull", checkerr); 
	z_c_hodlr_set_I_option(&option, "BACA_Batch", batch); 
	z_c_hodlr_set_I_option(&option, "LRlevel", LRlevel); 
	

    // construct hodlr with geometrical points	
	z_c_hodlr_construct(&_n, &_d, _data.data(), &_nlevel, _tree, _Hperm.data(), &_Hrows, &ho_bf, &option, &stats, &msh, &kerquant, &ptree, &c_emcurv_sample, quant_ptr, &Fcomm);		
	
	
    for (auto& i : _Hperm) i--; // Fortran to C
    MPI_Bcast(_Hperm.data(), _n, MPI_INT, 0, MPI_COMM_WORLD);
    _iHperm.resize(_n);
    for (int i=0; i<_n; i++) 
      _iHperm[_Hperm[i]] = i;
    _dist.resize(P+1);
    _dist[rank+1] = _Hrows;
    MPI_Allgather
      (MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
       _dist.data()+1, 1, MPI_INT, MPI_COMM_WORLD);
    for (int p=0; p<P; p++)
      _dist[p+1] += _dist[p];
    auto endtime = MPI_Wtime();
    if (!rank)
      cout << "# HODLR construction time " << endtime - starttime
           << " seconds" << endl;
      
#endif

  }

  void operator()(const vector<size_t> &I, const vector<size_t> &J,
                  DistM_t &B) {
	cscalar val;
    if (!B.active()) return;
    assert(I.size() == size_t(B.rows()) &&
           J.size() == size_t(B.cols()));
    for (size_t j = 0; j < J.size(); j++) {
      if (B.colg2p(j) != B.pcol()) continue;
      for (size_t i = 0; i < I.size(); i++) {
        if (B.rowg2p(i) == B.prow()) {
          assert(B.is_local(i, j));
		  int m,n;
		  m = I[i];
		  n = J[j];
		  c_emcurv_sample(&m,&n,&val, quant_ptr);
          B.global(i, j) = (myscalar)val;
        }
      }
    }
  }

  DenseM_t redistribute_2D_to_1D(DistM_t& R) {
    const auto P = mpi_nprocs();
    const auto rank = mpi_rank();
    const auto cols = R.cols();
    const auto lcols = R.lcols();
    const auto lrows = R.lrows();
    const auto B = DistM_t::default_MB;
    vector<vector<myscalar>> sbuf(P);
    if (R.active()) {
      // global, local, proc
      vector<tuple<int,int,int>> glp(lrows);
      {
	vector<size_t> count(P);
	for (int r=0; r<lrows; r++) {
	  auto gr = _Hperm[R.rowl2g(r)];
	  auto p = -1 + distance
	    (_dist.begin(), upper_bound(_dist.begin(), _dist.end(), gr));
	  glp[r] = tuple<int,int,int>{gr, r, p};
	  count[p] += lcols;
	}
	sort(glp.begin(), glp.end());
	for (int p=0; p<P; p++)
	  sbuf[p].reserve(count[p]);
      }
      for (int r=0; r<lrows; r++)
	for (int c=0, lr=get<1>(glp[r]), p=get<2>(glp[r]); c<lcols; c++)
	  sbuf[p].push_back(R(lr,c));
    }
	MPIComm c;
	std::vector<myscalar> rbuf;
	std::vector<myscalar*> pbuf;
    c.all_to_all_v(sbuf, rbuf, pbuf);	
    DenseM_t R1D(_Hrows, cols);
    if (_Hrows) {
      vector<int> src_c(cols);
      for (int c=0; c<cols; c++)
	src_c[c] = ((c / B) % _npcols) * _nprows;
      for (int r=0; r<_Hrows; r++) {
	auto gr = _iHperm[r + _dist[rank]];
	auto src_r = (gr / B) % _nprows;
	for (int c=0; c<cols; c++)
	  R1D(r, c) = *(pbuf[src_r + src_c[c]]++);
      }
    }
    return R1D;
  }

  void redistribute_1D_to_2D(DenseM_t& S1D, DistM_t& S) {
    const auto rank = mpi_rank();
    const auto P = mpi_nprocs();
    const auto B = DistM_t::default_MB;
    const auto cols = S1D.cols();
    const auto lcols = S.lcols();
    const auto lrows = S.lrows();
    vector<vector<myscalar>> sbuf(P);
    if (_Hrows) {
      vector<tuple<int,int,int>> glp(_Hrows);
      for (int r=0; r<_Hrows; r++) {
	auto gr = _iHperm[r + _dist[rank]];
	glp[r] = tuple<int,int,int>{gr,r,(gr / B) % _nprows};
      }
      sort(glp.begin(), glp.end());
      vector<int> pc(cols);
      for (int c=0; c<cols; c++)
	pc[c] = ((c / B) % _npcols) * _nprows;
      {
	vector<size_t> count(P);
	for (int r=0; r<_Hrows; r++)
	  for (int c=0, pr=get<2>(glp[r]); c<cols; c++)
	    count[pr+pc[c]]++;
	for (int p=0; p<P; p++) 
	  sbuf[p].reserve(count[p]);
      }
      for (int r=0; r<_Hrows; r++)
	for (int c=0, lr=get<1>(glp[r]), pr=get<2>(glp[r]); c<cols; c++)
	  sbuf[pr+pc[c]].push_back(S1D(lr,c));
    }
	MPIComm c;
	std::vector<myscalar> rbuf;
	std::vector<myscalar*> pbuf;
    c.all_to_all_v(sbuf, rbuf, pbuf);
    if (S.active()) {
      for (int r=0; r<lrows; r++) {
	auto gr = _Hperm[S.rowl2g(r)];
	auto p = -1 + distance
	  (_dist.begin(), upper_bound(_dist.begin(), _dist.end(), gr));
	for (int c=0; c<lcols; c++)
	  S(r,c) = *(pbuf[p]++);
      }
    }
  }

#if FAST_H_SAMPLING == 2 || FAST_H_SAMPLING == 3
  void operator()(DistM_t &R, DistM_t &Sr, DistM_t &Sc) {
    auto starttime = MPI_Wtime();
    int Ncol = R.cols();
    {
    DenseM_t S1D(_Hrows, Ncol);
	DenseM_t R1D = redistribute_2D_to_1D(R);
	z_c_hodlr_mult("N",(cscalar*)(R1D.data()), (cscalar*)(S1D.data()), &_Hrows, &Ncol, &ho_bf, &option, &stats, &ptree);
	redistribute_1D_to_2D(S1D, Sr);	
	
    DenseM_t S1D1(_Hrows, Ncol);
	DenseM_t R1D1 = redistribute_2D_to_1D(R);
	z_c_hodlr_mult("C",(cscalar*)(R1D1.data()), (cscalar*)(S1D1.data()), &_Hrows, &Ncol, &ho_bf, &option, &stats, &ptree);
	redistribute_1D_to_2D(S1D1, Sc);		
	
      
    }
	
    auto endtime = MPI_Wtime();
    if (!mpi_rank())
      cout << "# Apply time " << (endtime-starttime)
           << " seconds, per vector "
           << ((endtime-starttime)/Ncol)
           << " seconds" << endl;		   
		   
  }

#else

  void operator()(DistM_t &R, DistM_t &Sr, DistM_t &Sc) {
    Sr.zero();
    int maxlocrows = R.MB() * (R.rows() / R.MB());
    if (R.rows() % R.MB()) maxlocrows += R.MB();
    int maxloccols = R.MB() * (R.cols() / R.MB());
    if (R.cols() % R.MB()) maxloccols += R.MB();
    DenseM_t tmp(maxlocrows, maxloccols);
    // each processor broadcasts his/her local part of R to all
    // processes in the same column of the BLACS grid, one after the
    // other
    for (int p=0; p<R.nprows(); p++) {
      if (p == R.prow()) {
        strumpack::scalapack::gebs2d
          (R.ctxt(), 'C', ' ', R.lrows(), R.lcols(), R.data(), R.ld());
        DenseMW_t Rdense(R.lrows(), R.lcols(), R.data(), R.ld());
        strumpack::copy(Rdense, tmp, 0, 0);
      } else {
        int recvrows = strumpack::scalapack::numroc
          (R.rows(), R.MB(), p, 0, R.nprows());
        strumpack::scalapack::gebr2d
          (R.ctxt(), 'C', ' ', recvrows, R.lcols(),
           tmp.data(), tmp.ld(), p, R.pcol());
      }
      times(tmp, Sr, p);
    }
    Sc = Sr;
  }
#endif
};


int run(int argc, char *argv[]) {

  double tstart, tend;
  std::string filename, prefix, locfile;
  std::ifstream fp;

  int n=0; 
    // Setting command line arguments
  if (argc > 1) n = stoi(argv[1]);
  int m = n;
  double wavelength;
  if (argc > 2) wavelength = stof(argv[2]);
  
  
  auto P = mpi_nprocs(MPI_COMM_WORLD);
  // initialize the BLACS grid
  int npcol = floor(sqrt((float)P));
  int nprow = P / npcol;
  int prow, pcol;
  
  
  HSSOptions<myscalar> hss_opts;
  hss_opts.set_verbose(false);

  BLACSGrid grid(MPI_COMM_WORLD);
  DistributedMatrix<myscalar> A;

 
  int myid = mpi_rank();

  hss_opts.set_from_command_line(argc, argv);


  double total_time = 0.;
  int ACA = 2;
  if (argc > 2) ACA = stoi(argv[3]);
  
  string reorder("natural");
  reorder = string(argv[4]);
  
  
  TaskTimer::t_begin = GET_TIME_NOW();
  TaskTimer timer(string("compression"), 1);


  
  

  int cluster_size = hss_opts.leaf_size();

  
  if (!mpi_rank())
    cout << "# HSS compression .. " << endl;
  timer.start();

  HSSMatrixMPI<myscalar>* K = nullptr;

  CompressSetup kernel_matrix
    (n, hss_opts, nprow, npcol, ACA, cluster_size, reorder,wavelength);

  auto f0_compress = strumpack::params::flops.load();


	
#if FAST_H_SAMPLING == 3	
    // factor hodlr 	
	z_c_hodlr_factor(&kernel_matrix.ho_bf, &kernel_matrix.option, &kernel_matrix.stats, &kernel_matrix.ptree,&kernel_matrix.msh);		
	
#else	


    K = new HSSMatrixMPI<myscalar>
      (n, n, &grid, kernel_matrix, kernel_matrix,
       hss_opts);


  if (K->is_compressed()) {
    // reduction over all processors
    const auto max_rank = K->max_rank();
    const auto total_memory = K->total_memory();
    if (!mpi_rank())
      cout << "# created K matrix of dimension "
           << K->rows() << " x " << K->cols()
           << " with " << K->levels() << " levels" << endl
           << "# compression succeeded!" << endl
           << "# rank(K) = " << max_rank << endl
           << "# memory(K) = " << total_memory / 1e6 << " MB " << endl;
  } else {
    if (!mpi_rank())
      cout << "# compression failed!!!!!!!!" << endl;
    return 1;
  }

  if (!mpi_rank())
    cout << "#HSS compression took " << timer.elapsed() << endl;
  total_time += timer.elapsed();

  // auto total_flops_compress = Allreduce
    // (strumpack::params::flops.load() - f0_compress, MPI_SUM, MPI_COMM_WORLD);  
  double total_flops_compress=strumpack::params::flops.load() - f0_compress;
  MPI_Allreduce(MPI_IN_PLACE, &total_flops_compress, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (!mpi_rank())
    cout << "# compression flops = " << total_flops_compress << endl;

  // Starting factorization
  if (!mpi_rank())
    cout << "factorization start" << endl;
  timer.start();
  auto f0_factor = strumpack::params::flops.load();
  auto ULV = K->factor();
  const auto ULVmem = ULV.total_memory(MPI_COMM_WORLD);
  if (!mpi_rank())
    cout << "# factorization time = " << timer.elapsed() << endl
	<< "# ULV_memory(K) = " << ULVmem / 1e6 << " MB " << endl;
  total_time += timer.elapsed();
  // auto total_flops_factor = Allreduce
    // (strumpack::params::flops.load() - f0_factor, MPI_SUM, MPI_COMM_WORLD);
  double total_flops_factor=strumpack::params::flops.load() - f0_factor;
  MPI_Allreduce(MPI_IN_PLACE, &total_flops_factor, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);	
  if (!mpi_rank())
    cout << "# factorization flops = " << total_flops_factor << endl;
#endif	




           

	
	
	
  // Starting solve		
  vector<myscalar> rhs_vector(n);
  // normal_distribution<myscalar> normal_distr(0.0,1.0);
  for (int i = 0; i < n; i++) 
  {
    // rhs_vector[i] = normal_distr(generator);
    rhs_vector[i] = 1.0;
  }	
  
  DenseMatrix<myscalar> B(n, 1, &rhs_vector[0], n);
  DenseMatrix<myscalar> weights(B);
  DistributedMatrix<myscalar> Bdist(&grid, n,1);
  Bdist.scatter(B);
  DistributedMatrix<myscalar> wdist(&grid, n,1);
  wdist.scatter(weights);
  
  if (!mpi_rank())
    cout << "solve start" << endl;
  timer.start();
  auto f0_solve = strumpack::params::flops.load();
  
#if FAST_H_SAMPLING == 3  
	DenseMatrix<myscalar> S1D(kernel_matrix._Hrows, 1);
	DenseMatrix<myscalar> R1D = kernel_matrix.redistribute_2D_to_1D(wdist);
	int rhs=1;      
	z_c_hodlr_solve((cscalar*)(S1D.data()), (cscalar*)(R1D.data()), &kernel_matrix._Hrows, &rhs, &kernel_matrix.ho_bf,&kernel_matrix.option, &kernel_matrix.stats, &kernel_matrix.ptree);	
    kernel_matrix.redistribute_1D_to_2D(S1D, wdist);
  
  
	z_c_hodlr_mult("N",(cscalar*)(S1D.data()), (cscalar*)(R1D.data()), &kernel_matrix._Hrows, &rhs, &kernel_matrix.ho_bf,&kernel_matrix.option, &kernel_matrix.stats, &kernel_matrix.ptree);	
	DistributedMatrix<myscalar> Bcheck(&grid, n, 1);
	Bcheck.scatter(B);
    kernel_matrix.redistribute_1D_to_2D(R1D, Bcheck);  
 
#else
  K->solve(ULV, wdist);
  if (!mpi_rank())
    cout << "# solve time = " << timer.elapsed() << endl;
  total_time += timer.elapsed();
  // auto total_flops_solve = Allreduce
    // (strumpack::params::flops.load() - f0_solve, MPI_SUM, MPI_COMM_WORLD);
  double total_flops_solve=strumpack::params::flops.load() - f0_solve;
  MPI_Allreduce(MPI_IN_PLACE, &total_flops_solve, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);	
  if (!mpi_rank())
    cout << "# solve flops = " << total_flops_solve << endl;
		
  if (!mpi_rank())
    cout << "# total time (comp + fact): " << total_time << endl;
	
	auto Bcheck = K->apply(wdist);	
#endif


#if FAST_H_SAMPLING == 2 || FAST_H_SAMPLING == 3 	 
	z_c_hodlr_printstats(&kernel_matrix.stats, &kernel_matrix.ptree);	
#endif  

  Bcheck.scaled_add(-1., Bdist);
  auto Bchecknorm = Bcheck.normF() / Bdist.normF();
  if (!mpi_rank())
    cout << "# relative error = ||B-H*(H\\B)||_F/||B||_F = "
         << Bchecknorm << endl;

		 

//------generate random x vector for test solution accuracy----		 
		 
		 
if(n<=100000){


  vector<myscalar> sample_vector(n);
  // normal_distribution<myscalar> normal_distr(0.0,1.0);
  for (int i = 0; i < n; i++) 
  {
    // sample_vector[i] = normal_distr(generator);
    sample_vector[i] = 1.0;
  }
  DenseMatrix<myscalar> sample_v(n, 1, &sample_vector[0], n);
  

	
	// DistributedMatrix<myscalar> Kdense_dist(&grid, n,n);		
	// int lrows,lcols;
	// int r,c;
	// lrows = scalapack::numroc(Kdense_dist.desc()[2], Kdense_dist.desc()[4], Kdense_dist.prow(), Kdense_dist.desc()[6], Kdense_dist.nprows());
	// lcols = scalapack::numroc(Kdense_dist.desc()[3], Kdense_dist.desc()[5], Kdense_dist.pcol(), Kdense_dist.desc()[7], Kdense_dist.npcols());
	// // double tmp=0;
	// for( int myi=1; myi<= lrows; myi++){
		// r =  l2g(myi,Kdense_dist.prow(),n,Kdense_dist.nprows(),Kdense_dist.desc()[4])-1;
		// for( int myj=1; myj<= lcols; myj++){
			// c =  l2g(myj,Kdense_dist.pcol(),n,Kdense_dist.npcols(),Kdense_dist.desc()[5])-1;
			// // if(!mpi_rank())
				// // cout<<c<<endl;		
				
			// Kdense_dist(myi-1, myj-1) =  Aseq(r,c);

			// // tmp += Kdense_dist(myi-1, myj-1);
		// }
	// }	
  
    DistributedMatrix<myscalar> sample_v_dist(&grid, n,1);
	sample_v_dist.scatter(sample_v);
    DistributedMatrix<myscalar> sample_rhs_dist(sample_v_dist);	
    kernel_matrix.densemult(sample_v_dist, sample_rhs_dist);	
	DistributedMatrix<myscalar> sample_rhs_dist1(sample_rhs_dist);	 //copy exact rhs
	

#if FAST_H_SAMPLING == 3  
	DenseMatrix<myscalar> S1D(kernel_matrix._Hrows, 1);
	DenseMatrix<myscalar> R1D = kernel_matrix.redistribute_2D_to_1D(sample_rhs_dist);
	int rhs=1;      
	z_c_hodlr_solve((cscalar*)(S1D.data()), (cscalar*)(R1D.data()), &kernel_matrix._Hrows, &rhs, &kernel_matrix.ho_bf,&kernel_matrix.option, &kernel_matrix.stats, &kernel_matrix.ptree);	
    kernel_matrix.redistribute_1D_to_2D(S1D, sample_rhs_dist);
#else
  K->solve(ULV, sample_rhs_dist);
#endif  
  
  DistributedMatrix<myscalar> sample_v_dist1(sample_rhs_dist);	 //copy solution vector
  
  
  sample_rhs_dist.scaled_add(-1., sample_v_dist);
  double err_sol=sample_rhs_dist.normF()/sample_v_dist.normF();
  if (!mpi_rank()){
  cout << "# ||X_t-H\\(A*X_t)||_F/||X_t||_F = "<<  err_sol<< endl; 
  }  
  
  kernel_matrix.densemult(sample_v_dist1, sample_rhs_dist);	  
  sample_rhs_dist.scaled_add(-1., sample_rhs_dist1);
  double err_sol1=sample_rhs_dist.normF()/sample_rhs_dist1.normF();
  if (!mpi_rank()){
  cout << "# ||B-A*(H\\B)||_F/||B||_F = "<<  err_sol1<< endl; 
  }   
			
		 
}else{
  if (!mpi_rank()){
  cout << "Not computing ||X_t-H\\(A*X_t)||_F/||X_t||_F and ||B-A*(H\\B)||_F/||B||_F because matrix size too big "<< endl; 
  }  
} 


  return 0;
}


int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  int ierr;
#pragma omp parallel
#pragma omp single nowait
  ierr = run(argc, argv);

  scalapack::Cblacs_exit(1);
  MPI_Finalize();
  return ierr;
}
