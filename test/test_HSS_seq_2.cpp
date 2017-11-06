// #define CHECK_ERROR_DENSE 1
#define CHECK_ERROR_RANDOMIZED 1

#include <iostream>
#include <random>

using namespace std;

#include "DenseMatrix.hpp"
#include "HSS/HSSMatrix.hpp"
using namespace strumpack;
using namespace strumpack::HSS;


class IUV {
  using DenseM_t = DenseMatrix<double>;
public:
  double _alpha = 0.;
  double _beta = 0.;
  DenseM_t _U, _V;
  IUV() {}
  IUV(double alpha, double beta, int m, int rank, int decay_val) :
    _alpha(alpha), _beta(beta) {
    _U = DenseM_t(m, rank);
    _V = DenseM_t(m, rank);
    _U.random();
    _V.random();
    DenseM_t tmpV(m, rank);
    DenseM_t tmpD(rank, rank);
    tmpV.copy(_V);
    for (int j=0; j<rank; j++)
      for (int i=0; i<rank; i++)
        tmpD(i,j) = (i == j) ? (_beta * exp2(-decay_val*double(j)/rank))
          : double(0.);
    // Simulate an exponential decay of singular values, starting at 1
    // and ending at machine epsilon for double (2^-53) or single (2^-24).
    // Choosing 2^-24 allows us to distinguish between
    // 1e-2, 1e-4, 1e-6, and 1e-8 rel tol.
    gemm(Trans::N, Trans::N, 1., tmpV, tmpD, 0., _V);
    tmpD.clear();
    tmpV.clear();
  }
  void operator()(DenseM_t& Rr, DenseM_t& Rc, DenseM_t& Sr, DenseM_t& Sc) {
    DenseM_t tmp(_U.cols(), Rr.cols());
    gemm(Trans::C, Trans::N, 1., _V, Rr, 0., tmp);
    gemm(Trans::N, Trans::N, 1., _U, tmp, 0., Sr);
    Sr.scaled_add(_alpha, Rr);

    gemm(Trans::C, Trans::N, 1., _U, Rc, 0., tmp);
    gemm(Trans::N, Trans::N, 1., _V, tmp, 0., Sc);
    Sc.scaled_add(_alpha, Rc);
  }

  void operator()(const vector<size_t>& I,
                  const vector<size_t>& J, DenseM_t& B) {
    assert(I.size() == B.rows() && J.size() == B.cols());
    for (size_t j=0; j<J.size(); j++)
      for (size_t i=0; i<I.size(); i++) {
        assert(I[i] < _U.rows() && J[j] < _V.rows());
        if (I[i] == J[j]) B(i,j) = _alpha;
        else B(i,j) = 0.;
        for (size_t k=0; k<_U.cols(); k++)
          B(i,j) += _U(I[i], k) * _V(J[j], k);
      }
  }
};


int run(int argc, char* argv[]) {
  int m = 100;
  int rk;
  double alpha;
  double beta;
  int decay_val;

  HSSOptions<double> hss_opts;

  auto usage = [&]() {
    cout << "# Usage:\n"
    << "#     OMP_NUM_THREADS=4 ./test1 options [HSS Options]\n"
    << "#            options: m (matrix dimension)\n"
    << "#                     r (off-diagonal rank)\n"
    << "#                     alpha (constant)\n"
    << "#                     beta (constant)\n"
    << "#                     decay_val (int: larger gives faster decay)\n";
    hss_opts.describe_options();
    exit(1);
  };

  if (argc > 5) {
    m = stoi(argv[1]);
    rk = stoi(argv[2]);
    alpha = stod(argv[3]);
    beta = stod(argv[4]);
    decay_val = stoi(argv[5]);
  }
  if (argc <= 5 || m <= 0 || rk <= 0) {
    cout<< "# matrix dimension should be positive integer" <<endl;
    cout<< "# matrix-free rank should be positive integer" <<endl;
    usage();
  }
  cout << "# This will perform a matrix-free compression" << endl;
  cout << "# rank  = " << rk << endl;
  cout << "# size  = " << m << endl;
  cout << "# alpha = " << alpha << endl;
  cout << "# beta = " << beta << endl;
  cout << "# decay_val = " << decay_val << endl;

  IUV Amf(alpha, beta, m, rk, decay_val);

  hss_opts.set_from_command_line(argc, argv);

  cout << "# rel tol = " << hss_opts.rel_tol() << endl;
  cout << "# abs tol = " << hss_opts.abs_tol() << endl;
  cout << "# d0 = " << hss_opts.d0() << endl;
  cout << "# dd = " << hss_opts.dd() << endl;

  HSSMatrix<double> H(m, m, hss_opts);

  auto start = std::chrono::system_clock::now();
  H.compress(Amf, Amf, hss_opts);
  auto end = std::chrono::system_clock::now();

  std::chrono::duration<double> elapsed_seconds = end-start;
  std::cout << "## Compression elapsed time: " << elapsed_seconds.count() << "s\n";

  if (H.is_compressed()) {
    cout << "# created H matrix of dimension "
         << H.rows() << " x " << H.cols()
         << " with " << H.levels() << " levels" << endl;
    cout << "# compression succeeded!" << endl;
  } else {
    cout << "# compression failed!!!!!!!!" << endl;
    return 1;
  }

  // cout << "# GC: Printing some info:" << endl;
  // H.print_info();

  double Amem = m*m*sizeof(double);
  cout << "## Max rank(H) = " << H.rank() << endl;
  cout << "## Memory(H) = " << H.memory()/1e6 << " MB, "
       << 100. * H.memory() / Amem << "% of dense" << endl;

#if defined(CHECK_ERROR_DENSE)
  DenseMatrix<double> I(m, m), A(m, m), At(m, m);
  I.eye();
  A.zero();
  At.zero();
  Amf(I, I, A, At);
  auto Hdense = H.dense();
  Hdense.scaled_add(-1., A);
  auto HnormF = Hdense.normF();
  auto AnormF = A.normF();
  cout << "# Relative error to dense= ||A-H*I||_F/||A||_F = "
       << HnormF / AnormF << endl;
#endif

#if defined(CHECK_ERROR_RANDOMIZED)
  int r = 100; // Size of matrix to test compression
  DenseMatrix<double> norm_check(m, r), A_norm_est(m, r),
    At_norm_est(m, r);
  norm_check.random();
  Amf(norm_check, norm_check, A_norm_est, At_norm_est);
  At_norm_est.clear();
  DenseMatrix<double> H_norm_check = H.apply(norm_check);
  H_norm_check.scaled_add(-1., A_norm_est);
  cout << "## Relative error to samples("
       << r << ") = ||(A*R)-(H*R)||_F/||A*R||_F = "
       << H_norm_check.normF() / A_norm_est.normF() << endl;
  A_norm_est.clear();
#endif

  return 0;
}


int main(int argc, char* argv[]) {
#pragma omp parallel
#pragma omp single nowait
  run(argc, argv);
}
