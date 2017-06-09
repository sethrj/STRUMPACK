#include <iostream>
#include <random>
using namespace std;

#include "DenseMatrix.hpp"
#include "HSS/HSSMatrix.hpp"
using namespace strumpack;
using namespace strumpack::HSS;

int run(int argc, char* argv[]) {
  int m = 100, n = 1;

  HSSOptions<double> hss_opts;

  auto usage = [&]() {
    std::cout << "# Usage:\n"
    << "#     OMP_NUM_THREADS=4 ./test1 problem options [HSS Options]\n"
    << "# where:\n"
    << "#  - problem: a char that can be\n"
    << "#      'T': solve a Toeplitz problem\n"
    << "#            options: m (matrix dimension)\n"
    << "#      'f': read matrix from file (binary)\n"
    << "#            options: filename\n";
    hss_opts.describe_options();
    exit(1);
  };

  DenseMatrix<double> A;

  char test_problem = 'T';
  if (argc > 1) test_problem = argv[1][0];
  else usage();
  switch (test_problem) {
  case 'T': { // Toeplitz
    if (argc > 2) m = std::stoi(argv[2]);
    if (argc <= 2 || m < 0) {
      std::cout << "# matrix dimension should be positive integer" << std::endl;
      usage();
    }
    A = DenseMatrix<double>(m, m);
    for (int j=0; j<m; j++)
      for (int i=0; i<m; i++)
	if (i > j) A(i,j) = 0.;
	else A(i,j) = (i==j) ? 1. : 1./(1+abs(i-j));
  } break;
  case 'f': { // matrix from a file
    std::string filename;
    if (argc > 2) filename = argv[2];
    else {
      std::cout << "# specify a filename" << std::endl;
      usage();
    }
    std::cout << "Opening file " << filename << std::endl;
    std::ifstream file(filename, std::ifstream::binary);
    file.read(reinterpret_cast<char*>(&m), sizeof(int));
    A = DenseMatrix<double>(m, m);
    file.read(reinterpret_cast<char*>(A.data()), sizeof(double)*m*m);
  } break;
  default:
    usage();
    exit(1);
  }
  hss_opts.set_from_command_line(argc, argv);

  A.print("A");
  cout << "# tol = " << hss_opts.rel_tol() << endl;

  HSSMatrix<double> H(A, hss_opts);
  if (H.is_compressed()) {
    cout << "# created H matrix of dimension " << H.rows() << " x " << H.cols()
	 << " with " << H.levels() << " levels" << endl;
    cout << "# compression succeeded!" << endl;
  } else {
    cout << "# compression failed!!!!!!!!" << endl;
    return 1;
  }
  cout << "# rank(H) = " << H.rank() << endl;
  cout << "# memory(H) = " << H.memory()/1e6 << " MB, "
       << 100. * H.memory() / A.memory() << "% of dense" << endl;

  // H.print_info();
  auto Hdense = H.dense();
  Hdense.scaled_add(-1., A);
  cout << "# relative error = ||A-H*I||_F/||A||_F = " << Hdense.normF() / A.normF() << endl;

  if (!H.leaf()) {
    double beta = 0.;
    HSSMatrix<double>* H0 = H.child(0);
    DenseMatrix<double> B0(H0->cols(), H0->cols()),
      C0_check(H0->rows(), B0.cols());
    B0.random();
    DenseMatrixWrapper<double> A0(H0->rows(), H0->cols(), A, 0, 0);

    auto C0 = H0->apply(B0);
    gemm(Trans::N, Trans::N, 1., A0, B0, beta, C0_check);
    C0.scaled_add(-1., C0_check);
    cout << "# relative error = ||H0*B0-A0*B0||_F/||A0*B0||_F = " << C0.normF() / C0_check.normF() << endl;
    apply_HSS(Trans::C, *H0, B0, beta, C0);
    gemm(Trans::C, Trans::N, 1., A0, B0, beta, C0_check);
    C0.scaled_add(-1., C0_check);
    cout << "# relative error = ||H0'*B0-A0'*B0||_F/||A0'*B0||_F = " << C0.normF() / C0_check.normF() << endl;
  }
  if (!H.leaf()) {
    double beta = 0.;
    HSSMatrix<double>* H1 = H.child(1);
    DenseMatrix<double> B1(H1->cols(), H1->cols()),
      C1_check(H1->rows(), B1.cols());
    B1.random();
    DenseMatrixWrapper<double> A1(H1->rows(), H1->cols(), A, 0, 0);

    auto C1 = H1->apply(B1);
    gemm(Trans::N, Trans::N, 1., A1, B1, beta, C1_check);
    C1.scaled_add(-1., C1_check);
    cout << "# relative error = ||H1*B1-A1*B1||_F/||A1*B1||_F = " << C1.normF() / C1_check.normF() << endl;
    apply_HSS(Trans::C, *H1, B1, beta, C1);
    gemm(Trans::C, Trans::N, 1., A1, B1, beta, C1_check);
    C1.scaled_add(-1., C1_check);
    cout << "# relative error = ||H1'*B1-A1'*B1||_F/||A1'*B1||_F = " << C1.normF() / C1_check.normF() << endl;
  }

  default_random_engine gen;
  uniform_int_distribution<std::size_t> random_idx(0,m-1);
  cout << "# extracting individual elements, avg error = ";
  double ex_err = 0;
  int iex = 5;
  for (int i=0; i<iex; i++) {
    auto r = random_idx(gen);
    auto c = random_idx(gen);
    ex_err += std::abs(H.get(r,c) - A(r,c));
  }
  cout << ex_err/iex << std::endl;

  std::vector<std::size_t> I, J;
  auto nI = 8; //random_idx(gen);
  auto nJ = 8; //random_idx(gen);
  for (int i=0; i<nI; i++) I.push_back(random_idx(gen));
  for (int j=0; j<nJ; j++) J.push_back(random_idx(gen));
  cout << "# extracting I=[";
  for (auto i : I) { std::cout << i << " "; } cout << "];\n#            J=[";
  for (auto j : J) { std::cout << j << " "; } cout << "];" << endl;
  auto sub = H.extract(I, J);
  auto sub_dense = A.extract(I, J);
  // sub.print_to_file("sub", "sub.m");
  // sub_dense.print_to_file("sub_dense", "sub_dens.m");
  sub.print("sub");
  sub.scaled_add(-1., sub_dense);
  // sub.print("sub_error");
  cout << "# sub-matrix extraction error = " << sub.normF() / sub_dense.normF() << endl;

  cout << "# computing ULV factorization of HSS matrix .. " << endl;
  auto ULV = H.factor();
  cout << "# solving linear system .. " << endl;

  DenseMatrix<double> B(m, n);
  B.random();
  DenseMatrix<double> C(B);
  H.solve(ULV, C);
  auto Bcheck = H.apply(C);
  Bcheck.scaled_add(-1., B);
  cout << "# relative error = ||B-H*(H\\B)||_F/||B||_F = " << Bcheck.normF() / B.normF() << endl;

  if (!H.leaf()) {
    auto partialULV = H.partial_factor();
    cout << "# Computing Schur update .. " << endl;
    DenseMatrix<double> Theta, Phi, DUB01;
    H.Schur_update(partialULV, Theta, DUB01, Phi);
    // Theta.print("Theta");
    // Phi.print("Phi");
    // TODO check the Schur update
  }

  cout << "# exiting" << endl;
  return 0;
}


int main(int argc, char* argv[]) {
#pragma omp parallel
#pragma omp single nowait
  run(argc, argv);
}
