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
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "kernel/KernelRegression.hpp"
#include "misc/TaskTimer.hpp"
#include "dense/DenseMatrix.hpp"
#include "sparse/GMRes.hpp"
#include "sparse/BiCGStab.hpp"
#include "clustering/NeighborSearch.hpp"
#include "StrumpackOptions.hpp"

using namespace std;
using namespace strumpack;
using namespace strumpack::HSS;
using namespace strumpack::kernel;
using scalar_t = double;

template<typename scalar_t> vector<scalar_t>
read_from_file(string filename) {
  vector<scalar_t> data;
  ifstream f(filename);
  string l;
  while (getline(f, l)) {
    istringstream sl(l);
    string s;
    while (getline(sl, s, ','))
      data.push_back(stod(s));
  }
  data.shrink_to_fit();
  return data;
}

// inline scalar_t dist2(scalar_t* x, scalar_t* y, int d) {
//   scalar_t k = 0.;
//   for (int i=0; i<d; i++) k += pow(x[i] - y[i], 2.);
//     return k;
// }

// inline scalar_t Gauss_kernel(scalar_t* x, scalar_t* y, int d, scalar_t h) {
//   return exp(-dist2(x, y, d)/(2.*h*h));
// }

// void printSample(scalar_t* x) {
//   int d = 8;
//   for (int i = 0; i < d; ++i)
//     cout << x[i] << " ";
//   cout << endl;
// }

// void printPermutationVector(std::string NAME, std::vector<int> v) {
//   cout << NAME << ": ";
//   for(auto &i : v)
//     cout << i-1 << " ";
//   cout << endl;
// }

// void printMatrixBlock(DenseMatrix<scalar_t> Mat) {
//   for (int i = 0; i < 4; ++i){
//     for (int j = 0; j < 4; ++j){
//       cout << Mat(i,j) << " ";
//     }
//     cout << endl;
//   }
//   cout << endl;
// }

// inline bool areSameScalarsToEps(scalar_t a, scalar_t b) {
//   return fabs(a - b) < 1e-4;
// }

// void print_const_scalar_t(string str, const scalar_t* XXX){
//   cout << "const_" << str << endl;
//   cout << XXX[0] << endl;
//   cout << XXX[1] << endl;
// }

// void print_scalar_t(string str, scalar_t* XXX){
//   cout << str << endl;
//   cout << XXX[0] << endl;
//   cout << XXX[1] << endl;
// }

int main(int argc, char *argv[]) {
  // TaskTimer timer("compression");
  // string filename("smalltest.dat");
  // size_t d = 2;
  // scalar_t h = 3.;
  // scalar_t lambda = 1.;
  // KernelType ktype = KernelType::GAUSS;
  // string mode("test");

  // cout << "# usage: ./KernelRegressionANNdriver file d h lambda "
  // << "kernel(Gauss, Laplace) mode(valid, test) M gmres_it gmres_rtol" << std::endl;
  // if (argc > 1) filename   = string(argv[1]);
  // if (argc > 2) d          = stoi(argv[2]);
  // if (argc > 3) h          = stof(argv[3]);
  // if (argc > 4) lambda     = stof(argv[4]);
  // if (argc > 5) ktype      = kernel_type(string(argv[5]));
  // if (argc > 6) mode       = string(argv[6]);
  // // if (argc > 7) M          = stoi(argv[7]);
  // // if (argc > 8) gmres_it   = stoi(argv[8]);
  // // if (argc > 9) gmres_rtol = stof(argv[9]);
  // cout << std::endl;
  // cout << "# data dimension   = " << d << std::endl;
  // cout << "# kernel sigma (h) = " << h << std::endl;
  // cout << "# lambda           = " << lambda << std::endl;
  // cout << "# kernel type      = " << get_name(ktype) << std::endl;
  // cout << "# validation/test  = " << mode << std::endl;

  // HSSOptions<scalar_t> hss_opts;
  // hss_opts.set_verbose(true);
  // hss_opts.set_from_command_line(argc, argv);
  // hss_opts.describe_options();
  // cout << "# hss_rel_tol      = " << hss_opts.rel_tol() << std::endl;
  // cout << "# hss_abs_tol      = " << hss_opts.abs_tol() << std::endl;
  // cout << "# hss_leaf_size    = " << hss_opts.leaf_size() << std::endl ;
  // cout << "# hss_d0           = " << hss_opts.d0() << std::endl ;
  // cout << "# hss_dd           = " << hss_opts.dd() << std::endl;

  // cout << "# Reading data..." << std::endl;
  // timer.start();
  // auto training     = read_from_file<scalar_t>(filename + "_train.csv");
  // auto testing      = read_from_file<scalar_t>(filename + "_" + mode + ".csv");
  // auto train_labels = read_from_file<scalar_t>(filename + "_train_label.csv");
  // auto test_labels  = read_from_file<scalar_t>(filename + "_" + mode + "_label.csv");
  // cout << "# Reading took " << timer.elapsed() << std::endl << std::endl;

  // size_t n = training.size() / d;
  // size_t m = testing.size() / d;
  // cout << "# training dataset = " << n << " x " << d << std::endl;
  // cout << "# testing dataset  = " << m << " x " << d << std::endl;

  // DenseMatrixWrapper<scalar_t> training_points(d, n, training.data(), d);
  // auto K = create_kernel<scalar_t>(ktype, training_points, h, lambda);

  // // std::cout << "Forming Knn..." << std::endl;
  // // timer.start();
  // // DenseMatrix<scalar_t> Knn(n, n);
  // // for (std::size_t j=0; j<n; j++)
  // //   for (std::size_t i=0; i<n; i++)
  // //     Knn(i, j) = K->eval(i, j);
  // //  cout << "## Elapsed: " << timer.elapsed() << std::endl;

  // std::mt19937 gen(1);
  // DenseMatrix<std::uint32_t> ann;
  // DenseMatrix<scalar_t> scores;

  // timer.start();
  // std::cout << "Computing ANN..." << std::endl;

  // find_approximate_neighbors(K->data(), hss_opts.ann_iterations(),
  //   hss_opts.approximate_neighbors(), ann, scores, gen);

  // std::cout << "## k-ANN = " << hss_opts.approximate_neighbors()
  // << " approximate neighbor search time = "
  // << timer.elapsed() << std::endl;

  return 0;
}
