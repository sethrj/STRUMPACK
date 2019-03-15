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

int main(int argc, char *argv[]) {
  TaskTimer timer("compressions");
  string filename("smalltest.dat");
  string folder(".");
  size_t d = 2;
  scalar_t h = 3.;
  scalar_t lambda = 1.;
  KernelType ktype = KernelType::GAUSS;
  string mode("test");

  cout << "# usage: ./KernelRegressionANNdriver file d h lambda "
  << "kernel(Gauss, Laplace) mode(valid, test) folder" << std::endl;
  if (argc > 1) filename   = string(argv[1]);
  if (argc > 2) d          = stoi(argv[2]);
  if (argc > 3) h          = stof(argv[3]);
  if (argc > 4) lambda     = stof(argv[4]);
  if (argc > 5) ktype      = kernel_type(string(argv[5]));
  if (argc > 6) mode       = string(argv[6]);
  if (argc > 7) folder     = string(argv[7]);
  cout << std::endl;
  cout << "# data dimension   = " << d << std::endl;
  cout << "# kernel sigma (h) = " << h << std::endl;
  cout << "# lambda           = " << lambda << std::endl;
  cout << "# kernel type      = " << get_name(ktype) << std::endl;
  cout << "# validation/test  = " << mode << std::endl;

  HSSOptions<scalar_t> hss_opts;
  hss_opts.set_verbose(false);
  hss_opts.set_from_command_line(argc, argv);
  if (hss_opts.verbose())
    hss_opts.describe_options();
  cout << "# hss_rel_tol      = " << hss_opts.rel_tol() << std::endl;
  cout << "# hss_abs_tol      = " << hss_opts.abs_tol() << std::endl;
  cout << "# hss_leaf_size    = " << hss_opts.leaf_size() << std::endl ;
  cout << "# hss_d0           = " << hss_opts.d0() << std::endl ;
  cout << "# hss_dd           = " << hss_opts.dd() << std::endl;

  cout << "# Reading data..." << std::endl;
  timer.start();
  auto training     = read_from_file<scalar_t>(filename + "_train.csv");
  auto testing      = read_from_file<scalar_t>(filename + "_" + mode + ".csv");
  auto train_labels = read_from_file<scalar_t>(filename + "_train_label.csv");
  auto test_labels  = read_from_file<scalar_t>(filename + "_" + mode + "_label.csv");
  cout << "# Reading took " << timer.elapsed() << std::endl << std::endl;

  size_t n = training.size() / d;
  size_t m = testing.size() / d;
  cout << "# training dataset = " << n << " x " << d << std::endl;
  cout << "# testing dataset  = " << m << " x " << d << std::endl;

  DenseMatrixWrapper<scalar_t> training_points(d, n, training.data(), d);
  auto K = create_kernel<scalar_t>(ktype, training_points, h, lambda);

  // std::cout << "Forming Knn..." << std::endl;
  // timer.start();
  // DenseMatrix<scalar_t> Knn(n, n);
  // for (std::size_t j=0; j<n; j++)
  //   for (std::size_t i=0; i<n; i++)
  //     Knn(i, j) = K->eval(i, j);
  // cout << "## Elapsed: " << timer.elapsed() << std::endl;

  std::mt19937 gen(1);
  int kann = hss_opts.approximate_neighbors();

  string ann_filename = folder+"/"+"ann_"+std::to_string(kann)+"_"
                        +std::to_string(n)+".binmatrix";
  string scores_filename = folder+"/"+"scores_"+std::to_string(kann)+"_"
                        +std::to_string(n)+".binmatrix";

  if (FILE *file = fopen(ann_filename.c_str(), "r")) {
    fclose(file);
    DenseMatrix<std::uint32_t> ann(kann,n);
    DenseMatrix<scalar_t> scores(kann,n);
    std::cout << std::endl << "Found ANN matrices files, reading" << std::endl;
    ann.read_from_binary_file(ann_filename);
    scores.read_from_binary_file(scores_filename);
    ann.print("ann_read",true);
    scores.print("scores_read",true);
  } else {
    DenseMatrix<std::uint32_t> ann;
    DenseMatrix<scalar_t> scores;
    timer.start();
    std::cout << std::endl << "Computing ANN..." << std::endl;
    find_approximate_neighbors(K->data(), hss_opts.ann_iterations(),
      kann, ann, scores, gen);
    std::cout << "## k-ANN = " << kann
    << " approximate neighbor search time = "
    << timer.elapsed() << std::endl;
    std::cout << "Saving ANN matrices to file" << std::endl;
    ann.print_to_binary_file(ann_filename);
    scores.print_to_binary_file(scores_filename);
  }

  // Check writing and reading
  // ann.zero();
  // scores.zero();
  // ann.read_from_binary_file(ann_filename);
  // scores.read_from_binary_file(scores_filename);
  // ann.print("ann_read",true);
  // scores.print("scores_read",true);
  return 0;
}
