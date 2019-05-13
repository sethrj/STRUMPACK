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
 * Developers: Pieter Ghysels, Gustavo Chavez, Xiaoye S. Li.
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

using namespace std;
using namespace strumpack;
using namespace strumpack::HSS;
using namespace strumpack::kernel;

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
  TaskTimer total_time("total_time");
  total_time.start();
  using scalar_t = float;
  string filename("smalltest.dat");
  size_t d = 2;
  scalar_t h = 3.;
  scalar_t lambda = 1.;
  KernelType ktype = KernelType::GAUSS;
  string mode("test");

  cout << "# usage: ./KernelRegression_Multiple file d h lambda "
       << "kernel(Gauss, Laplace) mode(valid, test)" << endl;
  if (argc > 1) filename = string(argv[1]);
  if (argc > 2) d = stoi(argv[2]);
  if (argc > 3) h = stof(argv[3]);
  if (argc > 4) lambda = stof(argv[4]);
  if (argc > 5) ktype = kernel_type(string(argv[5]));
  if (argc > 6) mode = string(argv[6]);
  cout << endl;
  cout << "# data dimension  = " << d << endl;
  cout << "# kernel h        = " << h << endl;
  cout << "# lambda          = " << lambda << endl;
  cout << "# kernel type     = " << get_name(ktype) << endl;
  cout << "# validation/test = " << mode << endl;

  HSSOptions<scalar_t> hss_opts;
  hss_opts.set_verbose(false);
  hss_opts.set_from_command_line(argc, argv);
  if (hss_opts.verbose())
    hss_opts.describe_options();
  TaskTimer timer("misc");
  cout << "# rtol            = " << hss_opts.rel_tol() << endl << endl;

  cout << "# Reading data ..." << endl;
  timer.start();
  auto training     = read_from_file<scalar_t>(filename + "_train.csv");
  auto testing      = read_from_file<scalar_t>(filename + "_" + mode + ".csv");
  auto train_labels = read_from_file<scalar_t>(filename + "_train_label.csv");
  auto test_labels  = read_from_file<scalar_t>(filename + "_" + mode + "_label.csv");
  cout << "# Reading took " << timer.elapsed() << endl;

  size_t n = training.size() / d;
  size_t m = testing.size() / d;
  cout << "# training dataset (n) = " << n << " x " << d << endl;
  cout << "# testing dataset  (m) = " << m << " x " << d << endl << endl;
  DenseMatrixWrapper<scalar_t> training_points(d, n, training.data(), d),
                               test_points(d, m, testing.data(), d);

  std::vector<scalar_t> lambda_vec {
    1e0, 3e0, 6e0, 9e0,
    1e1, 3e1, 6e1, 9e1,
    1e2, 3e2, 6e2, 9e2,
    1e3, 3e3, 6e3, 9e3
  };
  // std::vector<scalar_t> lambda_vec {5e-0};

  auto K = create_kernel<scalar_t>(ktype, training_points, h, lambda);
  DenseMatrix<scalar_t> weights = K->fit_HSS_multiple(train_labels, hss_opts, lambda_vec);

  cout << "# prediction start..." << endl;
  timer.start();
  DenseMatrix<scalar_t> prediction = K->predict_multiple(test_points, weights);
  cout << "# prediction took " << timer.elapsed() << endl << endl;

  // weights.print("weights", false, 10);
  // prediction.print("prediction", false, 12);

  // compute accuracy score of prediction
  scalar_t best_cerr = 100.;
  int idx_best_cerr = -1;
  for(int w = 0; w<weights.cols(); w++){
    size_t incorrect_quant = 0;
    for (size_t i=0; i<m; i++){
      if ((prediction(i,w) >= 0 && test_labels[i] < 0) ||
        (prediction(i,w) < 0 && test_labels[i] >= 0))
        incorrect_quant++;
    }
    scalar_t c_err = (scalar_t(incorrect_quant) / m) * 100.;
    // cout << "# c-err: " << fixed << setprecision(2)
    //     << c_err << "%" << endl;
    cout << "# c-err: " << fixed << setprecision(2)
      << c_err << "%" << "   lambda = "
      << std::scientific << lambda_vec[w] << endl;
    cout << std::fixed;
    if( c_err <= best_cerr ){
      best_cerr = c_err;
      idx_best_cerr = w;
    }
  }

  // cout << "# best_c_err: " << best_cerr << " at " << idx_best_cerr << endl << endl;
  // cout << "# total_time: " << defaultfloat<< total_time.elapsed() << endl << endl;
  cout << endl;
  cout << "# best_c_err: " << std::fixed << std::setw(2) << best_cerr;
  cout << " with lambda =  " << std::scientific << lambda_vec[idx_best_cerr]
  << endl << endl;
  std::cout << "# total_time: "
  << std::fixed << std::setw(2) << total_time.elapsed()
  << std::endl << std::endl;

  return 0;
  }
