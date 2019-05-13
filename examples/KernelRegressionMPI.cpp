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

using namespace std;
using namespace strumpack;
using namespace strumpack::HSS;
using namespace strumpack::kernel;

void print_dense_counter_MPI(std::string description, MPIComm c){
#if defined(STRUMPACK_COUNT_FLOPS)

  std::cout << "### "
            << std::setw(15)
            << description
            << " "
            << "dense_counter_mpi_MB = "
            << std::setw(10)
            << std::left
            << (params::dense_counter)/1.e6 + (params::dense_counter_mpi)/1.e6
            << "    "
            << "peak_dense_counter_mpi_MB = "
            << std::setw(10)
            << std::left
            << (params::peak_dense_counter)/1.e6 + (params::peak_dense_counter_mpi)/1.e6
            << " r"
            << c.rank()
            << std::endl;
  return;
  // 1. Reduce dense counters
  std::array<long long int, 2> red_dense_counter = {
    params::dense_counter.load(),
    params::peak_dense_counter.load()
  };
  c.reduce(red_dense_counter.data(), red_dense_counter.size(), MPI_SUM);

  // 2. Reduce dense_MPI counters
  std::array<long long int, 2> red_dense_counter_mpi = {
    params::dense_counter_mpi.load(),
    params::peak_dense_counter_mpi.load()
  };
  c.reduce(red_dense_counter_mpi.data(), red_dense_counter_mpi.size(), MPI_SUM);

  // 3. Print
  if (c.is_root())
    std::cout << "### "
            << std::left
            << std::setw(15)
            << description
            << " "
            << "dense_mpi_MB = "
            << std::setw(10)
            << std::left
            << (red_dense_counter_mpi[0])/1.e6 + (red_dense_counter[0])/1.e6
            << "    "
            << "peak_dense_mpi_MB = "
            << std::setw(10)
            << std::left
            << (red_dense_counter_mpi[1])/1.e6 + (red_dense_counter[1])/1.e6
            << " "
            << "###  MPI COMPRESSION_REDUCEs"
            << std::endl;

  // 4. Set reduced counters to zero
  std::fill(red_dense_counter.begin(),red_dense_counter.end(),0);
  std::fill(red_dense_counter_mpi.begin(),red_dense_counter_mpi.end(),0);
#endif
}

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
  using scalar_t = float;
  TaskTimer timer_all("all");
  timer_all.start();
  MPI_Init(&argc, &argv);
  MPIComm c;
  {
  string filename("smalltest.dat");
  int d = 2;
  scalar_t h = 3.;
  scalar_t lambda = 1.;
  KernelType ktype = KernelType::GAUSS;
  string mode("test");
  TaskTimer timer("prediction");

  if (c.is_root())
    cout << "# usage: ./KernelRegression file d h lambda "
         << "kernel(Gauss, Laplace) mode(valid, test)" << endl;
  if (argc > 1) filename = string(argv[1]);
  if (argc > 2) d = stoi(argv[2]);
  if (argc > 3) h = stof(argv[3]);
  if (argc > 4) lambda = stof(argv[4]);
  if (argc > 5) ktype = kernel_type(string(argv[5]));
  if (argc > 6) mode = string(argv[6]);
  if (c.is_root())
    cout << "# data dimension  = " << d << endl
         << "# kernel h        = " << h << endl
         << "# lambda          = " << lambda << endl
         << "# kernel type     = " << get_name(ktype) << endl
         << "# validation/test = " << mode << endl;

  auto training     = read_from_file<scalar_t>(filename + "_train.csv");
  auto testing      = read_from_file<scalar_t>(filename + "_" + mode + ".csv");
  auto train_labels = read_from_file<scalar_t>(filename + "_train_label.csv");
  auto test_labels  = read_from_file<scalar_t>(filename + "_" + mode + "_label.csv");

  size_t n = training.size() / d;
  size_t m = testing.size() / d;
  if (c.is_root())
    cout << "# training dataset = " << n << " x " << d << endl
         << "# testing dataset  = " << m << " x " << d << endl << endl;

  DenseMatrixWrapper<scalar_t>
    training_points(d, n, training.data(), d),
    test_points(d, m, testing.data(), d);


  auto K = create_kernel<scalar_t>(ktype, training_points, h, lambda);
  {
    HSSOptions<scalar_t> opts;
    opts.set_verbose(false);
    opts.set_from_command_line(argc, argv);
    if (c.is_root() && opts.verbose())
      opts.describe_options();

    BLACSGrid g(c);
    timer.start();

    auto weights = K->fit_HSS(g, train_labels, opts);
    if (c.is_root()) cout << "# fit_HSS took " << timer.elapsed() << endl;

    #if 0
    // Save weights to binary file
    // Weights filename: "w_h_lambda_.bin", example: w_3.755000_0.700000_.bin
    string weights_filename = opts.scratch_folder() + "/w_" + std::to_string(h)
              + "_" + std::to_string(lambda) + "_.binweights";
    weights.print_to_binary_file(weights_filename);
    if (c.is_root())
      cout << "Saved weights: " << weights_filename << std::endl;

    // Read weights from binary file
    // if (c.is_root()){
    // DenseMatrix<scalar_t> read_weights_bin(n,1);
    //   read_weights_bin.read_from_binary_file("/Users/gichavez/Documents/github/code_pap3_tests/profile_memory/test.bin");
    //   read_weights_bin.print("Read weights", true, 8);
    // }
    #endif

    // Prediction
    auto check = [&](const std::vector<scalar_t>& prediction) {
      // compute accuracy score of prediction
      if (c.is_root()) {
        size_t incorrect_quant = 0;
        for (size_t i=0; i<m; i++)
          if ((prediction[i] >= 0 && test_labels[i] < 0) ||
              (prediction[i] < 0 && test_labels[i] >= 0))
            incorrect_quant++;
        cout << "# prediction score: "
        << (float(m - incorrect_quant) / m) * 100. << "%" << endl
        << "# c-err: "
        << (float(incorrect_quant) / m) * 100. << "%"
        << endl;
      }
    };
    if (c.is_root()) cout << endl << "# HSS prediction start..." << endl;
    auto prediction = K->predict(test_points, weights);
    if (c.is_root()) cout << "# prediction took " << timer.elapsed() << endl;
    check(prediction);
  }

  if (c.is_root())
    std::cout << "# total_time: "
      << timer_all.elapsed() << std::endl << std::endl;
  }
  // print_dense_counter_MPI("SANITY counter", c);
  MPI_Finalize();
  return 0;
}
