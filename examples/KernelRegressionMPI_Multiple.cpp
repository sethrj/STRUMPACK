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
  using DistM_t = DistributedMatrix<scalar_t>;
  using DenseM_t = DenseMatrix<scalar_t>;

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
    cout << "# usage: ./KernelRegressionMPI_Multiple file d h lambda "
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

    std::vector<scalar_t> lambda_vec {1e-2, 5e-2, 1e-1, 5e-1, 1e-0, 5e-0, 1e+1,
     5e+1, 1e+2, 5e+2, 1e+3, 5e+3, 1e+4, 5e+4, 1e+5, 5e+5, 1e+6, 5e+6};
    // std::vector<scalar_t> lambda_vec {5e-0};
    if (c.is_root()) cout << endl << "# HSS fit_HSS_multiple start..." << endl;
    DistM_t weights = K->fit_HSS_multiple(g, train_labels, opts, lambda_vec);
    if (c.is_root()) cout << "# fit_HSS_multiple took " << timer.elapsed() << endl;

    if (c.is_root()) cout << endl << "# HSS predict_multiple start..." << endl;
    timer.start();
    DistributedMatrix<scalar_t> prediction =
      K->predict_multiple(test_points, weights, c.comm(), &g);
    if (c.is_root()) cout << "# predict_multiple took "
      << timer.elapsed() << endl << endl;

    // weights.print("weights", false, 10);
    prediction.print("prediction", false, 12);


    // Gather to master/root rank, and compute prediction
    DenseMatrix<scalar_t> local_pred = prediction.gather();
    if(c.is_root()){
      // compute accuracy score of prediction
      scalar_t best_cerr = 100.;
      int idx_best_cerr = -1;
      for(int w = 0; w<weights.cols(); w++){
        size_t incorrect_quant = 0;
        for (size_t i=0; i<m; i++){
          if ((local_pred(i,w) >= 0 && test_labels[i] < 0) ||
            (local_pred(i,w) < 0 && test_labels[i] >= 0))
            incorrect_quant++;
        }
        scalar_t c_err = (scalar_t(incorrect_quant) / m) * 100.;
        cout << "# c-err: " << fixed << setprecision(2)
             << c_err << "%" << "   lambda = "
             << std::scientific << lambda_vec[w] << endl;
        cout << std::fixed;
        if( c_err <= best_cerr ){
          best_cerr = c_err;
          idx_best_cerr = w;
        }
      }
      cout << endl;
      cout << "# best_c_err: " << std::fixed << std::setw(2) << best_cerr;
      cout << " with lambda =  " << std::scientific << lambda_vec[idx_best_cerr]
           << endl << endl;
    }
  } // K

  if (c.is_root())
    std::cout << "# total_time: "
      << std::fixed << std::setw(2) << timer_all.elapsed()
      << std::endl << std::endl;
  } // comm

  // print_dense_counter_MPI("SANITY counter", c);
  MPI_Finalize();
  return 0;
}
