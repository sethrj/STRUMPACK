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
#include <dirent.h> // C header for opening files in directory

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

std::vector<float> parseFileName(string file){
  std::vector<float> params(2,0);
  std::vector<float> posVec;
  string toSearch = "_";
  // Find all ocurrences "toSearch"
	size_t pos = file.find(toSearch);
	while( pos != std::string::npos){
		posVec.push_back(pos);
		pos = file.find(toSearch, pos + toSearch.size());
	}
  std::string str_hval = file.substr(posVec[0]+1,posVec[1]-posVec[0]-1);
  std::string str_lval = file.substr(posVec[1]+1,posVec[2]-posVec[1]-1);
  params[0] = stof(str_hval);
  params[1] = stof(str_lval);
  return params;
}

int main(int argc, char *argv[]) {
  using scalar_t = float;
  TaskTimer timer_all("all");
  timer_all.start();
  MPI_Init(&argc, &argv);
  MPIComm c;
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

    // Compression, factorization and solve
    // auto weights = K->fit_HSS(g, train_labels, opts);
    // if (c.is_root()) cout << "# fit_HSS took " << timer.elapsed() << endl;

    // Reading weights from binary file on root rank
    int number_weights = 0;
    DenseMatrix<scalar_t> all_read_weights;

    if (c.is_root()){
      vector<vector<scalar_t>> vec_weights;
      // Open directory
      DIR *dir;
      struct dirent *ent;
      if (( dir = opendir (opts.scratch_folder().c_str() )) != NULL) {
        while ((ent = readdir (dir)) != NULL) {
          string file = ent->d_name;
          std::size_t pos = file.find("binweights");
          if (pos != string::npos){
            // std::cout << file << std::endl;
            std::vector<float> params = parseFileName(file);
            // std::cout << "hval = " << std::fixed << setprecision(3) << params[0] << std::endl;
            // std::cout << "lval = " << std::fixed << setprecision(3) << params[1] << std::endl;
            number_weights++;
            DenseMatrix<scalar_t> read_weight(n,1);
            std::string filepath = opts.scratch_folder() + "/" + file;
            read_weight.read_from_binary_file(filepath);
            // read_weight.print(file,true,8);
            std::vector<scalar_t> tmp(read_weight.data(), read_weight.data() + n);
            vec_weights.push_back(tmp); // Collects all weights in a vec<vec>
          }
        }
        closedir (dir);
      } else {
        std::cout << "Could not open directory" << std::endl;
        return -1;
      }
      cout << "Found " << number_weights << " weight files" << endl;
      // Combine all weights
      vector<scalar_t> allc;
      allc.reserve(number_weights*n);
      for (auto& items: vec_weights)
          std::move(items.begin(), items.end(), std::back_inserter(allc));
      all_read_weights = DenseMatrix<scalar_t>(n, number_weights, allc.data(), n);
      // all_read_weights.print("all_read_weights",true,12);
    }
    // Comunicate number of weights
    MPI_Bcast(&number_weights, 1, MPI_INT, 0, c.comm());
    DistributedMatrix<scalar_t> weights(&g, n, number_weights);
    weights.scatter(all_read_weights);
    // weights.print();

    // // Prediction lambda
    // auto check = [&](const std::vector<scalar_t>& prediction) {
    //   // compute accuracy score of prediction
    //   if (c.is_root()) {
    //     size_t incorrect_quant = 0;
    //     for (size_t i=0; i<m; i++)
    //       if ((prediction[i] >= 0 && test_labels[i] < 0) ||
    //           (prediction[i] < 0 && test_labels[i] >= 0))
    //         incorrect_quant++;
    //     cout << "# prediction score: "
    //     << (float(m - incorrect_quant) / m) * 100. << "%" << endl
    //     << "# c-err: "
    //     << (float(incorrect_quant) / m) * 100. << "%"
    //     << endl;
    //   }
    // };
    // if (c.is_root()) cout << endl << "# HSS prediction start..." << endl;
    // auto prediction = K->predict(test_points, weights);
    // if (c.is_root()) cout << "# prediction took " << timer.elapsed() << endl;
    // check(prediction);

  }
  // Kernel scope

  if (c.is_root())
    std::cout << "# total_time: "
      << timer_all.elapsed() << std::endl << std::endl;

  MPI_Finalize();
  return 0;
}
