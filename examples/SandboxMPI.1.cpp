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

// void print_dense_counter_MPI(std::string description, MPIComm c){
// #if defined(STRUMPACK_COUNT_FLOPS)
//   std::cout << "### "
//           << std::left
//           << std::setw(15)
//           << description
//           << " "
//           << "dense_MB = "
//           << std::setw(10)
//           << std::left
//           << params::dense_counter_mpi/1.e6
//           << "    "
//           << "peak_dense_MB = "
//           << std::setw(10)
//           << std::left
//           << params::peak_dense_counter_mpi/1.e6
//           << " "
//           << "### BEFORE_REDUCE rank[" << c.rank() << "]"
//           << std::endl;
//   // 1. Reduce
//   std::array<long long int, 2> red_dense_counter_mpi = {
//     params::dense_counter_mpi.load(),
//     params::peak_dense_counter_mpi.load()
//   };
//   c.reduce(red_dense_counter_mpi.data(), red_dense_counter_mpi.size(), MPI_SUM);

//   // 2. Print
//   if (c.is_root())
//     std::cout << "### "
//             << std::left
//             << std::setw(15)
//             << description
//             << " "
//             << "dense_MB = "
//             << std::setw(10)
//             << std::left
//             << red_dense_counter_mpi[0]/1.e6
//             << "    "
//             << "peak_dense_MB = "
//             << std::setw(10)
//             << std::left
//             << red_dense_counter_mpi[1]/1.e6
//             << " "
//             << "###"
//             << std::endl;
//   red_dense_counter_mpi[0] = 0;
//   red_dense_counter_mpi[1] = 0;
// #endif
// }

// template<typename scalar_t> vector<scalar_t>
// read_from_file(string filename) {
//   vector<scalar_t> data;
//   ifstream f(filename);
//   string l;
//   while (getline(f, l)) {
//     istringstream sl(l);
//     string s;
//     while (getline(sl, s, ','))
//       data.push_back(stod(s));
//   }
//   data.shrink_to_fit();
//   return data;
// }

// int main(int argc, char *argv[]) {
//   using scalar_t = float;
//   TaskTimer timer_all("all");
//   timer_all.start();
//   MPI_Init(&argc, &argv);
//   MPIComm c;

  // string filename("smalltest.dat");
  // int d = 2;
  // scalar_t h = 3.;
  // scalar_t lambda = 1.;
  // KernelType ktype = KernelType::GAUSS;
  // string mode("test");
  // TaskTimer timer("prediction");

  // if (c.is_root())
  //   cout << "# usage: ./KernelRegression file d h lambda "
  //        << "kernel(Gauss, Laplace) mode(valid, test)" << endl;
  // if (argc > 1) filename = string(argv[1]);
  // if (argc > 2) d = stoi(argv[2]);
  // if (argc > 3) h = stof(argv[3]);
  // if (argc > 4) lambda = stof(argv[4]);
  // if (argc > 5) ktype = kernel_type(string(argv[5]));
  // if (argc > 6) mode = string(argv[6]);
  // if (c.is_root())
  //   cout << "# data dimension  = " << d << endl
  //        << "# kernel h        = " << h << endl
  //        << "# lambda          = " << lambda << endl
  //        << "# kernel type     = " << get_name(ktype) << endl
  //        << "# validation/test = " << mode << endl << endl;

  // HSSOptions<scalar_t> opts;
  // opts.set_verbose(false);
  // opts.set_from_command_line(argc, argv);
  // if (c.is_root() && opts.verbose())
  //   opts.describe_options();

  // auto training     = read_from_file<scalar_t>(filename + "_train.csv");
  // auto testing      = read_from_file<scalar_t>(filename + "_" + mode + ".csv");
  // auto train_labels = read_from_file<scalar_t>(filename + "_train_label.csv");
  // auto test_labels  = read_from_file<scalar_t>(filename + "_" + mode + "_label.csv");

  // size_t n = training.size() / d;
  // size_t m = testing.size() / d;
  // if (c.is_root())
  //   cout << "# training dataset = " << n << " x " << d << endl
  //        << "# testing dataset  = " << m << " x " << d << endl << endl;

  // DenseMatrixWrapper<scalar_t>
  //   training_points(d, n, training.data(), d),
  //   test_points(d, m, testing.data(), d);

  // auto K = create_kernel<scalar_t>(ktype, training_points, h, lambda);
  // {
  //   BLACSGrid g(c);
  //   timer.start();
  //   auto weights = K->fit_HSS(g, train_labels, opts);
  //   if (c.is_root()) cout << "# fit_HSS took " << timer.elapsed() << endl;
  // }

  // {
  // print_dense_counter_MPI("p0", c);
  // DenseMatrix<scalar_t> mat1(1000,1000); //4MB
  // DenseMatrix<scalar_t> mat2(1000,1000); //4MB
  // DenseMatrix<scalar_t> mat3(1000,1000); //4MB
  // print_dense_counter_MPI("p1", c);      // mem 24 peak 24
  // {
  // DenseMatrix<scalar_t> mat3(1000,1000); //4MB
  // print_dense_counter_MPI("p2", c);      // mem 32 peak 32
  // }
  // print_dense_counter_MPI("p3", c);      // mem 24 peak 32
  // }
  // print_dense_counter_MPI("end", c);      // mem 0 peak 32

//   if (c.is_root())
//     std::cout << "# total_time: "
//       << timer_all.elapsed() << std::endl << std::endl;

//   MPI_Finalize();
//   return 0;
// }

int main(int argc, char *argv[]) {
  using scalar_t = float;
  using DistM_t = DistributedMatrix<scalar_t>;
  using DenseM_t = DenseMatrix<scalar_t>;

  // TaskTimer timer_all("all");
  // timer_all.start();
  MPI_Init(&argc, &argv);
  MPIComm comm;
  {
  BLACSGrid grid(comm);

  int n = 10;
  std::vector<scalar_t> lambda_vec {1,2,3};
  int number_weights = lambda_vec.size();

  // cout << "r" << c.rank() << " grid.P() = " << grid.P() << endl; // 4
  // cout << "r" << c.rank() << " grid.nprows() = " << grid.nprows() << endl; //2
  // cout << "r" << c.rank() << " grid.npcols() = " << grid.npcols() << endl; //2
  // cout << "r" << c.rank() << " grid.prow() = " << grid.prow() << endl; // unk
  // cout << "r" << c.rank() << " grid.pcol() = " << grid.pcol() << endl; // unk

  // DistM_t weights(&grid, n, n, 4, 4);
  // weights.random();
  // weights.print();

  // if (c.is_root()){
  //   // std::vector<scalar_t> tmp(weight.data(), weight.data() + n());
  //   cout << "r" << c.rank() << " weights.lrows() = " << weights.lrows() << endl;
  //   cout << "r" << c.rank() << " weights.lcols() = " << weights.lcols() << endl;
  // }

  // TASK: Explore contents of matrices in a given rank
  // Get entries on rank 2
  // if (c.rank()==2){
  //   // std::vector<scalar_t> tmp(weight.data(), weight.data() + n());
  //   cout << "r" << c.rank() << " weights.lrows() = " << weights.lrows() << endl;
  //   cout << "r" << c.rank() << " weights.lcols() = " << weights.lcols() << endl;
  //   int rlo; int rhi; int clo; int chi;
  //   weights.lranges(rlo, rhi, clo, chi);
  //   //  Loops column major (rows is the fast loop)
  //   for (int c=clo; c<chi; ++c){
  //     for (int r=rlo; r<rhi; ++r){
  //       std::cout << weights(r,c) << std::endl;
  //     }
  //     std::cout << std::endl;
  //   }
  // }

  // TASK: Mutiply 2 DistM_t matrices
  // DistM_t matA(&grid, n, n, 4, 4);
  // matA.random();
  // matA.print();

  // DistM_t matB(&grid, n, n, 4, 4);
  // matB.fill(41);
  // matB.print();

  // DistM_t matC(&grid, n, n, 4, 4);
  // gemm(Trans::N, Trans::N, scalar_t(1.0), matA, matB, scalar_t(0.0), matC);
  // matC.print();

  // cout << "weights.rowblocks() = " << weights.rowblocks() << endl;
  // cout << "weights.colblocks() = " << weights.colblocks() << endl;

  // for(auto ilambda: lambda_vec){
  //   DistM_t mat1(&grid, n, 1);
  //   mat1.random();
  //   weights.hconcat(mat1);
  // }
  // weights.print();


  // TASK: touch all entries of distributed matrix, within if guard
  // n= 8;
  // DistM_t weights(&grid, n, n, 4, 4);
  // weights.random();
  // // weights.print();

  // float cnt = 0;
  // if (weights.active() && weights.lrows() && weights.lcols()) {
  //     for (std::size_t r=0; r<weights.lrows(); r++){
  //       for (std::size_t c=0; c<weights.lcols(); c++){
  //         cout << comm.rank() << " " << weights(r, c) << endl;
  //         cnt += weights(r, c);
  //       }
  //       cout << endl;
  //     }
  // }
  // weights.Comm().all_reduce(&cnt, 1, MPI_SUM);
  // if (comm.is_root())
  //   cout << "cnt = " << cnt << endl;

  // TASK: generate a portion of kernel matrix




  } // comm scope

  // if (c.is_root())
  // std::cout << "# total_time: "
  //   << timer_all.elapsed() << std::endl << std::endl;

  MPI_Finalize();
  return 0;
}
