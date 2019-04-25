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

using scalar_t = float;
using DistM_t = DistributedMatrix<scalar_t>;
using DistMW_t = DistributedMatrixWrapper<scalar_t>;
using DenseM_t = DenseMatrix<scalar_t>;

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

template<typename T>
void printArr(string NAME, int val, vector<T> arr){
  cout << NAME << " " << val << ": ";
  for(auto x: arr)
    cout << std::fixed << std::setw(3) <<x << " ";
  cout << endl;
}

std::vector<int> getBlockRange(int n, int size, int rank){
  std::vector<int> range;
  int lastBlock = int(std::ceil( scalar_t(n)/ scalar_t(size)));
  int sub_len;
  int sub_start;
  if ((rank == lastBlock-1) && (n % size)){
    // std::cout << "LAST" << " ";
    sub_start = rank * size;
    sub_len   = n % size;
  } else {
    // std::cout << "REST" << " ";
    sub_start = rank * size;
    sub_len   = size;
  }
  range.push_back(sub_start);
  range.push_back(sub_start + sub_len);
  return range;
}

  // template<typename scalar_t>
  // void Kernel<scalar_t>::
  void
  getBlock(std::unique_ptr<Kernel<scalar_t>> &K, DistributedMatrix<scalar_t> &_D, // here
  const DenseMatrix<scalar_t> &test,
  std::vector<int>& vec_rows, std::vector<int>& vec_cols,
  MPIComm c, BLACSGrid *grid) {
    using DistM_t = DistributedMatrix<scalar_t>;

    // MPI Element extraction
    auto Aelem = [&]
      (const std::vector<std::size_t>& I, const std::vector<std::size_t>& J,
      DistM_t& B, std::size_t rlo, std::size_t clo,
      MPI_Comm comm) {
      std::vector<std::size_t> lI, lJ;
      lI.reserve(B.lrows());
      lJ.reserve(B.lcols());
      for (size_t j=0; j<J.size(); j++)
        if (B.colg2p(j) == B.pcol())
          lJ.push_back(J[j]);
      for (size_t i=0; i<I.size(); i++)
        if (B.rowg2p(i) == B.prow())
          lI.push_back(I[i]);
      auto lB = B.dense_wrapper();

      // K->eval_vec(lI, lJ, lB); // operator call
      // eval_vec(lI, lJ, lB); // operator call

      printArr("I",c.rank(),lI);
      printArr("J",c.rank(),lJ);
      cout << "r" << c.rank() << "lI.size() = " << lI.size() << endl;
      cout << "r" << c.rank() << "lJ.size() = " << lJ.size() << endl;
      // cout << lI[i] << "," << lJ[j] << " ";
      // coll2g
      // rowl2g

      for (auto i=0; i<lI.size(); i++) {
        for (auto j=0; j<lJ.size(); j++){
          lB(i, j) = K->eval_kernel_function(K->data_.ptr(0, lJ[j]), test.ptr(0, I[i]));
        }
      }
    };

    int numRows = vec_rows[1] - vec_rows[0];
    int numCols = vec_cols[1] - vec_cols[0];
    // cout << "numRows =" << numRows << endl;
    // cout << "numCols =" << numCols << endl;

    std::vector<std::size_t> I, J;
    I.reserve(numRows);
    J.reserve(numCols);
    for (std::size_t i=vec_rows[0]; i<vec_rows[1]; i++)
      I.push_back(i);
    for (std::size_t j=vec_cols[0]; j<vec_cols[1]; j++)
      J.push_back(j);

    _D = DistM_t(grid, numRows, numCols);
    // Aelem(I, J, _D, 0, 0, comm);
    Aelem(I, J, _D, vec_rows[0], vec_cols[0], c.comm());

    DistM_t _Dt = _D.transpose();
    _Dt.print("_Dt", true, 11);
  }


int main(int argc, char *argv[]) {
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
    cout << "# usage: ./SandboxMPI file d h lambda "
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
    if (c.is_root() && opts.verbose()) opts.describe_options();
    BLACSGrid grid(c);
    timer.start();

    int LB = int(m);
    int NB = std::min(int(n), int(n));
    int numb_rows = int(std::ceil( scalar_t(m)/ scalar_t(LB)));
    int numb_cols = int(std::ceil( scalar_t(n)/ scalar_t(NB)));

    if (c.is_root()){
      std::cout <<  "       m  = " << m << std::endl;
      std::cout <<  "       n  = " << n << std::endl;
      std::cout <<  "       LB = " << LB  << std::endl;
      std::cout <<  "       NB = " << NB << std::endl;
      std::cout <<  "numb_rows = " << numb_rows << std::endl;
      std::cout <<  "numb_cols = " << numb_cols << std::endl;
    }

    DistM_t bKp;
    // vector<int> rowRange{0, 10};
    // vector<int> colRange{0, 10};
    vector<int> rowRange = getBlockRange( int(m), 10, 0); // m
    vector<int> colRange = getBlockRange( int(n), 10, 0);  // n
    getBlock(K, bKp, test_points, rowRange, colRange, c.comm(), &grid);
    // bKp.print("bKp",true,10);

    // // Complete weights matrix
    // DistM_t matW(&grid, n, LB);
    // matW.random();

    // // Complete prediction matrix
    // DistM_t matP(&grid, m, LB);
    // matP.zero();

    // for(int ib = 0; ib < numb_rows; ib++){
    //   vector<int> rowRange = getBlockRange( int(m), int(LB), ib);
    //   // Wrapping prediction matrix
    //   DistMW_t bP(rowRange[1]-rowRange[0], LB, matP, rowRange[0], 0);
    //   // cout << "bP " << rowRange[1]-rowRange[0] << " " << LB << " " << rowRange[0] << " " << 0 << setw(2) << endl;
    //   for(int jb = 0; jb < numb_cols; jb++){
    //     vector<int> colRange = getBlockRange( int(n), int(NB), jb);
    //     // Wrapping weights matrix
    //     DistMW_t bW(colRange[1]-colRange[0], LB, matW, colRange[0], 0);
    //     // Forming block of Kp matrix
    //     DistM_t bKp;
    //     getBlock(K, test_points, bKp, rowRange, colRange, c.comm(), &grid);
    //     // Perform multiplication
    //     gemm(Trans::N, Trans::N, scalar_t(1.0), bKp, bW, scalar_t(1.0), bP);
    //   }
    // }

    // matP.print("matP", 10);

    // cout << K->eval_kernel_function(K->data().ptr(0, ii),
    //   test_points.ptr(0, ii)) << endl;

    // TASK: Form dense blocks
    // vector<int> rowRange = getBlockRange( int(m), int(LB), 0); // LB
    // vector<int> colRange = getBlockRange( int(n), int(7),  0); // m
    // // vector<int> rowRange = getBlockRange( int(m), int(m), 0); // entire K
    // // vector<int> colRange = getBlockRange( int(n), int(n), 0); // entire K
    // DistM_t bKp;
    // getBlock(K, bKp, rowRange, colRange, c.comm(), &grid);
    // bKp.print("B00", 8);

    // gemm(Trans::N, Trans::N, scalar_t(1.0), bKp, bW, scalar_t(0.0), bP);
    // bP.print();

    // TASK: Subvide a length, by chunks of "size"
    #if 0
    if (c.is_root()){

      int numCols = int(std::ceil( scalar_t(n)/ scalar_t(m)));
      cout << "n       = " << n << endl;
      cout << "m       = " << m << endl;
      cout << "numCols = " << numCols << endl;
      for(size_t rank = 0; rank < numCols; rank++){
        vector<int> colRange;
        colRange = getBlockRange( int(n), int(m), rank);
        cout << rank << "| ";
        printArr(colRange);
      }

      int LB = 10;
      int numRows = int(std::ceil( scalar_t(m)/ scalar_t(LB)));
      cout << "m       = " << m << endl;
      cout << "LB      = " << LB << endl;
      cout << "numRows = " << numRows << endl;
      for(size_t rank = 0; rank < numRows; rank++){
        vector<int> rowRange;
        rowRange = getBlockRange( int(m), int(LB), rank);
        cout << rank << "| ";
        printArr(rowRange);
      }

    }
    #endif

  }
  // Kernel scope

  if (c.is_root())
    std::cout << "# total_time: "
      << timer_all.elapsed() << std::endl << std::endl;

  MPI_Finalize();
  return 0;
}
