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

#include "misc/TaskTimer.hpp"
#include "kernel/KernelRegression.hpp"

using namespace std;
using namespace strumpack;
using namespace strumpack::HSS;
using namespace strumpack::kernel;
using scalar_t = float;

extern int dense_memory_counter;

#ifdef STRUMPACK_COUNT_FLOPS
# define STR_DEBUG_ONLY(cmd) cmd;
# define STR_RELEASE_ONLY(cmd)
#else
# define STR_DEBUG_ONLY(cmd)
# define STR_RELEASE_ONLY(cmd) cmd;
#endif

int main(int argc, char *argv[]) {

  STR_DEBUG_ONLY(
    std::cout << "hello from STR_DEBUG_ONLY" << std::endl;
  )

  STR_RELEASE_ONLY(
    std::cout << "hello from STR_RELEASE_ONLY" << std::endl;
  )

  STRUMPACK_DENSE_ADD_MEM(20);
  STRUMPACK_DENSE_ADD_MEM(645);
  STRUMPACK_DENSE_SUB_MEM(45);
  params::print_dense_counter("Testing");

  std::size_t n = 1024;
  DenseMatrix<scalar_t> ann1(n, n);
  ann1.fill(41.0);

  std::cout << "hey1" << std::endl;
  {
  DenseMatrix<scalar_t> ann2(n*2, n*2);
  ann2.fill(41.0);
  }
  std::cout << "hey2" << std::endl;

  DenseMatrix<scalar_t> ann3(n/2, n/2);
  ann3.fill(41.0);

  DenseMatrix<scalar_t> ann4;
  ann4 = ann1;

  // std::cout << "Hello world" << std::endl;
  // // ann.clear();

  // std::cout << "dense_memory_counter = " <<  dense_memory_counter << std::endl;
  // std::cout << "Next it's return" << std::endl;


  // std::vector<int> myvector;
  // for (int i=0; i<100; i++) myvector.push_back(i);

  // std::cout << "size: " << (int) myvector.size() << '\n';
  // std::cout << "capacity 1: " << (int) myvector.capacity() << '\n';
  // // std::cout << "max_size: " << myvector.max_size() << '\n';

  // myvector.shrink_to_fit();
  // std::cout << "capacity 2: " << (int) myvector.capacity() << '\n';

  return 0;
}
