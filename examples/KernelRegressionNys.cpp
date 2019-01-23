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

inline scalar_t dist2(scalar_t* x, scalar_t* y, int d) {
  scalar_t k = 0.;
  for (int i=0; i<d; i++) k += pow(x[i] - y[i], 2.);
  return k;
}

inline scalar_t Gauss_kernel(scalar_t* x, scalar_t* y, int d, scalar_t h) {
  return exp(-dist2(x, y, d)/(2.*h*h));
}

void printSample(scalar_t* x) {
  int d = 8;
  for (int i = 0; i < d; ++i)
    cout << x[i] << " ";
  cout << endl;
}

void printPermutationVector(std::string NAME, std::vector<int> v) {
  cout << NAME << ": ";
  for(auto &i : v)
    cout << i-1 << " ";
  cout << endl;
}

void printMatrixBlock(DenseMatrix<scalar_t> Mat) {
  for (int i = 0; i < 4; ++i){
    for (int j = 0; j < 4; ++j){
      cout << Mat(i,j) << " ";
    }
    cout << endl;
  }
  cout << endl;
}

inline bool areSameScalarsToEps(scalar_t a, scalar_t b) {
    return fabs(a - b) < 1e-4;
}

int main(int argc, char *argv[]) {
  string filename("smalltest.dat");
  size_t d = 2;
  scalar_t h = 3.;
  scalar_t lambda = 1.;
  KernelType ktype = KernelType::GAUSS;
  string mode("test");
  int M;
  TaskTimer timer("compression");

  cout << "# usage: ./KernelRegression file d h lambda "
       << "kernel(Gauss, Laplace) mode(valid, test)" << std::endl;
  if (argc > 1) filename = string(argv[1]);
  if (argc > 2) d = stoi(argv[2]);
  if (argc > 3) h = stof(argv[3]);
  if (argc > 4) lambda = stof(argv[4]);
  if (argc > 5) ktype = kernel_type(string(argv[5]));
  if (argc > 6) mode = string(argv[6]);
  if (argc > 7) M = stoi(argv[7]);
  cout << std::endl;
  cout << "# data dimension   = " << d << std::endl;
  cout << "# kernel sigma (h) = " << h << std::endl;
  cout << "# lambda           = " << lambda << std::endl;
  cout << "# kernel type      = " << get_name(ktype) << std::endl;
  cout << "# validation/test  = " << mode << std::endl;

  HSSOptions<scalar_t> hss_opts;
  hss_opts.set_verbose(true);
  hss_opts.set_from_command_line(argc, argv);
  // hss_opts.describe_options();
  cout << "# hss_rel_tol      = " << hss_opts.rel_tol() << std::endl;
  cout << "# hss_leaf_size    = " << hss_opts.leaf_size() << std::endl ;
  cout << "# hss_d0           = " << hss_opts.d0() << std::endl ;
  cout << "# hss_dd           = " << hss_opts.dd() << std::endl;
  cout << "# M (Nystrom)      = " << M << std::endl << std::endl;

  cout << "# Reading data ..." << std::endl;
  timer.start();
  // Read from csv files
  auto training     = read_from_file<scalar_t>(filename + "_train.csv");
  auto testing      = read_from_file<scalar_t>(filename + "_" + mode + ".csv");
  auto train_labels = read_from_file<scalar_t>(filename + "_train_label.csv");
  auto test_labels  = read_from_file<scalar_t>(filename + "_" + mode + "_label.csv");
  cout << "## Reading took " << timer.elapsed() << std::endl;

  size_t n = training.size() / d;
  size_t m = testing.size() / d;
  cout << "# training dataset = " << n << " x " << d << std::endl;
  cout << "# testing dataset  = " << m << " x " << d << std::endl;

  // Wrap datasets into DenseMatrixWrapper objects
  DenseMatrixWrapper<scalar_t>
    training_points(d, n, training.data(), d),
    test_points(d, m, testing.data(), d);

  #if 1
    // Generate 2x random numbers than neccesary
    std::vector<std::size_t> Mid(2*M);
    // std::random_device r;
    // std::mt19937 gen(r());
    std::mt19937 gen(1); // reproducible
    std::uniform_int_distribution<std::size_t> dist(0, n-1);
    std::generate(Mid.begin(), Mid.end(), [&]() { return dist(gen); });
    // Sort before call to unique
    std::sort(begin(Mid), end(Mid));
    // Remove duplicates
    auto last = std::unique(begin(Mid), end(Mid));
    Mid.erase(last, end(Mid));
    assert(Mid.size() >= M);
    // Keep only M random numbers
    Mid.resize(M);
    Mid.shrink_to_fit();
  #else
    // M = 10;
    // std::vector<std::size_t> Mid{8148-1,9058-1,1270-1,9132-1,6322-1,975-1,2784-1,5465-1,9568-1,9641-1};
    // M = 100;
    // std::vector<std::size_t> Mid{8148-1,9058-1,1270-1,9132-1,6322-1,975-1,2784-1,5465-1,9568-1,9641-1,1575-1,9696-1,9561-1,4848-1,7992-1,1417-1,4211-1,9142-1,7908-1,9577-1,6545-1,357-1,8473-1,9319-1,6772-1,7559-1,7413-1,3912-1,6537-1,1707-1,7040-1,318-1,2761-1,461-1,969-1,8206-1,6924-1,3160-1,9467-1,344-1,4370-1,3800-1,7624-1,7918-1,1861-1,4876-1,4436-1,6433-1,7060-1,7510-1,2747-1,6763-1,6517-1,1618-1,1184-1,4957-1,9544-1,3385-1,5819-1,2225-1,7468-1,2536-1,5029-1,6947-1,8853-1,9531-1,5437-1,1377-1,1483-1,2558-1,8349-1,2525-1,8085-1,2418-1,9224-1,3474-1,1952-1,2492-1,6113-1,4696-1,3489-1,8241-1,5805-1,5452-1,9095-1,2835-1,7507-1,7472-1,3771-1,5628-1,752-1,535-1,5260-1,7720-1,9253-1,1287-1,5634-1,4649-1,118-1,3338-1};
    M = 500;
    std::vector<std::size_t> Mid{8148-1,9058-1,1270-1,9132-1,6322-1,975-1,2784-1,5465-1,9568-1,9641-1,1575-1,9696-1,9561-1,4848-1,7992-1,1417-1,4211-1,9142-1,7908-1,9577-1,6545-1,357-1,8473-1,9319-1,6772-1,7559-1,7413-1,3912-1,6537-1,1707-1,7040-1,318-1,2761-1,461-1,969-1,8206-1,6924-1,3160-1,9467-1,344-1,4370-1,3800-1,7624-1,7918-1,1861-1,4876-1,4436-1,6433-1,7060-1,7510-1,2747-1,6763-1,6517-1,1618-1,1184-1,4957-1,9544-1,3385-1,5819-1,2225-1,7468-1,2536-1,5029-1,6947-1,8853-1,9531-1,5437-1,1377-1,1483-1,2558-1,8349-1,2525-1,8085-1,2418-1,9224-1,3474-1,1952-1,2492-1,6113-1,4696-1,3489-1,8241-1,5805-1,5452-1,9095-1,2835-1,7507-1,7472-1,3771-1,5628-1,752-1,535-1,5260-1,7720-1,9253-1,1287-1,5634-1,4649-1,118-1,3338-1,1606-1,7863-1,3081-1,5231-1,1640-1,5957-1,2602-1,6471-1,6818-1,7400-1,4456-1,829-1,2265-1,9031-1,1507-1,8164-1,5321-1,9845-1,773-1,4375-1,1054-1,9503-1,46-1,7654-1,8072-1,8579-1,834-1,3948-1,2566-1,7898-1,4259-1,8988-1,1795-1,2603-1,1436-1,1343-1,8575-1,5718-1,5423-1,1430-1,8411-1,6133-1,3460-1,5060-1,3961-1,749-1,2365-1,1216-1,1812-1,2364-1,4111-1,490-1,8890-1,9304-1,4834-1,4817-1,3325-1,8860-1,3635-1,1095-1,7678-1,3835-1,2378-1,3974-1,949-1,1298-1,9265-1,9402-1,5656-1,588-1,2308-1,3472-1,8071-1,152-1,423-1,1661-1,6377-1,7188-1,6363-1,4429-1,5372-1,2910-1,7312-1,1855-1,6742-1,1802-1,3617-1,6140-1,7656-1,796-1,9118-1,7609-1,4775-1,4275-1,4382-1,3004-1,4986-1,5008-1,8015-1,7791-1,6315-1,3710-1,7952-1,5221-1,3436-1,9198-1,9875-1,5388-1,6096-1,5748-1,2034-1,2949-1,4610-1,2256-1,8263-1,1906-1,2211-1,1671-1,2228-1,4262-1,3043-1,9030-1,4207-1,1807-1,8847-1,9578-1,4290-1,1086-1,2522-1,3994-1,5813-1,2562-1,5889-1,9937-1,2166-1,1147-1,2897-1,3113-1,4141-1,4958-1,835-1,9769-1,7817-1,286-1,9062-1,7125-1,4766-1,5643-1,2314-1,4475-1,9391-1,5331-1,5081-1,2258-1,4765-1,6082-1,6618-1,3854-1,3580-1,9624-1,368-1,8621-1,8894-1,7753-1,962-1,2550-1,3265-1,6616-1,1329-1,7019-1,1039-1,6361-1,4808-1,7578-1,6955-1,8789-1,8664-1,3250-1,6794-1,1923-1,297-1,7232-1,4860-1,4664-1,8791-1,5925-1,6001-1,8348-1,7823-1,5601-1,1777-1,2330-1,8607-1,279-1,4755-1,1630-1,9498-1,6916-1,4856-1,4571-1,579-1,6615-1,412-1,693-1,5058-1,938-1,7932-1,7925-1,7002-1,1453-1,6392-1,5025-1,9427-1,6287-1,9737-1,4396-1,4188-1,9986-1,809-1,1290-1,1679-1,3784-1,8047-1,7775-1,586-1,3863-1,5097-1,4032-1,6354-1,6074-1,2824-1,4174-1,150-1,9513-1,1616-1,1027-1,3599-1,1915-1,4732-1,3280-1,9193-1,9848-1,509-1,7126-1,2599-1,4083-1,5290-1,9101-1,4033-1,9488-1,9819-1,6765-1,6429-1,5201-1,6734-1,9648-1,1718-1,1235-1,9634-1,1650-1,315-1,5410-1,8500-1,6449-1,1836-1,3555-1,4439-1,9457-1,9886-1,8240-1,6210-1,3624-1,1839-1,4123-1,4640-1,1161-1,5674-1,2177-1,3701-1,5609-1,2423-1,2794-1,5936-1,2552-1,7928-1,9449-1,7021-1,3306-1,5615-1,1036-1,8710-1,8453-1,7858-1,2505-1,5710-1,217-1,4085-1,9805-1,1551-1,1717-1,4060-1,905-1,5745-1,4520-1,6679-1,6716-1,6127-1,323-1,660-1,3066-1,5091-1,6276-1,3909-1,7862-1,6887-1,9285-1,5093-1,3116-1,1013-1,5854-1,7461-1,4057-1,870-1,9617-1,1472-1,2691-1,4214-1,5047-1,4379-1,8379-1,9761-1,9779-1,6102-1,9163-1,2303-1,6468-1,2765-1,6425-1,6647-1,651-1,2436-1,2142-1,6384-1,8070-1,3292-1,7458-1,6453-1,65-1,5752-1,3695-1,8748-1,11-1,4416-1,4052-1,4400-1,7352-1,3078-1,7489-1,4498-1,342-1,1678-1,6885-1,4517-1,1457-1,3253-1,5792-1,1829-1,9970-1,2315-1,8744-1,2565-1,7295-1,1798-1,2739-1,868-1,5489-1,6509-1,5206-1,4054-1,6136-1,6166-1,6464-1,6052-1,8996-1,1989-1,6749-1,2248-1,1136-1,5777-1,4282-1,4363-1,6295-1,7324-1,3330-1,6294-1,3956-1,8002-1,7916-1,2437-1,5829-1};
    cout << "# Nystrom centers  = " << M << std::endl;
  #endif

  // Take a random subset Mid of size Mid.size() from the original dataset
  DenseMatrix<scalar_t> training_Nystrom(d, M);
  for (std::size_t j=0; j<M; j++)
    for (std::size_t i=0; i<d; i++)
      training_Nystrom(i, j) = training_points(i, Mid[j]);
  // training_Nystrom.print("training_Nystrom", true, 10);

  scalar_t l0(0.);
  // Kernel object setup for the complete dataset
  auto K = create_kernel<scalar_t>(ktype, training_points, h, l0);
  // Kernel object setup for Nystrom (reduced) dataset
  auto K_Nystrom = create_kernel<scalar_t>(ktype, training_Nystrom, h, l0);

  //here
  #if 0
  // Clustering
  std::cout << "Clustering ..." << std::endl;
  timer.start();
  std::vector<int> perm;
  // training_Nystrom.print("bef_training_Nystrom", true, 10);
  auto t = binary_tree_clustering(hss_opts.clustering_algorithm(),
     K_Nystrom->data(), perm, hss_opts.leaf_size()); //Permutes training_Nystrom
  // printPermutationVector("perm", perm);
  // training_Nystrom.print("aft_training_Nystrom", true, 10);

  // Permute columns of matrix training_Nystrom
  // training_Nystrom.lapmt(perm, true);
  // Get inverse permutation
  // std::vector<int> iperm;
  // iperm.resize(perm.size());
  // for (std::size_t i=0; i<M; i++)
  //   iperm[perm[i]-1] = i+1;
  cout << "## Clustering took: " << timer.elapsed() << std::endl;
  #endif

  std::cout << "Forming Kmm ..." << std::endl;
  timer.start();
  DenseMatrix<scalar_t> Kmm(M, M);
  for (std::size_t j=0; j<M; j++)
    for (std::size_t i=0; i<M; i++)
      Kmm(i, j) = K_Nystrom->eval(i, j); // Nystrom -reduced- kernel object
  cout << "## Elapsed: " << timer.elapsed() << std::endl;
  // Kmm.print("Kmm",true,10);

  std::cout << "Forming Knm ..." << std::endl;
  timer.start();
  DenseMatrix<scalar_t> Knm(n, M);
  for (std::size_t j=0; j<M; j++)
    for (std::size_t i=0; i<n; i++)
      Knm(i, j) = K->eval(i, Mid[j]); // Full kernel object
      // Knm(i, j) = K->eval(i, Mid[iperm[j]-1]);
  cout << "## Elapsed: " << timer.elapsed() << std::endl;
  // Knm.print("Knm",true,10);

  std::cout << "Forming H dense ..."  << std::endl;
  std::cout << "H = (KnM^t)*(KnM) + n*lambda*(Kmm)" << std::endl;
  timer.start();
  DenseMatrix<scalar_t> Hdense(Kmm);
  gemm(Trans::T, Trans::N, scalar_t(1.), Knm, Knm,
       scalar_t(n*lambda), Hdense);
  cout << "## Elapsed: " << timer.elapsed() << std::endl;
  // Hdense.print("Hdense",true,10);
  DenseMatrix<scalar_t> Hc(Hdense); // Copy to check compression error
  // printMatrixBlock(Hdense);
  // cout << "Hdense.norm() = " << Hdense.norm() << endl;
  // cout << "Hdense(1,0) = " << Hdense(1,0) << endl;
  // vector<scalar_t> eigs = Hdense.singular_values();
  // cout << "===> eigs[0] = " << eigs[0] << endl;

  // #if 0
  // // Clustering
  // std::cout << "Clustering ..." << std::endl;
  // timer.start();
  // std::vector<int> perm;
  // auto t = binary_tree_clustering(hss_opts.clustering_algorithm(),
  //    K_Nystrom->data(), perm, hss_opts.leaf_size()); // this permutes data
  // printPermutationVector("perm", perm);
  // // Permute columns of matrix training_Nystrom
  // // training_Nystrom.lapmt(perm, true); //permutes columns
  // // Get inverse permutation
  // // std::vector<int> iperm;
  // // iperm.resize(perm.size());
  // // for (std::size_t i=0; i<M; i++)
  // //   iperm[perm[i]-1] = i+1;
  // cout << "## Clustering took: " << timer.elapsed() << std::endl;
  // #endif

  #if 1
  // Clustering
  std::cout << "Clustering ..." << std::endl;
  timer.start();
  std::vector<int> perm;
  auto t = binary_tree_clustering(hss_opts.clustering_algorithm(),
     K_Nystrom->data(), perm, hss_opts.leaf_size()); // this permutes data
  // training_Nystrom.lapmt(perm, true); y tho???
  // Get inverse permutation
  std::vector<int> iperm;
  iperm.resize(perm.size());
  for (std::size_t i=0; i<M; i++)
    iperm[perm[i]-1] = i+1;
  // Add one to use LAPACK routines
  for (int i = 0; i < Mid.size(); ++i){
    perm[i] = perm[i] + 1;
    iperm[i] = iperm[i] + 1;
  }
  // See permutation vectors
  if ( Mid.size() <= 64){
    printPermutationVector(" perm", perm);
    printPermutationVector("iperm", iperm);
  }
  cout << "## Clustering took: " << timer.elapsed() << std::endl;
  #endif

  // Compression to HSS
  std::cout << "# H compression to HSS ..." << std::endl;
  timer.start();
  HSSMatrix<scalar_t> H(t, hss_opts);
  H.compress(Hdense, hss_opts);

  if (H.is_compressed())
    std::cout << "# created HSS matrix of dimension "
              << H.rows() << " x " << H.cols()
              << " with " << H.levels() << " levels" << std::endl
              << "# compression succeeded!" << std::endl;
  else std::cout << "# compression failed!!!" << std::endl;

  cout << "# rank(H) = " << H.rank() << std::endl;
  cout << "# memory(H) = " << H.memory()/1e6 << " MB, "
       << 100. * H.memory() / Hdense.memory() << "% of dense" << std::endl;
  cout << "## Compression took: " << timer.elapsed() << std::endl;

  auto HSSd = H.dense();
  // Hdense.print("Hdense",true,10);
  // Hc.print("Hc",true,10);
  HSSd.scaled_add(-1., Hc);
  // HSSd.print("HSSd",true,10);
  cout << "# relative error = ||HSSd-Hd||_F/||Hd||_F = "
       << HSSd.normF() / Hdense.normF() << endl;

  // Factorization and solve
  cout << "# Factorization ..." << std::endl;
  timer.start();
  auto ULV = H.factor();
  cout << "## Factorization took: " << timer.elapsed() << std::endl;

  cout << "# Solve ..." << std::endl;
  timer.start();
  // z = (Knm') * y;
  // weights = H\z;
  DenseMatrix<scalar_t> y(n, 1, &train_labels[0], n);
  DenseMatrix<scalar_t> z(M, 1);
  gemm(Trans::T, Trans::N, scalar_t(1.), Knm, y, scalar_t(0.), z);
  // z.print("z",true,10);
  // cout << "z.norm() = " << z.norm() << endl;

  // Need to permute z before?

  DenseMatrix<scalar_t> weights(z);

  H.solve(ULV, weights); // Inexact solve not good enough for c-err

  // Need to inverse permute weights?
  // weights.lapmr(iperm, true); //permutes rows
  //

  // weights.print("weights",false,10);
  // cout << "weights.norm() = " << weights.norm() << endl;
  cout << "## Solve took: " << timer.elapsed() << std::endl;

  // Prediction
  cout << "# Prediction ..." << endl;
  timer.start();

  DenseMatrix<scalar_t> Kr(m,M);
  for (int c = 0; c < m; c++)
    for (int r = 0; r < M; r++)
      Kr(c, r) = Gauss_kernel(&testing[c*d], &training[Mid[r]*d], d, h);
  // Kr.print("Kr",true,10);

  // Matrix vector product to compute predictions
  DenseMatrix<scalar_t> prediction(m, 1);
  gemm(Trans::N, Trans::N, scalar_t(1.), Kr, weights, scalar_t(0.), prediction);
  // prediction.print("prediction", false, 10);
  cout << "prediction.norm() = " << prediction.norm() << endl;

  // Quantify threshold
  for (int i = 0; i < m; ++i)
    prediction(i,0) = ((prediction(i,0) > 0) ? 1. : -1.);

  // Compute accuracy score of the predictions
  scalar_t incorrect_quant = 0;
  for (int i = 0; i < m; ++i) {
    scalar_t a = (prediction(i,0) - test_labels[i]) / 2;
    incorrect_quant += (a > 0 ? a : -a);
  }

  scalar_t c_err = 100.0 - (((m - incorrect_quant) / m) * 100);
  // cout << "## Prediction took :" << timer.elapsed() << std::endl;
  cout << "## c-err: " << c_err << "%" << std::endl;

  return 0;
}
