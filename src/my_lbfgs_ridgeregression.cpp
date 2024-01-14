#include "my_lbfgs.hpp"
#include <iostream>
#include <iomanip>

#include <cmath>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;
using namespace Rcpp;

std::vector<std::tuple<int, double, double, arma::vec>> progressValues;

//' class L_BFGS: using L-BFGS algorithm to solve the Optimization Problems
//'               where the target_objection is ridge regression.
//' 
//' @name L_BFGS
//' @description
//' The function includes initialization of the class(L-BFGS()), 
//' the ridge regression loss function(ridge_objective()),
//' function to monitor the progress of the optimization process(monitorProgress()), 
//' the solving function of minimize loss by iteration(monitorProgress()),
//' and complete solution of the function by my_lbfgs::lbfgs_optimize(getResult())
class L_BFGS
{
private:
  arma::mat dataX;
  arma::vec dataY;
  const double lambda; 
  
  // 优化超参数
  int mem_size = 8; // 限制内存大小
  double g_epsilon = 1.0e-5; // 梯度收敛
  int past = 3; // 收敛迭代次数
  double delta = 1.0e-6; // 收敛测试
  int max_iterations = 0; // 最大迭代次数
  
  
public:

  // 初始化
  L_BFGS(arma::mat dataX_, arma::vec dataY_, double lambda_, int mem_size_, 
         int max_iterations_, double g_epsilon_, int past_, double delta_):
  dataX(dataX_), dataY(dataY_), lambda(lambda_), mem_size(mem_size_), 
  max_iterations(max_iterations_), g_epsilon(g_epsilon_), past(past_), delta(delta_){
    if(dataX.n_rows!=dataY.n_rows){
      cout<<"Error! The dimensions are not equal."<<endl;
    }
  }

  // 岭回归目标函数
  static double ridge_objective(arma::mat dataX, arma::vec datay, void *instance, const arma::vec &beta, arma::vec &grad) {
    arma::mat X = dataX;
    arma::vec y = datay;
    double lambda = lambda; 

    int n = X.n_rows;
    // int p = X.n_cols;
    arma::vec predictions = X * beta;
    arma::vec residual = predictions - y;
    double mse = arma::accu(residual % residual) / (2.0 * n);
    double ridge_penalty = 0.5 * lambda * arma::dot(beta, beta);
    double cost = mse + ridge_penalty;
    grad = X.t() * residual / n + lambda * beta;
    return cost;
  }
  
  static int monitorProgress(void *instance,
                             const arma::vec &beta,
                             const arma::vec &g,
                             const double fx,
                             const double step,
                             const int k,
                             const int ls)
  { 
    progressValues.emplace_back(k, fx, arma::max(arma::abs(g)), beta);
    std::cout << std::setprecision(4)
              << "================================" << std::endl
              << "Iteration: " << k << std::endl
              << "Function Value: " << fx << std::endl
              << "Gradient Inf Norm: " << arma::max(arma::abs(g)) << std::endl
              << "Variables: " << std::endl
              << beta.t() << std::endl;
    return 0;
  }

public:
  int getResult(const int N)
  {
    double finalCost;
    arma::vec beta = arma::zeros<arma::vec>(N);
    
    // 设置优化参数
    my_lbfgs::lbfgs_parameter_t params;
    params.max_iterations = max_iterations;
    params.mem_size = mem_size;
    params.g_epsilon = g_epsilon;
    params.past = past;
    params.delta = delta;

    // 开始优化
    int ret = my_lbfgs::lbfgs_optimize(dataX, dataY,
                                    beta,
                                    finalCost,
                                    ridge_objective,
                                    nullptr,
                                    monitorProgress,
                                    this,
                                    params);
    
    // 输出结果
    std::cout << std::setprecision(4)
              << "================================" << std::endl
              << "L-BFGS Optimization Returned: " << ret << std::endl
              << "Minimized Cost: " << finalCost << std::endl
              << "Optimal Variables: " << std::endl
              << beta.t() << std::endl;
    
    return ret;
  }

};

//' Solve ridge regression using L-BFGS.
//' 
//' @param X the independent variable X (size n×p)
//' @param y the response variable y (size n×1)
//' @param lambda the penalty factor (size 1×1)
//' @param men_size the number of corrections to approximate the inverse hessian matrix
//' @param max_iterations the maximum number of iterations
//' @param g_epsilon the epsilon for grad convergence test
//' @param past the distance for delta-based convergence test
//' @param delta for convergence test
//' @return Estimated value of targetValues, beta(last and historical), iterations and loss
// [[Rcpp::export]]
Rcpp::List RidgeRegression_LBFGS(arma::mat X, arma::vec y, double lambda, int mem_size=8, 
                           int max_iterations=64, double g_epsilon=1.0e-5, int past=3, double delta=1.0e-6){
  
  L_BFGS lbfgs_solver(X, y, lambda, mem_size, max_iterations, g_epsilon, past, delta);
  int iter_num = 0;
  int m=X.n_cols;
  iter_num = lbfgs_solver.getResult(m);
  arma::vec K = arma::zeros<arma::vec>(max_iterations);
  arma::vec f_values = arma::zeros<arma::vec>(max_iterations);
  arma::vec grad_values = arma::zeros<arma::vec>(max_iterations);
  arma::mat beta_values = arma::zeros<arma::mat>(max_iterations, m);

  
  int index = 0;
  for (const auto &progress : progressValues) {
    int k;
    double fx, gradientNorm;
    arma::vec beta;
    std::tie(k, fx, gradientNorm, beta) = progress;
    
    K(index) = k;
    f_values(index) = fx;
    grad_values(index) = gradientNorm;
    beta_values.row(index) = beta.t();

    ++index;
  }
  
  return Rcpp::List::create(Rcpp::Named("targetObjection") = f_values.head(index),
                            Rcpp::Named("betaHat") = beta_values.row(index-1),
                            Rcpp::Named("betaHis") = beta_values.rows(0,index-1),
                            Rcpp::Named("iterations") = K.head(index));
}
