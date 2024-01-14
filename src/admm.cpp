#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;
using namespace Rcpp;

//' ADMM class.
//' @name ADMM
//' @description
//' The attributes include regression data(dataX, dataY, lambda, rho), iteration parameters(max_iteration, eps_abs, eps_rel), variables to be solved(x, z, u), etc;
//' The function includes initialization of the class(ADMM()), determination of convergence(ifConvergent()), updates of each solving variable(UpdateX(), UpdateZ(), UpdateU()), and complete solution of the function(getResult()).
class ADMM{
private:
  // 数据
  const mat dataX;
  const vec dataY;
  const double lambda; // l2正则惩罚项系数
  const double rho; // 增广拉格朗日参数
  
  vec x, z, u, z_old;// 待迭代求解的变量,z_old记录上一次的z以便收敛性计算
  int dim; // 变量维度--由传入的数据决定
  
  mat x_his;// 存储历史信息
  
  // 一些更新中频繁用到的矩阵
  vec XtY; //X'Y
  mat Tmp; //X'X+rho*I
  
  const int max_iteration; // 最大迭代次数
  const double eps_abs, eps_rel; // 收敛阈值（绝对与相对）
  
public:
  // 初始化
  ADMM(mat dataX_, vec dataY_, double lambda_, double rho_, int max_iter, double eps_abs_, double eps_rel_):
  dataX(dataX_), dataY(dataY_),lambda(lambda_), rho(rho_), max_iteration(max_iter), eps_abs(eps_abs_), eps_rel(eps_rel_)
  {
    if(dataX.n_rows!=dataY.n_rows){
      cout<<"Error! The dimensions are not equal."<<endl;
    }
    
    dim = dataX.n_cols;
    x = zeros<vec>(dim);
    z = zeros<vec>(dim);
    u = zeros<vec>(dim);
    z_old = zeros<vec>(dim);
    x_his = zeros<mat>(6,dim);
      
    XtY = dataX.t()*dataY;
    Tmp = inv(dataX.t()*dataX+rho*eye(dim,dim));
  }
  
  // 迭代求解
  void UpdateX(){
    x = Tmp*(rho*(z-u)+XtY);
  }
  void UpdateZ(){
    z_old = z;
    z = rho/(2*lambda+rho)*(x+u);
  }
  void UpdateU(){
    u = u+(x-z);
  }
  
  // 判断是否收敛（终止条件）
  int ifConvergent(){
    // 计算Primal feasibility的残差和收敛阈值
    double res_pri = norm(x-z);
    double eps_pri = sqrt(dim)*eps_abs+eps_rel*std::max(norm(x),norm(z));
    
    // 计算dual feasibility的残差和收敛阈值
    double res_dual = rho*norm(z-z_old);
    double eps_dual = sqrt(dim)*eps_abs+eps_rel*rho*norm(u);
    
    // 判断是否收敛
    if((res_pri<eps_pri)&&(res_dual<eps_dual)){
      return 1;
    }else{
      return 0;
    }
  }
  
  // 完整求解
  int getResult(){
    int i;
    //cout<<"init:"<<"x:"<<x<<"  z:"<<z<<"  u:"<<u<<endl;
    x_his.row(0)=x.t();
    for(i=0;i<max_iteration;i++){
      UpdateX();
      UpdateZ();
      UpdateU();
      if(i<=4){
        x_his.row(i+1)=x.t();
      }
      //cout<<"iter "<<i+1<<":"<<"x:"<<x<<"  z:"<<z<<"  u:"<<u<<endl;
      if(ifConvergent()){
        break;
      }
    }
    return i;
  }
  
  vec getX(){
    return x;
  }
  
  mat getXHis(){
    return x_his;
  }
};


//' Solve ridge regression using ADMM.
//' 
//' @param X the independent variable X (size n×p)
//' @param y the response variable y (size n×1)
//' @param lambda the penalty factor (size 1×1)
//' @param rho the lagrange multiplier (size 1×1)
//' @param max_iter the maximum number of iterations
//' @param eps_abs the absolute convergence threshold
//' @param eps_rel the relative convergence threshold
//' @return Estimated value of beta(last and historical), iterations and loss
// [[Rcpp::export]]
Rcpp::List RidgeRegression_ADMM(arma::mat X,arma::vec y,double lambda,double rho=0.1,int max_iter=1000, double eps_abs=1e-5, double eps_rel=1e-5){
  ADMM admm_solver(X,y,lambda,rho,max_iter,eps_abs,eps_rel);
  int iter_num = admm_solver.getResult(); // 迭代求解，并返回迭代次数
  vec betaHat = admm_solver.getX();
  double loss = 0.5*pow(norm(y-X*betaHat),2)+lambda*pow(norm(betaHat),2);

  return Rcpp::List::create(Rcpp::Named("betaHat") = betaHat,
                            Rcpp::Named("betaHis") = admm_solver.getXHis(),
                            Rcpp::Named("loss") = loss,
                            Rcpp::Named("iterations") = iter_num);
}
