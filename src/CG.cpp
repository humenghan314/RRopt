#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;
using namespace Rcpp;


//' CG class.
//' @name CG
//' @description
//' The function includes initialization of the class(CG()), calculate the loss of beta_k,two ways to determine the step(alpha),the solving function of minimize loss by iteration,and complete solution of the function(getResult())
class CG{
private:
  const mat X;
  const vec Y;
  const double lambda;// l2正则惩罚项系数
  const double t;//共轭条件中的参数
  const double eps; //收敛阈值
  
  int dim;
  const int max_iteration;//最大迭代次数
  mat beta; //存储每次迭代的结果
  mat g;
  mat d;
  int k;//迭代次数
  vec beta_k;
  vec g_k;
  vec d_k;
  double alpha_k;
public:
//初始化
  CG(mat X_,vec Y_,double lambda_,double t_,double eps_,int max_iteration_):
  X(X_),Y(Y_),lambda(lambda_),t(t_),eps(eps_),max_iteration(max_iteration_){
    if(X.n_rows!=Y.n_rows){
      cout<<"Error! The dimensions are not equal."<<endl;
      exit(0);}
    dim = X.n_cols;
    beta_k = zeros<vec>(dim);
    g_k = -X.t()*(Y-X*beta_k)+2*lambda*beta_k;
    d_k = -g_k;
    alpha_k = 1.0;
    k = 0;
    beta = zeros<mat>(max_iteration+1,dim);
    g= zeros<mat>(max_iteration+1,dim);
    d= zeros<mat>(max_iteration+1,dim);
  }
  

  //损失函数函数值求解
  double loss(vec beta_1){
    double loss_;
    vec res = Y-X*beta_1;
    loss_ = dot(res,res)+lambda*dot(beta_1,beta_1);
    return loss_;
  }
  
  //利用strong-wlofe准则求步长
  double step_wlofe(){
    double c1 = 0.01,c2=0.1,alpha = 1; //0<c1<c2<1
    int i = 0;
    double a=0, b=0x7FF0000000000000;
    vec beta_next = beta_k+alpha*d_k;
    double f1 = loss(beta_k),f2=loss(beta_next);
    vec g1 = g_k;
    vec g2 = -X.t()*(Y-X*beta_next)+2*lambda*beta_next;
    double l = f1 + c1*alpha*dot(g1,d_k);
    while (true){
      if (i>50){
        break;
      }
      if (f2>l){ //充分下降条件
        i = i + 1;
        b = alpha;
        alpha = (a+b)/2;
        beta_next = beta_k+alpha*d_k;
        f2 = loss(beta_next);
        g2 = -X.t()*(Y-X*beta_next)+2*lambda*beta_next;
        l = f1 + c1*alpha*dot(g1,d_k);
        continue;
      }
      if (dot(g2,d_k) < abs(c2*dot(g1,d_k))){ //曲率条件
        i = i + 1;
        a = alpha;
        alpha = (2*alpha > (a+b)/2) ? (a+b)/2 : 2*alpha;
        beta_next = beta_k+alpha*d_k;
        f2 = loss(beta_next);
        g2 = -X.t()*(Y-X*beta_next)+2*lambda*beta_next;
        l = f1 + c1*alpha*dot(g1,d_k);
        continue;
      }
      break;
    }
    if(alpha<1e-3) return 1e-3;
    else return alpha;

  }
  
  //使用Armijo准则求步长
  double step_armijo(){
    double be = 0.5,sigma=0.1;
    int m=0,mmax=20;
    int mk;
    vec beta_k_next;
    while(m<mmax){
      beta_k_next = beta_k + pow(be,m)*d_k;
      if (loss(beta_k_next) <= loss(beta_k) + sigma*pow(be,m)*dot(g_k,d_k)){
        mk=m;
        break;
      }
      m = m+1;
    }
    if(pow(be,mk)<1e-3){
      return 1e-3;
    }
    else
      return pow(be,mk);
  }
  
  //迭代求解
  void solve(){
    vec beta_k_next;
    vec g_k_next;
    double co_factor;
    beta.row(0) = beta_k.t();
    g.row(0) = g_k.t();
    d.row(0) = d_k.t();
    for (k=0;k<max_iteration;k++){
      if(sqrt(dot(g_k,g_k)) < eps){
        break;
        }
      alpha_k = step_wlofe();
      beta_k_next = beta_k + alpha_k * d_k;
      g_k_next = -X.t()*(Y-X*beta_k_next)+2*lambda*beta_k_next;
      double s;
      s = ((dot(g_k_next,g_k_next-g_k))/(dot(d_k,g_k_next-g_k)) > 0) ? (dot(g_k_next,g_k_next-g_k))/(dot(d_k,g_k_next-g_k)) : 0;
      co_factor = s-t*alpha_k*dot(g_k_next,d_k)/(dot(d_k,g_k_next-g_k));//计算共轭条件下的组合系数
      //co_factor = (dot(g_k_next,g_k_next-g_k))/(dot(d_k,g_k_next-g_k));
      d_k = -g_k_next + co_factor * d_k;
      beta_k = beta_k_next;
      g_k = g_k_next;
      beta.row(k+1) = beta_k.t();
      g.row(k+1) = g_k.t();
      d.row(k+1) = d_k.t();
    }
 
  }
  
  vec get_beta_k(){return beta_k;}
  mat get_beta(){return beta;}
  int get_iteration_numbers(){return k;}
  double loss_min(){return loss(beta_k);}

};


//'Compute the optimal estimates of beta by minimize the loss function with l2 penalty through CG method
//'
//'@param X the independent variable matrix
//'@param y the dependent variable vector
//'@param lambda the coeffient of l2 regularization
//'@t the coeffient of the Dai-Liao's conjugacy conditions when solving the co_factor of search direction
//'@eps the threshold of gradient to judge whether converging
//'@max_iteration the maximum number of iterations
//'@return the list contains the estimated value of beta,the matrix of the beta under every iteration,the iterations number,the final loss
//'@examples
//'seed_num = 123
//'n = 100
//'p = 2d
//'simudata = getData(seed_num,n,p)
//'result = RidgeRegression_CG(simudata$X,simudata$y,lambda = 0.1,max_iteration = 1000)
//'result$betaHat
// [[Rcpp::export]]
Rcpp::List RidgeRegression_CG(arma::mat X,arma::vec y,double lambda,double t=0.1,double eps=1e-5,int max_iteration=1000){
  CG CG_solver(X,y,lambda,t,eps,max_iteration);
  CG_solver.solve();
  
  return Rcpp::List::create(Rcpp::Named("betaHat") = CG_solver.get_beta_k(),
                            Rcpp::Named("betaHis") = CG_solver.get_beta(),
                            Rcpp::Named("iterations") = CG_solver.get_iteration_numbers(),
                            Rcpp::Named("loss") = CG_solver.loss_min());
}

