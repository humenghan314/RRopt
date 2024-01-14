#ifndef MY_LBFGS_HPP
#define MY_LBFGS_HPP

#include <cmath>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;
using namespace Rcpp;

namespace my_lbfgs{

// 优化超参数
struct lbfgs_parameter_t{
  int mem_size = 8; // 限制内存大小
  double g_epsilon = 1.0e-5; // 梯度收敛阈值
  int past = 3; // 收敛测试的距离
  double delta = 1.0e-6; // 收敛测试的超参数
  int max_iterations = 0; // 最大迭代次数
  int max_linesearch = 64; // 线搜索最大次数
  double min_step = 1.0e-20; // 线搜索最小步长
  double max_step = 1.0e+20; // 线搜索最大步长
  double f_dec_coeff = 1.0e-4;  // Armijo阈值
  double s_curv_coeff = 0.9; // weak wolfe阈值
  double cautious_factor = 1.0e-6; // 非凸优化全局收敛
  double machine_prec = 1.0e-16; // 机器精度
};

// 返回值
enum
{
  LBFGS_CONVERGENCE = 0, // 达到收敛
  LBFGS_STOP, // 满足停止要求
  LBFGS_CANCELED, // 迭代被终止
  LBFGSERR_UNKNOWNERROR = -1024,  // 未知错误
  LBFGSERR_INVALID_N, // 参数无效
  LBFGSERR_INVALID_MEMSIZE, // men_size设置无效
  LBFGSERR_INVALID_GEPSILON, // g_epsilon设置无效
  LBFGSERR_INVALID_TESTPERIOD, // past设置无效
  LBFGSERR_INVALID_DELTA, // delta设置无效
  LBFGSERR_INVALID_MINSTEP, // min_step设置无效
  LBFGSERR_INVALID_MAXSTEP, // max_step设置无效
  LBFGSERR_INVALID_FDECCOEFF, // f_dec_coeff设置无效
  LBFGSERR_INVALID_SCURVCOEFF, // s_curv_coeff设置无效
  LBFGSERR_INVALID_MACHINEPREC, // machine_prec设置无效
  LBFGSERR_INVALID_MAXLINESEARCH, // max_linesearch设置无效
  LBFGSERR_INVALID_FUNCVAL, // 函数值错误
  LBFGSERR_MINIMUMSTEP, // 搜索步长小于设置的最小步长
  LBFGSERR_MAXIMUMSTEP, // 搜索步长大于设置的最大步长
  LBFGSERR_MAXIMUMLINESEARCH, // 线性搜索次数达到最大值但没找到最优点
  LBFGSERR_MAXIMUMITERATION, // 算法达到最大迭代次数
  LBFGSERR_WIDTHTOOSMALL, // 相对搜索间隔小于machine_prec
  LBFGSERR_INVALIDPARAMETERS, // 步长为负
  LBFGSERR_INCREASEGRADIENT, // 搜索方向是使目标函数增加的方向
};

/**
 * 回调接口：定义目标函数和梯度向量
 *  @param  instance    数据
 *  @param  x           当前迭代点
 *  @param  g           梯度向量
 *  @retval double      当前目标函数值
 */
typedef double (*lbfgs_evaluate_t)(arma::mat dataX, arma::vec datay, void *instance, const arma::vec &x, arma::vec &g);

/**
 * 回调接口：提供线搜索的上限值
 *  @param  instance    数据
 *  @param  xp          线搜索前的迭代点
 *  @param  d           步长向量
 *  @retval double      当前线搜索的最大步长
 */
typedef double (*lbfgs_stepbound_t)(void *instance, const arma::vec &xp, const arma::vec &d);

/**
 * 回调接口：返回优化过程值
 *  @param  instance    数据
 *  @param  x           当前迭代点
 *  @param  g           当前变量的梯度值
 *  @param  fx          当前目标函数值
 *  @param  step        当前线性搜索步长
 *  @param  k           迭代次数
 *  @param  ls          本次迭代的计算次数
 *  @retval int         返回0则继续优化，否则停止优化
 */
typedef int (*lbfgs_progress_t)(void *instance, const arma::vec &x, 
             const arma::vec &g, const double fx, const double step, const int k, const int ls);

// 回调数据结构
struct callback_data_t {
  void *instance = nullptr;
  lbfgs_evaluate_t proc_evaluate = nullptr;
  lbfgs_stepbound_t proc_stepbound = nullptr;
  lbfgs_progress_t proc_progress = nullptr;
};


/**
 * 线搜索：同时满足armijo条件和弱Wolfe条件
 *  @param  x           初始迭代点
 *  @param  f           当前目标函数值
 *  @param  g           初始梯度值
 *  @param  stp         当前步长
 *  @param  s           搜索方向
 *  @param  xp          当前迭代点
 *  @param  gp          当前梯度值
 *  @param  stpmin      步长最小值
 *  @param  stpmax      步长最大值
 *  @param  cd          回调接口
 *  @param  param       参数
 *  @retval int         步长迭代次数
 */
inline int line_search_lewisoverton(arma::mat X, arma::vec y,
                             arma::vec &x,
                             double &f,
                             arma::vec &g,
                             double &stp,
                             const arma::vec &s,
                             const arma::vec &xp,
                             const arma::vec &gp,
                             const double stpmin,
                             const double stpmax,
                             const callback_data_t &cd,
                             const lbfgs_parameter_t &param){
  arma::mat dataX = X;
  arma::mat datay = y;
  int count = 0;
  bool brackt = false, touched = false;
  double finit, dginit, dgtest, dstest;
  double mu = 0.0, nu = stpmax;
  
  if (!(stp > 0.0)) // step>0
  { return LBFGSERR_INVALIDPARAMETERS;}
  
  dginit = arma::dot(gp, s); // 线搜索的初始梯度
  
  if (dginit > 0.0){ // 确保下降方向
    return LBFGSERR_INCREASEGRADIENT;}

  finit = f; // 目标函数初始化
  dgtest = param.f_dec_coeff * dginit; 
  dstest = param.s_curv_coeff * dginit;
  
  while (true){
    x = xp + stp * s; // 更新x
    
    f = cd.proc_evaluate(dataX, datay, cd.instance, x, g);
    ++count; // 步长更新 迭代的次数
    
    if (std::isinf(f) || std::isnan(f)){
      return LBFGSERR_INVALID_FUNCVAL;}
    
    // 检查Armijo条件是否满足
    if (f > finit + stp * dgtest){
      nu = stp;
      brackt = true;}
    else{
      // 弱 Wolfe 条件是否满足
      if (arma::dot(g, s) < dstest)
      { mu = stp;}
      else{
        return count;
      }
      }
    if (param.max_linesearch <= count){
      return LBFGSERR_MAXIMUMLINESEARCH;} // 达到最大步长迭代次数
    if (brackt && (nu - mu) < param.machine_prec * nu){
      return LBFGSERR_WIDTHTOOSMALL;} // 搜索间隔小于机器精度
    
    if (brackt){ // 不满足armijo条件则继续迭代
      stp = 0.5 * (mu + nu);}
    else{
      stp *= 2.0;}
    
    if (stp < stpmin){ // 步长越界
      return LBFGSERR_MINIMUMSTEP;}
    if (stp > stpmax){
      if (touched){
        return LBFGSERR_MAXIMUMSTEP;} 
      else{
        touched = true;
        stp = stpmax; // 达到最大值则取最大值
      }
    }
  }
}


/**
 * L-BFGS优化
 *
 *  @param  x               优化变量x，设置初始值x0
 *  @param  f               目标函数值
 *  @param  proc_evaluate   回调接口：包含目标函数f(x)和梯度g
 *  @param  proc_stepbound  回调接口：线搜索的上限值
 *  @param  proc_progress   回调接口：返回过程值
 *  @param  instance        数据
 *  @param  param           优化超参数
 *  @retval int             如果返回非负值则没有发生错误，否则发生错误
 */
inline int lbfgs_optimize(arma::mat X,arma::vec y, arma::vec &x,
                   double &f,
                   lbfgs_evaluate_t proc_evaluate,
                   lbfgs_stepbound_t proc_stepbound,
                   lbfgs_progress_t proc_progress,
                   void *instance,
                   const lbfgs_parameter_t &param)
{
  arma::mat dataX=X;
  arma::vec datay=y;
  int ret, i, j, k, ls, end, bound;
  double step, step_min, step_max, fx, ys, yy;
  double gnorm_inf, xnorm_inf, beta, rate, cau;
  
  const int n = x.n_elem;
  const int m = param.mem_size;
  
  // 检查超参数设置错误
  if (n <= 0)
  { return LBFGSERR_INVALID_N;}
  if (m <= 0)
  { return LBFGSERR_INVALID_MEMSIZE;} 
  if (param.g_epsilon < 0.0)
  { return LBFGSERR_INVALID_GEPSILON;}
  if (param.past < 0)
  { return LBFGSERR_INVALID_TESTPERIOD;}
  if (param.delta < 0.0)
  { return LBFGSERR_INVALID_DELTA;}
  if (param.min_step < 0.0)
  { return LBFGSERR_INVALID_MINSTEP;}
  if (param.max_step < param.min_step)
  { return LBFGSERR_INVALID_MAXSTEP;}
  if (!(param.f_dec_coeff > 0.0 && param.f_dec_coeff < 1.0))
  { return LBFGSERR_INVALID_FDECCOEFF;}
  if (!(param.s_curv_coeff < 1.0 && param.s_curv_coeff > param.f_dec_coeff))
  { return LBFGSERR_INVALID_SCURVCOEFF;}
  if (!(param.machine_prec > 0.0))
  { return LBFGSERR_INVALID_MACHINEPREC;}
  if (param.max_linesearch <= 0)
  { return LBFGSERR_INVALID_MAXLINESEARCH;}
  
  // 初始化过程值
  arma::vec xp(n);
  arma::vec g(n);
  arma::vec gp(n);
  arma::vec d(n);
  arma::vec pf(std::max(1, param.past));
  
  // 初始化限制内存的空间
  arma::vec lm_alpha = arma::zeros<arma::vec>(m);
  arma::mat lm_s = arma::zeros<arma::mat>(n, m);
  arma::mat lm_y = arma::zeros<arma::mat>(n, m);
  arma::vec lm_ys = arma::zeros<arma::vec>(m);
  
  //  创建一个回调接口
  callback_data_t cd;
  cd.instance = instance;
  cd.proc_evaluate = proc_evaluate;
  cd.proc_stepbound = proc_stepbound;
  cd.proc_progress = proc_progress;
  
  // 计算初始目标函数值
  fx = cd.proc_evaluate(dataX, datay, cd.instance, x, g);
  pf(0) = fx; // 存储fx
  
  // 计算下降方向 H_0初始化为单位矩阵I
  d = -g;
  
  // 保证初始向量不是最优点
  // arma::vec g_abs = arma::abs(g);
  gnorm_inf = arma::max(arma::abs(g));
  // arma::vec x_abs = arma::abs(x);
  xnorm_inf = arma::max(arma::abs(x));
  
  if (gnorm_inf / std::max(1.0, xnorm_inf) <= param.g_epsilon)
  { // 初始迭代点为最优点
    ret = LBFGS_CONVERGENCE;}
  else
  { step = 1.0 / arma::norm(d);// 初始步长
    k = 1;
    end = 0;
    bound = 0;
    
    // 进入迭代
    while (true) 
    { // 保存当前迭代点和梯度
      xp = x;
      gp = g;
      
      step_min = param.min_step;
      step_max = param.max_step;
      if (cd.proc_stepbound){
        step_max = cd.proc_stepbound(cd.instance, xp, d);
        step_max = step_max < param.max_step ? step_max : param.max_step; // 设置最大步长
      }
      step = step < step_max ? step : 0.5 * step_max;
      
      // 寻找最优步长
      ls = line_search_lewisoverton(dataX, datay, x, fx, g, step, d, xp, gp, step_min, step_max, cd, param);
      
      if (ls < 0) // 如果步长搜索返回负值则说明没有最优步长或有其他错误 结束迭代
      { x = xp;
        g = gp;
        ret = ls; // 返回迭代信息
        break;}
      
      // 输出过程值
      if (cd.proc_progress){
        if (cd.proc_progress(cd.instance, x, g, fx, step, k, ls)){
          ret = LBFGS_CANCELED;
          break;}
        }
      
      // 收敛测试
      gnorm_inf = arma::max(arma::abs(g));
      xnorm_inf = arma::max(arma::abs(x));
      if (gnorm_inf / std::max(1.0, xnorm_inf) < param.g_epsilon){
        // 收敛
        ret = LBFGS_CONVERGENCE;
        break;
      }
      
      // 停止准则
      if (param.past>0){
        if (param.past <= k){
          rate = std::fabs(pf(k % param.past) - fx) / std::max(1.0, std::fabs(fx));
          
          if (rate < param.delta){
            ret = LBFGS_STOP;
            break;
          }
        }
        // 在限制空间中存储当前目标函数值
        pf(k % param.past) = fx;
      }
      
      if (param.max_iterations != 0 && param.max_iterations <= k){ // 达到最大迭代次数则终止
        ret = LBFGSERR_MAXIMUMITERATION;
        break;}
      
      // 迭代器
      ++k;
      
      // 更新向量s和y
      // s_{k+1} = x_{k+1} - x_{k} = \step * d_{k}.
      // y_{k+1} = g_{k+1} - g_{k}.
      lm_s.col(end) = x - xp;
      lm_y.col(end) = g - gp;
      

      // 计算ys和yy
      // ys = y^t \cdot s = 1 / \rho.
      // yy = y^t \cdot y.
      ys = arma::dot(lm_y.col(end), lm_s.col(end));
      yy = arma::dot(lm_y.col(end), lm_y.col(end));
      
      lm_ys(end) = ys;
      
      // 计算梯度
      d = -g;
      
      // 步长更新条件
      cau = arma::dot(lm_s.col(end), lm_s.col(end)) * arma::norm(gp) * param.cautious_factor;
      
      if (ys > cau)  // 如果满足步长更新条件则进行two-loop双循环递归来更新搜索方向
      {
        ++bound;
        bound = m < bound ? m : bound;
        end = (end + 1) % m;
        
        j = end;
        for (i = 0; i < bound; ++i){
          j = (j + m - 1) % m;     
          lm_alpha(j) = arma::dot(lm_s.col(j), d) / lm_ys(j);
          d += (-lm_alpha(j)) * lm_y.col(j);
        }
        
        d *= ys / yy;
        
        for (i = 0; i < bound; ++i){
          beta = arma::dot(lm_y.col(j),d) / lm_ys(j);
          d += (lm_alpha(j) - beta) * lm_s.col(j);
          j = (j + 1) % m; 
        }
      }
      
      // 步长设置为1
      step = 1.0;
    }
  }
  
  // 返回目标函数最终值
  f = fx;
  
  return ret;
}

/**
 * 返回由优化器返回的字符串的描述
 *
 *  @param err          lbfgs_optimize()的返回值
 */
inline const char *lbfgs_strerror(const int err)
{
  switch (err)
  {
  case LBFGS_CONVERGENCE:
    return "Success: reached convergence (g_epsilon).";
    
  case LBFGS_STOP:
    return "Success: met stopping criteria (past f decrease less than delta).";
    
  case LBFGS_CANCELED:
    return "The iteration has been canceled by the monitor callback.";
    
  case LBFGSERR_UNKNOWNERROR:
    return "Unknown error.";
    
  case LBFGSERR_INVALID_N:
    return "Invalid number of variables specified.";
    
  case LBFGSERR_INVALID_MEMSIZE:
    return "Invalid parameter lbfgs_parameter_t::mem_size specified.";
    
  case LBFGSERR_INVALID_GEPSILON:
    return "Invalid parameter lbfgs_parameter_t::g_epsilon specified.";
    
  case LBFGSERR_INVALID_TESTPERIOD:
    return "Invalid parameter lbfgs_parameter_t::past specified.";
    
  case LBFGSERR_INVALID_DELTA:
    return "Invalid parameter lbfgs_parameter_t::delta specified.";
    
  case LBFGSERR_INVALID_MINSTEP:
    return "Invalid parameter lbfgs_parameter_t::min_step specified.";
    
  case LBFGSERR_INVALID_MAXSTEP:
    return "Invalid parameter lbfgs_parameter_t::max_step specified.";
    
  case LBFGSERR_INVALID_FDECCOEFF:
    return "Invalid parameter lbfgs_parameter_t::f_dec_coeff specified.";
    
  case LBFGSERR_INVALID_SCURVCOEFF:
    return "Invalid parameter lbfgs_parameter_t::s_curv_coeff specified.";
    
  case LBFGSERR_INVALID_MACHINEPREC:
    return "Invalid parameter lbfgs_parameter_t::machine_prec specified.";
    
  case LBFGSERR_INVALID_MAXLINESEARCH:
    return "Invalid parameter lbfgs_parameter_t::max_linesearch specified.";
    
  case LBFGSERR_INVALID_FUNCVAL:
    return "The function value became NaN or Inf.";
    
  case LBFGSERR_MINIMUMSTEP:
    return "The line-search step became smaller than lbfgs_parameter_t::min_step.";
    
  case LBFGSERR_MAXIMUMSTEP:
    return "The line-search step became larger than lbfgs_parameter_t::max_step.";
    
  case LBFGSERR_MAXIMUMLINESEARCH:
    return "Line search reaches the maximum try number, assumptions not satisfied or precision not achievable.";
    
  case LBFGSERR_MAXIMUMITERATION:
    return "The algorithm routine reaches the maximum number of iterations.";
    
  case LBFGSERR_WIDTHTOOSMALL:
    return "Relative search interval width is at least lbfgs_parameter_t::machine_prec.";
    
  case LBFGSERR_INVALIDPARAMETERS:
    return "A logic error (negative line-search step) occurred.";
    
  case LBFGSERR_INCREASEGRADIENT:
    return "The current search direction increases the cost function value.";
    
  default:
    return "(unknown)";
  }
}

}

#endif
