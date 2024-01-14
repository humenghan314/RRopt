#' Get simulation data
#' 
#' @param seed_num the random seed
#' @param n the sample size
#' @param p the feature size
#' @return the generated data and beta as a list
getData <- function(seed_num,n,p){
    set.seed(seed_num)
    X <- matrix(runif(n*p,-2,2),n,p)
    b <- runif(p)
    y <- X %*% b + rnorm(n,0,1)
    return(list(X = X, y = as.numeric(y), beta = b))
}

loss <- function(X,y,lambda,beta0,beta1){
    res = y-X[,1]*beta0-X[,2]*beta1
    return(1/2*sum(res*res)+lambda*(beta0*beta0+beta1*beta1));
}

#' Draw contour maps(when the dimensions of beta is 2).
#' 
#' @param X the independent variable X (size n×p)
#' @param y the response variable y (size n×1)
#' @param lambda the penalty factor (size 1×1)
#' @param beta the estimated value of beta (size p×1)
#' @return the contour map
#' @examples  
#' library(ggplot2)
#' library(ggisoband)
#' data = getData(123,100,2)
#' result = RidgeRegression(data$X,data$y,lambda = 0.1,max_iter = 100)
#' plotSample(data$X,data$y,lambda = 0.1,beta = result$betaHis)
plotSample <- function(X,y,lambda,beta){
    if(dim(beta)[2]!=2){
        print('The dimension must be one.')
        return()
    }
    
    beta_loss = data.frame()
    idx <- seq(0,1,0.01)
    len <- length(idx)
    for(i in 1:len){
        for(j in 1:len){
            row_id = (i-1)*len+j
            beta_loss[row_id,1]=idx[i]
            beta_loss[row_id,2]=idx[j]
            beta_loss[row_id,3]=loss(X,y,lambda,idx[i],idx[j])
        }
    }
    names(beta_loss) <- c('x','y','loss')    
    beta <- as.data.frame(beta)
    names(beta) <- c('beta0','beta1')
    
    p <- ggplot(beta_loss, aes(x, y, z = loss)) +
        geom_isobands(fill = NA) +
        scale_color_viridis_c() +
        xlab("beta0") + ylab("beta1") + 
        coord_cartesian(expand = FALSE) +
        theme_bw() +
        geom_point(data=beta,aes(x=beta$beta0,y=beta$beta1,z=0),size=10,pch=20,color="black") +
        geom_line(data=beta,aes(x=beta$beta0,y=beta$beta1,z=0))
    return(p)
}

