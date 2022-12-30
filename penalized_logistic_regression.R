Evaluate <- function(true_label, pred_label) {
  #  Compute the 0-1 loss between two vectors
  # 
  #  @param true_label: A vector of true labels with length n
  #  @param pred_label: A vector of predicted labels with length n
  #  @return: fraction of points get misclassified
  
  error <- 0
  
  error = sum(true_label!=pred_label)/length(pred_label)
  
  return(error)
}



Predict_logis <- function(data_feature, beta, beta0, type) {
  # Predict by the logistic classifier.
  # 
  # Note: n is the number of examples
  #       p is the number of features per example
  # 
  # @param data_feature: A matrix with dimension n x p, where each row corresponds to
  #   one data point.
  # @param beta: A vector of coefficients with length equal to p.
  # @param beta0: the intercept.
  # @param type: a string value within {"logit", "prob", "class"}.
  # @return: A vector with length equal to n, consisting of 
  #   predicted logits,         if type = "logit";
  #   predicted probabilities,  if type = "prob";
  #   predicted labels,         if type = "class". 
  
  n <- nrow(data_feature)
  pred_vec <- rep(0, n)
  

  # TODO: check if data_feature and *betas match orientation and check if 
  #       colSums works as intended
  
  # finding values
  logit = beta0 + colSums(t(data_feature)*beta)
  
  prob = exp(logit) / (1+exp(logit))
  
  label = rep(0,n)
  label[prob>=0.5] = 1
  
  
  # reporting values
  if (type == 'logit'){pred_vec = logit}
  
  else if (type == 'prob'){pred_vec = prob}
  
  else {pred_vec = label}
  
  return(pred_vec)
}



Comp_gradient <- function(data_feature, data_label, beta, beta0, lbd) {
  # Compute and return the gradient of the penalized logistic regression 
  # 
  # Note: n is the number of examples
  #       p is the number of features per example
  # 
  # @param data_feature: A matrix with dimension n x p, where each row corresponds to
  #   one data point.
  # @param data_label: A vector of labels with length equal to n.
  # @param beta: A vector of coefficients with length equal to p.
  # @param beta0: the intercept.
  # @param lbd: the regularization parameter
  # 
  # @return: a (p+1) x 1 vector of gradients, the first coordinate is the gradient
  #   w.r.t. the intercept.
  
  n <- nrow(data_feature)
  p <- ncol(data_feature)
  grad <- rep(0, 1 + p)

  
  # find the logit
  logit = beta0 + (data_feature)%*%beta
  
  # find the prob 
  prob = exp(logit) / (1+exp(logit))
  
  # find the gradients
  gradm = (1/n) * (t(-data_label + prob)%*%data_feature) + (lbd*sum(beta))
  grad0 = (1/n) * sum(-data_label + prob)
  
  # compile gradients
  grad = c(grad0, gradm)
  

  return(grad)
}



Comp_loss <- function(data_feature, data_label, beta, beta0, lbd) {
  # Compute and return the loss of the penalized logistic regression 
  # 
  # Note: n is the number of examples
  #       p is the number of features per example
  # 
  # @param data_feature: A matrix with dimension n x p, where each row corresponds to
  #   one data point.
  # @param data_label: A vector of labels with with length equal to n.
  # @param beta: A vector of coefficients with length equal to p.
  # @param beta0: the intercept.
  # @param lbd: the regularization parameter
  # 
  # @return: a value of the loss function
  

  # TODO: check if formula is valid
  
  loss <- 0
  
  n = nrow(data_feature)
  
  # find the logit
  logit = beta0 + (data_feature)%*%beta
  
  # find the prob 
  prob = exp(logit) / (1+exp(logit))
  
  # find loss
  loss = ((-1/n) * sum((data_label * log(prob)) + ((1 - data_label)*log(1-prob)))) + ((lbd/2)*sum(beta^2))
  

  return(loss)
}





Penalized_Logistic_Reg <- function(x_train, y_train, lbd, stepsize, max_iter) {
  # This is the main function to fit the Penalized Logistic Regression
  #
  # Note: n is the number of examples
  #       p is the number of features per example
  #
  # @param x_train: A matrix with dimension n x p, where each row corresponds to
  #   one training point.
  # @param y_train: A vector of labels with length equal to n.
  # @param lbd: the regularization parameter.
  # @param stepsize: the learning rate.
  # @param max_iter: a positive integer specifying the maximal number of 
  #   iterations.
  # 
  # @return: a list containing four components:
  #   loss: a vector of loss values at each iteration
  #   error: a vector of 0-1 errors at each iteration
  #   beta: the estimated p coefficient vectors
  #   beta0: the estimated intercept.
  
  p <- ncol(x_train)
  
  # Initialize parameters to 0
  beta_cur <- rep(0, p)
  beta0_cur <- 0
  
  # Create the vectors for recording values of loss and 0-1 error during 
  # the training procedure
  loss_vec <- rep(0, max_iter)
  error_vec <- rep(0, max_iter)
  
  
  for (i in 1:max_iter){
   
    grad = Comp_gradient(x_train, y_train, beta_cur, beta0_cur, lbd)
    
    beta0_cur = beta0_cur - stepsize*(grad[1])
    beta_cur = beta_cur - stepsize*(grad[-1])
    
    
    loss_vec[i] = Comp_loss(x_train,y_train,beta_cur,beta0_cur,lbd)
    
    pred = Predict_logis(x_train, beta_cur, beta0_cur, 'label')
    
    error_vec[i] = Evaluate(y_train,pred)
    
  }
  
  
  return(list("loss" = loss_vec, "error" = error_vec,
              "beta" = beta_cur, "beta0" = beta0_cur))
}
