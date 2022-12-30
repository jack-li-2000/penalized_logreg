rm(list = ls())

## You should set the working directory to the folder of hw3_starter by
## uncommenting the following and replacing YourDirectory by what you have
## in your local computer / labtop

setwd("D:/github_repo/penalized_logreg")

## Load utils.R and penalized_logistic_regression.R

source("utils.R")
source("penalized_logistic_regression.R")

# =================================================================

beta = c(1,1.5,2,2.5,3)
data_feature = matrix(rnorm(60), ncol=5)
data_label = c(1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1)
beta0 = 1.5

Predict_logis(data_feature, beta, beta0, 'label')

# =================================================================


prob = exp(beta0 + colSums(t(data_feature)*beta)) / (1+exp(beta0 + colSums(t(data_feature)*beta)))
prob

# n x p matrix has n rows and p columns 
logit = beta0 + colSums(t(data_feature)*beta)
logit

label = rep(0,nrow(data_feature))
label[prob>=0.5] = 1

label


# =================================================================
type='logit'

pred_vec = rep(0,nrow(data_feature))

if (type == 'logit'){
  pred_vec = logit
} else if (type == 'prob'){
  pred_vec = prob
}else{pred_vec <- label}
pred_vec


# =================================================================

# find for each beta which means calculations for each feature column which is transposed to be row

n <- nrow(x_train)
p <- ncol(x_train)

beta0 = 0
lbd = 0.1
beta = rep(0.1,p)
grad <- rep(0, 1 + p)



logit = beta0 + (x_train)%*%beta
prob = exp(logit) / (1+exp(logit))



gradm = (1/n) * (t(-y_train + prob)%*%x_train) + (lbd*sum(beta))

grad0 = (1/n) * (-y_train + prob)

grad = c(grad0, gradm)


grad


lbd*beta
grad
prob
dim((x_train)*beta)

sum(t(-y_train + prob)%*%x_train)/n

(x_train)%*%beta

# test = 1xp matrix with 256 columns 
# =================================================================


Comp_loss(data_feature, data_label, beta, beta0, 0.5)

c(1,2,3,4,5,6)[-1]

# =================================================================

rm(list = ls())

## You should set the working directory to the folder of hw3_starter by
## uncommenting the following and replacing YourDirectory by what you have
## in your local computer / labtop

setwd("D:/OneDrive - University of Toronto/university/4th year/sta314/hw3/hw3_starter/hw3_starter")

## Load utils.R and penalized_logistic_regression.R

source("utils.R")
source("penalized_logistic_regression.R")
train <- Load_data("./data/train.csv")
valid <- Load_data("./data/valid.csv")
test <- Load_data("./data/test.csv")

x_train <- train$x
y_train <- train$y

x_valid <- valid$x
y_valid <- valid$y

x_test <- test$x
y_test <- test$y

y_train

Penalized_Logistic_Reg(x_train, y_train, 0, 1/3, 10)

max_iter = 10
p = ncol(x_train)
beta_cur <- rep(0, p)
beta0_cur <- 0
lbd = 0
stepsize = 1/30

for (i in 1:max_iter){
  
  grad = Comp_gradient(x_train, y_train, beta_cur, beta0_cur, lbd)
  
  beta0_cur = beta0_cur - stepsize*(grad[1])
  beta_cur = beta_cur - stepsize*(grad[-1])
  
  # loss_vec[i] = Comp_loss(x_train,y_train,beta_cur,beta0_cur,lbd)
  # 
  # pred = Predict_logis(x_train, beta_cur, beta0_cur, 'label')
  # 
  # error_vec[i] = Evaluate(y_train,pred)
}

