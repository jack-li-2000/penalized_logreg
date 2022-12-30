rm(list = ls())

setwd("D:/github_repo/penalized_logreg")

## Load utils.R and penalized_logistic_regression.R

source("utils.R")
source("penalized_logistic_regression.R")



## load data sets

train <- Load_data("./data/train.csv")
valid <- Load_data("./data/valid.csv")
test <- Load_data("./data/test.csv")

x_train <- train$x
y_train <- train$y

x_valid <- valid$x
y_valid <- valid$y

x_test <- test$x
y_test <- test$y


## visualize the first five and 301th-305th digits in the training data. 
Plot_digits(c(1:5, 301:305), x_train)




# checking hyperparameters

lbd = 0
stepsize = 1/5
max_iter = 400

result = Penalized_Logistic_Reg(x_train, y_train, lbd, stepsize, max_iter)

result$loss
result$error

plot(result$loss)
plot(result$error)


# =============================================================================

# =============================================================================

stepsize <- 1/5
max_iter <- 400  

lbd_grid <- c(0, 0.01, 0.05, 0.1, 0.5, 1)

result = Penalized_Logistic_Reg(x_train, y_train, lbd_grid[1], stepsize, max_iter)
plot(result$error)

result2 = Penalized_Logistic_Reg(x_train, y_train, lbd_grid[2], stepsize, max_iter)
plot(result2$error)

result3 = Penalized_Logistic_Reg(x_train, y_train, lbd_grid[3], stepsize, max_iter)
plot(result3$error)

result4 = Penalized_Logistic_Reg(x_train, y_train, lbd_grid[4], stepsize, max_iter)
plot(result4$error)

result5 = Penalized_Logistic_Reg(x_train, y_train, lbd_grid[5], stepsize, max_iter)
plot(result5$error)

result6 = Penalized_Logistic_Reg(x_train, y_train, lbd_grid[6], stepsize, max_iter)
plot(result6$error)




# =============================================================================


# =============================================================================


# fit based on best hyperparameters


stepsize <- 1/5
max_iter <- 400
lbd <- 0.01

# can also use cross validation to fit hyperparameters




