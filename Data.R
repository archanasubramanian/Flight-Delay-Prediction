#### Standardising data ####
library(dplyr)
library(randomForest)
library(nnet)   # For logistic regression
library(car)    # For ANOVA after logistic regression with nnet
library(glmnet) # For logistic regression and LASSO
library(MASS) # For discriminant analysis
library(FNN)

data = read.csv("Datasets/Jan_2020_ontime.csv")

data <- data %>% 
  filter(!is.na(ARR_DEL15)) %>% 
  filter(CANCELLED==0) %>%
  dplyr::select(DAY_OF_MONTH,
         DAY_OF_WEEK,
         OP_CARRIER_AIRLINE_ID,
         OP_CARRIER,
         OP_CARRIER_FL_NUM,
         ORIGIN_AIRPORT_ID,
         DEST_AIRPORT_ID,
         DEP_DEL15,
         ARR_DEL15,
         CANCELLED,
         DIVERTED,
         DISTANCE)

#sub_data = data[sample(nrow(data), 10000), ]

write.table(sub_data, "JAN_DELAY.csv", sep = ",", row.names = T, col.names = T)


sub_data = sub_data[,-22]
sub_data = na.omit(sub_data)

cols_as_factor <- c("DEP_DEL15", "ARR_DEL15", "OP_CARRIER")
sub_data[cols_as_factor] <- lapply(sub_data[cols_as_factor], factor)

#Converting variables to numeric
sub_data$OP_CARRIER = as.numeric(sub_data$OP_CARRIER)
#int = as.integer(sub_data$OP_CARRIER)
#sub_data$OP_CARRIER = int

sub_data$DEP_DEL15 = as.numeric(sub_data$DEP_DEL15)

sapply(sub_data,class)

set.seed (46685326 , kind="Mersenne-Twister")

perm <- sample ( x = nrow ( sub_data ))
train_set <- sub_data [ which ( perm <= 3* nrow ( sub_data )/4) , ]
test_set <- sub_data [ which ( perm > 3* nrow ( sub_data )/4) , ]
Y.train = train_set[,9]
#Y.train = Y.train[,1]
Y.valid = test_set[,9]
#Y.valid = Y.valid[,1]
X.train = train_set[,-9]
X.valid = test_set[,-9]

#### Correlation ####
library(corrr)
cor(sub_data[sapply(sub_data, function(x) is.numeric(x))])

### Defaults
default.tree <- rpart(class ~ ., data = train_set, method="class")
### cp = 0 tree
fit.tree.full = rpart(class ~ ., data = train_set, method = "class", cp = 0)

#### KNN ####
pred.knn = knn(X.train, X.valid, Y.train, k=8)
table(pred.knn, Y.valid, dnn = c("Predicted", "Observed"))
# Misclassification rate
(misclass.knn = mean(pred.knn != Y.valid))

#### RF ####
rf.default<- randomForest(data=train_set, ARR_DEL15~., importance=TRUE, keep.forest=TRUE)

round(importance(rf.default),3) 
x11(h=7,w=15)
varImpPlot(rf.default)

pred.rf.train <- predict(rf.default, newdata=train_set, type="response")
(misclass.train.rf <- mean(Y.train != pred.rf.train))

pred.rf.test <- predict(rf.default, newdata=test_set, type="response")
(misclass.test.rf <- mean(Y.valid != pred.rf.test))

#### Logistic regression ####
fit.log.nnet = multinom(ARR_DEL15 ~ ., data = train_set, maxit = 50)
summary(fit.log.nnet)
# Anova
Anova(fit.log.nnet)

# Fit on test set
pred.log.nnet.train = predict(fit.log.nnet, X.train)
pred.log.nnet.test = predict(fit.log.nnet, X.valid)

# Misclassification errors rate
(misclass.log.nnet.train = mean(pred.log.nnet.train != Y.train))
(misclass.log.nnet.test = mean(pred.log.nnet.test != Y.valid))

# Confusion matrix
table(Y.valid, pred.log.nnet.test,dnn = c("Observed", "Predicted"))

#### Rescaling ####
scale.1 <- function(x1,x2){
  for(col in 1:ncol(x1)){
    a <- mean(x2[,col])
    b <- sd(x2[,col])
    x1[,col] <- (x1[,col]-a)/b
  }
  x1
}

# Including the factor variable in data
data.train.scale = train_set
data.valid.scale = test_set
data.train.scale[,-9] = scale.1(data.train.scale[,-(9:11)], data.train.scale[,-(9:11)])
data.train.scale = cbind(data.train.scale,train_set[,(9:11)] )
data.valid.scale[,-9] = scale.1(data.valid.scale[,-(9:11)], data.train.scale[,-(9:11)])
data.valid.scale = cbind(data.valid.scale,test_set[,(9:11)] )


### LASSO
# Data
X.train.scale = as.matrix(data.train.scale[,-9])
#Y.train = data.train.scale[,9]
X.valid.scale = as.matrix(data.valid.scale[,-9])
#Y.valid = data.valid.scale[,9]
set.seed(46685326, kind = "Mersenne-Twister")

# Fit the model
fit.CV.lasso = cv.glmnet(X.train.scale, Y.train, family = "multinomial")

# Lambda Min
lambda.min = fit.CV.lasso$lambda.min
coef(fit.CV.lasso, s = lambda.min)
# Now we can get predictions for min "best" model
pred.lasso.min.train = predict(fit.CV.lasso, X.train.scale, s = lambda.min, type = "class")
pred.lasso.min.test = predict(fit.CV.lasso, X.valid.scale, s = lambda.min, type = "class")
# Confusion matrix
table(Y.valid, pred.lasso.min.test, dnn = c("Obs", "Pred"))
# Misclassification rate
(miss.lasso.min.train = mean(Y.train != pred.lasso.min.train))
(miss.lasso.min.test = mean(Y.valid != pred.lasso.min.test))

# Lambda 1-SE
lambda.1se = fit.CV.lasso$lambda.1se
coef(fit.CV.lasso, s = lambda.1se)
# Now we can get predictions for min "best" model
pred.lasso.1se.train = predict(fit.CV.lasso, X.train.scale, s = lambda.1se, type = "class")
pred.lasso.1se.test = predict(fit.CV.lasso, X.valid.scale, s = lambda.1se, type = "class")
# Confusion matrix
table(Y.valid, pred.lasso.1se.test, dnn = c("Obs", "Pred"))
# Misclassification rate
(miss.lasso.1se.train = mean(Y.train != pred.lasso.1se.train))
(miss.lasso.1se.test = mean(Y.valid != pred.lasso.1se.test))

