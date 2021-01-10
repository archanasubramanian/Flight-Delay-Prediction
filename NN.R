source("Data.R")
library(nnet)        # For neural networks

set.seed(46536737, kind = "Mersenne-Twister")

#######################
### Neural Networks ###
#######################


p.train = 0.75
n = nrow(sub_data)
n.train = floor(p.train*n)

data = sub_data
ind.random = sample(1:n)
data.train = data[ind.random <= n.train,]
X.train.raw = data.train[,-9]
Y.train = data.train[,9]
data.valid = data[ind.random > n.train,]
X.valid.raw = data.valid[,-9]
Y.valid = data.valid[,9]

### Rescale columns of X to fall between 0 and 1 using Tom's function
rescale <- function(x1,x2){
  for(col in 1:ncol(x1)){
    a <- min(x2[,col])
    b <- max(x2[,col])
    x1[,col] <- (x1[,col]-a)/(b-a)
  }
  x1
}
X.train = rescale(X.train.raw[,-(9:10)], X.train.raw[,-(9:10)])
X.train = cbind(X.train,X.train.raw[,(9:10)] )
X.valid = rescale(X.valid.raw[,-(9:10)], X.train.raw[,-(9:10)])
X.valid = cbind(X.valid,X.valid.raw[,(9:10)] )

### Make sure that we rescaled correctly
summary(X.train)
summary(X.valid)

### Convert Y to a factor and the corresponding indicator. The nnet() function
### needs a separate indicator for each class. We can get these indicators
### using the class.ind() function after converting our response to a factor
#Y.train.fact = factor(Y.train, levels = c("low", "med", "high"))
Y.train.num = class.ind(Y.train)
#Y.valid.fact = factor(Y.valid, levels = c("low", "med", "high"))
Y.valid.num = class.ind(Y.valid)

### Check that our conversion worked
head(Y.train.fact)
head(Y.train.num)
head(Y.valid.fact)
head(Y.valid.num)


### Now we can fit a neural network model. We do this using the nnet()
### function from the nnet package. We still use size and decay to set
### the number of hidden nodes and the shrinkage respectively. maxit still
### controls the maximum number of iterations (this just needs to be large
### enough that the function says it has converged). Instead of setting 
### linout, we now need to set softmax to TRUE to get classification labels.
### To demonstrate, let's fit a neural net with arbitrarily chosen tuning
### parameters (please don't do this in practice; it is a terrible idea).
fit.nnet.0 = nnet(X.train, Y.train.num, size = 1, decay = 0, maxit = 20000, 
                  softmax = T)

### Remember that the nnet function can get stuck in local minima, so
### we need to re-run the function a few times with the same parameter
### values and choose the model with the lowest sMSE. Let's use Tom's method
### to do this. 
MSE.best = Inf    ### Initialize sMSE to largest possible value (infinity)
M = 20            ### Number of times to refit.

for(i in 1:M){
  ### For convenience, we stop nnet() from printing information about
  ### the fitting process by setting trace = F.
  this.nnet = nnet(X.train, Y.train.num, size = 1, decay = 0, maxit = 2000, 
                   softmax = T, trace = F)
  this.MSE = this.nnet$value
  if(this.MSE < MSE.best){
    NNet.best.0 = this.nnet
    MSE.best = this.MSE
  }
}

### Now we can evaluate the validation-set performance of our naive neural
### network. We can get the predicted class labels using the predict()
### function and setting type to "class"
pred.nnet.0 = predict(NNet.best.0, X.valid, type = "class")

table(Y.valid, pred.nnet.0, dnn = c("Obs", "Pred"))

(mis.nnet.0 = mean(Y.valid != pred.nnet.0))


