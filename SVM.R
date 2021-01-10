source("Data.R")

### Activate packages
library(e1071)    # For SVM
library(caret)    # For tuning
library(kernlab)  # For SVM with caret

set.seed(46536737, kind = "Mersenne-Twister")

data = sub_data
### Split the data into training and validation sets
p.train = 0.75
n = nrow(data)
n.train = floor(p.train*n)

ind.random = sample(1:n)
data.train = data[ind.random <= n.train,]
X.train = data.train[,-9]
Y.train = data.train[,9]
data.valid = data[ind.random > n.train,]
X.valid = data.valid[,-9]
Y.valid = data.valid[,9]


###############################
### Support Vector Machines ###
###############################

fit.svm.0 = svm(ARR_DEL15 ~ ., data = data.train, kernel = "radial", 
                cost = 10, gamma = 0.1)
pred.svm.0 = predict(fit.svm.0, data.valid)
table(Y.valid, pred.svm.0, dnn = c("Obs", "Pred"))
(mis.svm.0 = mean(Y.valid != pred.svm.0))



caret.settings = trainControl(method="repeatedcv", number=10, repeats=2,
                              returnResamp="all")

### Next, we choose all the parameter combinations that we will tune
all.Cs = 10^(0:5)
all.sigmas = 10^(-(0:5)) # The caret version of SVM calls it 
# sigma instead of gamma
all.pars = expand.grid(C = all.Cs, sigma = all.sigmas)

### Now we use the train() function to fit our model. See the video for
### a discussion of the syntax. Setting method to "svmRadial" gives us
### a Gaussian radial basis kernel. We can also set method to "svmPoly" to
### get a polynomial kernel.
info.SVM = train(X.train, Y.train, method = "svmRadial",
                 preProcess = c("center", "scale"), tuneGrid = all.pars,
                 trControl = caret.settings)

### The output from train() has lots of useful information. We will focus
### on $resample, which gives the CV accuracy for each model 
### (i.e. 1 - misclassification rate). The formatting
### of these results is different from what we're used to, but we can
### fix that without too much trouble (just copy this code if you need it)
CV.acc.raw = info.SVM$resample
CV.acc.wide = reshape(data=CV.acc.raw[,-2], idvar=c("C", "sigma"), 
                      timevar="Resample", direction="wide")
CV.mis = 1 - t(CV.acc.wide[,c(-1, -2)])
colnames(CV.mis) = apply(CV.acc.wide[,1:2], 1, paste, collapse = ",")
head(CV.mis)

### Make absolute and relative boxplots
boxplot(CV.mis, las = 2)
rel.CV.mis = apply(CV.mis, 1, function(W) W/min(W))
boxplot(t(rel.CV.mis), las=2)

### To me, the best parameter combination looks like C=10^5 and gamma=10^(-5)
### These values are on the boundary of the tuning region, so let's re-tune
all.Cs2 = 10^(4:10)
all.sigmas2 = signif(10^(seq(-4.5, -7, by = -0.5)), 3)
all.pars2 = expand.grid(C = all.Cs2, sigma = all.sigmas2)
info.SVM2 = train(X.train, Y.train, method = "svmRadial",
                  preProcess = c("center", "scale"), tuneGrid = all.pars2,
                  trControl = caret.settings)

### Process output so we can make boxplots
CV.acc.raw2 = info.SVM2$resample
CV.acc.wide2 = reshape(data=CV.acc.raw2[,-2], idvar=c("C", "sigma"), 
                       timevar="Resample", direction="wide")
CV.mis2 = 1 - t(CV.acc.wide2[,c(-1, -2)])
colnames(CV.mis2) = apply(CV.acc.wide2[,1:2], 1, paste, collapse = ",")
head(CV.mis2)

### Make absolute and relative boxplots
boxplot(CV.mis2, las = 2)
rel.CV.mis2 = apply(CV.mis2, 1, function(W) W/min(W))
boxplot(t(rel.CV.mis2), las=2)

### There is a clear winner in this competition, but I can't read the label
### on the plot. We can just find which parameter combination has the
### smallest median relative misclassification rate
all.meds = apply(rel.CV.mis2, 1, median)
ind.best = which.min(all.meds)
(pars.best = all.pars2[ind.best,])

### Finally, let's manually fit an SVM model to the entire training set using
### our chosen parameter values
fit.SVM.rad = svm(quality ~ ., data = data.train, kernel = "radial", 
                  cost = 10^7, gamma = 3.16 * 10^(-6))
pred.SVM.rad = predict(fit.SVM.rad, data.valid)
table(Y.valid, pred.SVM.rad, dnn = c("Obs", "Pred"))
(mis.SVM.rad = mean(Y.valid != pred.SVM.rad))


### You can also fit SVMs using polynomial kernels. The code is a bit
### different, but the idea is the same. I will show how to fit the model
### and run train(), but I won't go through the whole tuning process again.
### As with radial basis kernels, things are named differently between
### svm() and train().


### svm() ###

### degree - degree of polynomial
### cost - C in Tom's notes
### gamma - s in formula for polynomial kernel from Tom's notes
### coef0 - c in formula for polynomial kernel from Tom's notes
fit.SVM.poly = svm(quality ~ ., data = data.train, kernel = "poly", 
                   degree = 2, cost = 10, gamma = 1, coef0 = 0)

### train() ###

### degree - degree of polynomial
### cost - C in Tom's notes
### scale - s in formula for polynomial kernel from Tom's notes
all.degrees = 1:2
all.scales = c(0.5, 1, 2)
all.Cs = 10^(c(0, 2, 4))
all.pars.poly = expand.grid(degree = all.degrees,
                            scale = all.scales, C = all.Cs)
# info.SVM.poly = train(X.train, Y.train, method = "svmPoly",
#   preProcess = c("center", "scale"), tuneGrid = all.pars.poly, 
#   trControl = caret.settings)


### Validation-set misclassification rates
### KNN, K=3:               0.508
### KNN, CV-min:            0.460
### KNN, CV-1se:            0.427
### Logistic Regression:    0.387
### Logistic LASSO, CV-min: 0.387
### Logistic LASSO, CV-1se: 0.476
### LDA:                    0.379
### QDA:                    0.444
### GAM:                    0.363
### Naive Bayes:            0.524
### Naive Bayes with PCA:   0.492
### Full Tree:              0.508
### Tree, CV-min:           0.419
### Tree, CV-1se:           0.435
### Random Forest:          0.444
### Naive NNet              0.452
### Tuned NNet              0.419
### Naive SVM - GRB         0.435
### Tuned SVM - GRB         0.403