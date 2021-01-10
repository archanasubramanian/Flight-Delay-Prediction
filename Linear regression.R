library(dplyr)
source("Data.R")

fit.Solar = lm(train_set$traffic_volume ~ ., data = tarin_set)