# Clear environment workspace
rm(list=ls())
# Load data
train <- read.csv("train.csv")
test <- read.csv("test.csv")
samplesub <- read.csv("sampleSubmission.csv")
# Remove id column so it doesn't get picked up by the current classifier
train <- train[,-1]
summary(train)
summary(test)
# Create sample train and test datasets for prototyping new models from the train dataset
strain <- train[sample(nrow(train), 10000, replace=FALSE),]
stest <- train[sample(nrow(train), 5000, replace=FALSE),]

install.packages('rpart')
library(rpart)
set.seed(13)
# Create a decision tree model using the target field as the response and all 93 features as inputs (.)
fit1 <- rpart(as.factor(target) ~ ., data=train, method="class")
# Plot the decision tree
par(xpd=TRUE)
plot(fit1, compress=TRUE)
title(main="rpart")
text(fit1)
# Test the rpart (tree) model on the holdout test dataset
fit1.pred <- predict(fit1, stest, type="class")
# Create a confusion matrix of predictions vs actuals
table(fit1.pred,stest$target)
# Determine the error rate for the model
fit1$error <- 1-(sum(fit1.pred==stest$target)/length(stest$target))
fit1$error


install.packages('randomForest')
library(randomForest)
set.seed(16)
ptm2 <- proc.time()
# Create a random forest model using the target field as the response and all 93 features as inputs (.)
fit2 <- randomForest(as.factor(target) ~ ., data=strain, importance=TRUE, ntree=50, mtry=2)
# Finish timing the model
fit2.time <- proc.time() - ptm2
# Create a dotchart of variable/feature importance as measured by a Random Forest
varImpPlot(fit2)
# Test the randomForest model on the holdout test dataset
fit2.pred <- predict(fit2, stest, type="response")
# Create a confusion matrix of predictions vs actuals
table(fit2.pred,stest$target)
# Determine the error rate for the model
fit2$error <- 1-(sum(fit2.pred==stest$target)/length(stest$target))
fit2$error

# Install gbm package
install.packages('gbm')
library(gbm)
set.seed(17)
# Begin recording the time it takes to create the model
ptm3 <- proc.time()
# Create a random forest model using the target field as the response and all 93 features as inputs (.)
fit3 <- gbm(target ~ ., data=strain, distribution="multinomial", n.trees=1000,
            shrinkage=0.05, interaction.depth=12)
# Finish timing the model
fit3.time <- proc.time() - ptm3
# Test the boosting model on the holdout test dataset
trees <- gbm.perf(fit3)
fit3.stest <- predict(fit3, stest, n.trees=trees, type="response")
fit3.stest <- as.data.frame(fit3.stest)
names(fit3.stest) <- c("Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9")
fit3.stest.pred <- rep(NA,2000)
for (i in 1:nrow(stest)) {
  fit3.stest.pred[i] <- colnames(fit3.stest)[(which.max(fit3.stest[i,]))]}
fit3.pred <- as.factor(fit3.stest.pred)
# Create a confusion matrix of predictions vs actuals
table(fit3.pred,stest$target)
# Determine the error rate for the model
fit3$error <- 1-(sum(fit3.pred==stest$target)/length(stest$target))
fit3$error


