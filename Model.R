## Clear memory
rm(list=ls())

set.seed(24)

## Load packages
library(corrplot)
library(kernlab)
library(caret)
library(randomForest)
library(compiler)
  
## Set working directory
setwd("C:/Users/Jvandegevel/Documents/Coursera/Practical machine learning")

## Read-in training data
trainingdata1 <- read.csv("pml-training.csv", na.strings=c("", "NA", "NULL", " "))

## Remove predictor variables with NA values
trainingdata2 <- trainingdata1[ , colSums(is.na(trainingdata1)) == 0]

## Remove variables that are useless for predicting
remove <- c('X', 'user_name', 'raw_timestamp_part_1', 'raw_timestamp_part_2', 'cvtd_timestamp', 'new_window', 'num_window')
trainingdata3 <- trainingdata2[, -which(names(trainingdata2) %in% remove)]

## Perform PCA to further reduce training data
corMatrix <- cor(trainingdata3[sapply(trainingdata3, is.numeric)])
corrplot(corMatrix, order = "FPC", method = "color", type = "lower", tl.cex = 0.5, tl.col = rgb(0, 0, 0))

preProc <- preProcess(trainingdata3[sapply(trainingdata3, is.numeric)], method = "pca", thresh = 0.99)
trainingdata4 <- predict(preProc, trainingdata3)
  
# Create a train and test set
inTrain <- createDataPartition(y=trainingdata4$classe, p=0.7, list=FALSE)
training <- trainingdata4[inTrain,]
testing <- trainingdata4[-inTrain,]

# Train the model
ModelFit <- randomForest(classe~., data=training, ntree=100, importance=TRUE)

# Make a prediction
PredictionModel <- predict(ModelFit, testing)
Accuracy <- postResample(testing$classe, PredictionModel)
Accuracy

#Read-in validation data
validationdata1 <- read.csv("pml-testing.csv", na.strings=c("", "NA", "NULL", " "))
validationdata2 <- validationdata1[ , colSums(is.na(validationdata1)) == 0]
validationdata3 <- validationdata2[, -which(names(validationdata2) %in% remove)]
validationdata4 <- predict(preProc, validationdata3)



prediction <- predict(ModelFit, validationdata4)

prediction