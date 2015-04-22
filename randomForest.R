library(caret)
library(ggplot2)
library(rpart)
library(kernlab)
library(rattle)
library(AppliedPredictiveModeling)
library(randomForest)
library(rattle)
library(tree)

###load data###
train<-read.csv("pml-training.csv", header=T)
test<-read.csv("pml-testing.csv",header=T)

###split train data set into training and testing subsets###
inTrain<-createDataPartition(y=train$classe,p=0.7,list=FALSE)
training<-train[inTrain,]
testing<-train[-inTrain,]
dim(training)
dim(testing)

###removing zero covariates from training subset###
nsv<-nearZeroVar(training);nsv
training<-training[,-nsv];dim(training)

###removing columns with high amount of 'Na'### 
training<-training[,colSums(is.na(training)) == 0]
dim(training)

###removing timestamps,row numbers, usernames###
training<-training[,-c(1,2,3,4,5)]
dim(training)

###removing highly correlated variables###
M<-abs(cor(training[,-54]))
diag(M)<-0
which(M>0.85,arr.ind=T)
correlated<-findCorrelation(M, cutoff = .85, verbose = FALSE)
training<-training[,-correlated]
dim(training)
summary(training)

###classification with tree###
tree<-tree(classe ~., data = training)
summary(tree)
print(tree)
plot(tree)
text(tree, cex=0.5)

###classification using rpart###
rpart <- train(classe ~ .,method="rpart",data=training)
fancyRpartPlot(rpart$finalModel)
print(rpart)

###applying randomForest###
r<-randomForest(classe ~., data = training, importance = TRUE, do.trace = 100)
print(r) # view results 
#OOB estimate of  error rate< 0.4% = high accuracy

###prediction of testing set observations (train data set subset)### 
predict(r, testing)
cols <- names(training)
t<-table(testing$classe, predict(r, testing[cols]))
prop.table(t, 1)
sum(testing$classe==predict(r, testing[cols])) / nrow(testing)

###imortance of  variables###
importance(r,type=1)
varImpPlot(r, type=1)

###marginal function - with how high precission observations are classified to  correct class ###
plot(margin(r, testing$classe))

plot(r)

###predict test data set variables using randomForest###
predict(r, test)