# Carreguem les llibreries que utilitzaremc
llibreries <- function() {
  require(rpart)
  require(rpart.plot)
  require(ElemStatLearn)
  require(e1071)
  require(MASS)
  require(adabag)
  require(randomForest)
}

########
#
# Regressio Logistica
#
#######
reg_log_astro <- function() {
  #Preparacio de les dades
  astroparticle.train<-read.table("svmguide1.train.txt",header=TRUE)
  astroparticle.test<-read.table("svmguide1.test.txt",header=TRUE)
  astroparticle.train <- data.frame(astroparticle.train)
  astroparticle.test <- data.frame(astroparticle.test)
  astroparticle.train$y <- as.factor(astroparticle.train$y)
  astroparticle.test$y <- as.factor(astroparticle.test$y)


  astro.logit1 <- glm(y~.,data=astroparticle.train,family=binomial)

  astro.pred <- predict(astro.logit1,newdata<-astroparticle.test,type="response")
  astro.pred.crisp<-1*(astro.pred>=0.5)

  mc<-table("True"=astroparticle.test$y,"Predicted"=astro.pred.crisp)
  print(mc)
  print(sum(diag(mc))/sum(mc))
  return(mc)
}

reg_log_vehicle <- function() {
  vehicle.train<-read.table("svmguide3.train.txt",header=FALSE,fill=TRUE)
  vehicle.test<-read.table("svmguide3.test.txt",header=FALSE,fill=TRUE)
  colnames(vehicle.train)<-c("y",sprintf("x%02d",1:22))
  colnames(vehicle.test)<-c("y",sprintf("x%02d",1:22))
  vehicle.train<-vehicle.train[,-23]
  vehicle.test<-vehicle.test[,-23]
  vehicle.train$y <- as.factor(vehicle.train$y)
  vehicle.test$y <- as.factor(vehicle.test$y)
  vehicle.logit1 <- glm(y~.,data=vehicle.train,family=binomial)
  vehicle.pred <- predict(vehicle.logit1,newdata<-vehicle.test,type="response")
  vehicle.pred.crisp<-1*(vehicle.pred>=0.5)

  mc<-table("True"=vehicle.test$y,"Predicted"=vehicle.pred.crisp)
  print(mc)

}

########
#
# LDA
#
#######
lda_astro <- function() {
  #Preparacio de les dades
  astroparticle.train<-read.table("svmguide1.train.txt",header=TRUE)
  astroparticle.test<-read.table("svmguide1.test.txt",header=TRUE)
  astroparticle.train <- data.frame(astroparticle.train)
  astroparticle.test <- data.frame(astroparticle.test)
  astroparticle.train$y <- as.factor(astroparticle.train$y)
  astroparticle.test$y <- as.factor(astroparticle.test$y)


  astro.lda <- lda(y~.,data=astroparticle.train)

  astro.pred <- predict(astro.lda,newdata<-astroparticle.test)


  mc<-table("True"=astroparticle.test$y,"Predicted"=astro.pred$class)
  print(mc)
  print(sum(diag(mc))/sum(mc))
  return(mc)
}

lda_vehicle <- function(){
  vehicle.train<-read.table("svmguide3.train.txt",header=FALSE,fill=TRUE)
  vehicle.test<-read.table("svmguide3.test.txt",header=FALSE,fill=TRUE)
  colnames(vehicle.train)<-c("y",sprintf("x%02d",1:22))
  colnames(vehicle.test)<-c("y",sprintf("x%02d",1:22))
  vehicle.train<-vehicle.train[,-23]
  vehicle.test<-vehicle.test[,-23]
  vehicle.train$y <- as.factor(vehicle.train$y)
  vehicle.test$y <- as.factor(vehicle.test$y)
  vehicle.logit1 <- lda(y~.,data=vehicle.train)
  vehicle.pred <- predict(vehicle.logit1,newdata<-vehicle.test,type="response")

  mc<-table("True"=vehicle.test$y,"Predicted"=vehicle.pred$class)
  print(mc)

}

lda_bio <- function() {
  bioinformatics<-scan("svmguide2.txt",what=double(),nmax=21*391)
  Bioinformatics<-as.data.frame(t(matrix(bioinformatics,nrow=21)))
  Bioinformatics[,1]<-as.factor(Bioinformatics[,1])
  colnames(Bioinformatics)<-c("y",sprintf("x%02d",1:20))

  n <- nrow(Bioinformatics)
	ntrain <- ceiling(0.6*n)
	ntest <- n - ntrain
	Itrain <- sample(1:n,ntrain,replace=FALSE)
	bio.train <- Bioinformatics[Itrain,]
	bio.test <- Bioinformatics[-Itrain,]

  #Ajusta el model
	bio.lda <-lda(y~.,data=bio.train)

	#spam.test <- spam.test[,-58] #<- Aparentment no afecta al resultat
	#Fa la prediccio
	bio.pred<-predict(bio.lda,newdata<-bio.test)
	#Matriu de confusio
	C<-table("True"=bio.test$y,"Predicted"=bio.pred$class)
	print(C)
	print(sum(diag(C))/sum(C))
	return(C)
}

########
#
# SVM
#
#######
svm_astro <-function() {
  #Preparacio de les dades
  astroparticle.train<-read.table("svmguide1.train.txt",header=TRUE)
  astroparticle.test<-read.table("svmguide1.test.txt",header=TRUE)
  astroparticle.train <- data.frame(astroparticle.train)
  astroparticle.test <- data.frame(astroparticle.test)
  astroparticle.train$y <- as.factor(astroparticle.train$y)
  astroparticle.test$y <- as.factor(astroparticle.test$y)


  #astro1 <- svm(y~.,data=astroparticle.train)
  astro1 <- best.tune(svm, y~.,data=astroparticle.train,ranges = list(gamma = 2^(-4:4), cost = 2^(-4:4)),scale=TRUE,tunecontrol = tune.control(sampling = "fix"))

  astro.pred <- predict(astro1,newdata<-astroparticle.test,type="response")
  print(astro1)

  mc<-table("True"=astroparticle.test$y,"Predicted"=astro.pred)
  print(mc)
  print(sum(diag(mc))/sum(mc))


  return(mc)
}

svm_vehicle <- function() {
  vehicle.train<-read.table("svmguide3.train.txt",header=FALSE,fill=TRUE)
  vehicle.test<-read.table("svmguide3.test.txt",header=FALSE,fill=TRUE)
  colnames(vehicle.train)<-c("y",sprintf("x%02d",1:22))
  colnames(vehicle.test)<-c("y",sprintf("x%02d",1:22))
  vehicle.train$y <- as.factor(vehicle.train$y)
  vehicle.test$y <- as.factor(vehicle.test$y)
  vehicle.train<-vehicle.train[,-23]
  vehicle.test<-vehicle.test[,-23]
  vehicle.train <-data.frame(vehicle.train)
  vehicle.test <-data.frame(vehicle.test)

  ##vehicle1 <- svm(y~.,data=vehicle.train)
  vehicle1 <- best.tune(svm, y~.,data=vehicle.train,ranges = list(gamma = 2^(-4:4), cost = 2^(-4:4)),scale=TRUE,tunecontrol = tune.control(sampling = "fix"))
  print(vehicle1)
  vehicle.pred <- predict(vehicle1,newdata<-vehicle.test,type="response")
  mc<-table("True"=vehicle.test$y,"Predicted"=vehicle.pred)
  print(mc)

}

svm_bio <- function() {
  bioinformatics<-scan("svmguide2.txt",what=double(),nmax=21*391)
  Bioinformatics<-as.data.frame(t(matrix(bioinformatics,nrow=21)))
  Bioinformatics[,1]<-as.factor(Bioinformatics[,1])
  colnames(Bioinformatics)<-c("y",sprintf("x%02d",1:20))

  n <- nrow(Bioinformatics)
	ntrain <- ceiling(0.6*n)
	ntest <- n - ntrain
	Itrain <- sample(1:n,ntrain,replace=FALSE)
	bio.train <- Bioinformatics[Itrain,]
	bio.test <- Bioinformatics[-Itrain,]

  #Ajusta el model
	#bio.lda <-svm(y~.,data=bio.train)
  bio.svm <-best.tune(svm, y~.,data=bio.train,ranges = list(gamma = 2^(-4:4), cost = 2^(-4:4)),scale=TRUE,tunecontrol = tune.control(sampling = "fix"))
  print(bio.svm)
	#spam.test <- spam.test[,-58] #<- Aparentment no afecta al resultat
	#Fa la prediccio
	bio.pred<-predict(bio.svm,newdata<-bio.test)
	#Matriu de confusio
	C<-table("True"=bio.test$y,"Predicted"=bio.pred)
	print(C)
	print(sum(diag(C))/sum(C))
	return(C)
}

lda_i_svm_bio <- function() {
  bioinformatics<-scan("svmguide2.txt",what=double(),nmax=21*391)
  Bioinformatics<-as.data.frame(t(matrix(bioinformatics,nrow=21)))
  Bioinformatics[,1]<-as.factor(Bioinformatics[,1])
  colnames(Bioinformatics)<-c("y",sprintf("x%02d",1:20))

  n <- nrow(Bioinformatics)
	ntrain <- ceiling(0.6*n)
	ntest <- n - ntrain
	Itrain <- sample(1:n,ntrain,replace=FALSE)
	bio.train <- Bioinformatics[Itrain,]
	bio.test <- Bioinformatics[-Itrain,]

  #Ajusta el model
	#bio.lda <-svm(y~.,data=bio.train)
  bio.svm <-best.tune(svm, y~.,data=bio.train,ranges = list(gamma = 2^(-4:4), cost = 2^(-4:4)),scale=TRUE,tunecontrol = tune.control(sampling = "fix"))
  print("SVM")
  print(bio.svm)
	#spam.test <- spam.test[,-58] #<- Aparentment no afecta al resultat
	#Fa la prediccio
	bio.pred<-predict(bio.svm,newdata<-bio.test)
	#Matriu de confusio
	C<-table("True"=bio.test$y,"Predicted"=bio.pred)
	print(C)
	print(sum(diag(C))/sum(C))

  bio.lda <-lda(y~.,data=bio.train)
  print("LDA")
  #spam.test <- spam.test[,-58] #<- Aparentment no afecta al resultat
  #Fa la prediccio
  bio.pred<-predict(bio.lda,newdata<-bio.test)
  #Matriu de confusio
  C<-table("True"=bio.test$y,"Predicted"=bio.pred$class)
  print(C)
  print(sum(diag(C))/sum(C))

}


########
#
# CART
#
#######
cart_astro <- function(){
  #Preparacio de les dades
  astroparticle.train<-read.table("svmguide1.train.txt",header=TRUE)
  astroparticle.test<-read.table("svmguide1.test.txt",header=TRUE)
  astroparticle.train <- data.frame(astroparticle.train)
  astroparticle.test <- data.frame(astroparticle.test)
  astroparticle.train$y <- as.factor(astroparticle.train$y)
  astroparticle.test$y <- as.factor(astroparticle.test$y)

  astro1 <- best.rpart(formula(y~.),data=astroparticle.train,minsplit=seq(2,20,2))
  plot(astro1)
  text(astro1,use.n=TRUE,xpd=2)
  #print(astro1)
  print(summary(astro1))

  astro.pred <- predict(astro1,astroparticle.test,type="class")
  mc<-table("True"=astroparticle.test$y,"Predicted"=astro.pred)
  print(mc)
  print(sum(diag(mc))/sum(mc))
}

cart_vehicle <- function() {
  vehicle.train<-read.table("svmguide3.train.txt",header=FALSE,fill=TRUE)
  vehicle.test<-read.table("svmguide3.test.txt",header=FALSE,fill=TRUE)
  colnames(vehicle.train)<-c("y",sprintf("x%02d",1:22))
  colnames(vehicle.test)<-c("y",sprintf("x%02d",1:22))
  vehicle.train$y <- as.factor(vehicle.train$y)
  vehicle.test$y <- as.factor(vehicle.test$y)
  vehicle.train<-vehicle.train[,-23]
  vehicle.test<-vehicle.test[,-23]
  vehicle.train <-data.frame(vehicle.train)
  vehicle.test <-data.frame(vehicle.test)

  ##vehicle1 <- svm(y~.,data=vehicle.train)
  vehicle1 <- best.rpart(formula(y~.),data=vehicle.train,minsplit=seq(2,20,2))
  plot(vehicle1)
  text(vehicle1,use.n=TRUE,xpd=2)
  #print(vehicle1)
  print(summary(vehicle1))

  vehicle.pred <- predict(vehicle1,vehicle.test,type="class")
  #vehicle.pred <- predict(vehicle1,newdata<-vehicle.test,type="response")
  mc<-table("True"=vehicle.test$y,"Predicted"=vehicle.pred)
  print(mc)

}

cart_bio <-function() {
  bioinformatics<-scan("svmguide2.txt",what=double(),nmax=21*391)
  Bioinformatics<-as.data.frame(t(matrix(bioinformatics,nrow=21)))
  Bioinformatics[,1]<-as.factor(Bioinformatics[,1])
  colnames(Bioinformatics)<-c("y",sprintf("x%02d",1:20))

  n <- nrow(Bioinformatics)
  ntrain <- ceiling(0.6*n)
  ntest <- n - ntrain
  Itrain <- sample(1:n,ntrain,replace=FALSE)
  bio.train <- Bioinformatics[Itrain,]
  bio.test <- Bioinformatics[-Itrain,]

  #Ajusta el model
  #bio.lda <-svm(y~.,data=bio.train)
  bio1 <- best.rpart(formula(y~.),data=bio.train,minsplit=seq(2,20,2))
  plot(bio1)
  text(bio1,use.n=TRUE,xpd=2)
  #print(bio1)
  #print(summary(bio1))

  bio.pred <- predict(bio1,bio.test,type="class")
  #bio.pred <- predict(bio1,newdata<-bio.test,type="response")
  mc<-table("True"=bio.test$y,"Predicted"=bio.pred)
  print(mc)
  print(sum(diag(mc))/sum(mc))
}

###########
#
# ADABAG
#
############
adabag_astro <-function(mf=10) {
  #Preparacio de les dades
  astroparticle.train<-read.table("svmguide1.train.txt",header=TRUE)
  astroparticle.test<-read.table("svmguide1.test.txt",header=TRUE)
  astroparticle.train <- data.frame(astroparticle.train)
  astroparticle.test <- data.frame(astroparticle.test)
  astroparticle.train$y <- as.factor(astroparticle.train$y)
  astroparticle.test$y <- as.factor(astroparticle.test$y)
  print("mfinal:")
  print(mf)
  print("Temps emprat")
  ptm <- proc.time()
  astro.bagging <-bagging(y~.,data=astroparticle.train,mfinal=mf)
  print(proc.time() - ptm)
  astro.bagging.pred <- predict.bagging(astro.bagging,newdata<-astroparticle.test)

  astro.bagging.conf <- astro.bagging.pred$confusion
  #erro <- prob.err(astro.bagging.conf)
  print(astro.bagging.conf)
  print(sum(diag(astro.bagging.conf))/sum(astro.bagging.conf))
  #print(erro)
}

adabag_vehicle <- function(mf=10) {
  vehicle.train<-read.table("svmguide3.train.txt",header=FALSE,fill=TRUE)
  vehicle.test<-read.table("svmguide3.test.txt",header=FALSE,fill=TRUE)
  colnames(vehicle.train)<-c("y",sprintf("x%02d",1:22))
  colnames(vehicle.test)<-c("y",sprintf("x%02d",1:22))
  vehicle.train$y <- as.factor(vehicle.train$y)
  vehicle.test$y <- as.factor(vehicle.test$y)
  vehicle.train<-vehicle.train[,-23]
  vehicle.test<-vehicle.test[,-23]
  vehicle.train <-data.frame(vehicle.train)
  vehicle.test <-data.frame(vehicle.test)
  print("mfinal:")
  print(mf)
  print("Temps emprat")
  ptm <- proc.time()
  vehicle.bagging <-bagging(y~.,data=vehicle.train,mfinal=mf)
  print(proc.time() - ptm)
  vehicle.bagging.pred <- predict.bagging(vehicle.bagging,newdata<-vehicle.test)

  vehicle.bagging.conf <- vehicle.bagging.pred$confusion
  #erro <- prob.err(vehicle.bagging.conf)
  print(vehicle.bagging.conf)
}

adabag_bio <- function(mf=10) {
  bioinformatics<-scan("svmguide2.txt",what=double(),nmax=21*391)
  Bioinformatics<-as.data.frame(t(matrix(bioinformatics,nrow=21)))
  Bioinformatics[,1]<-as.factor(Bioinformatics[,1])
  colnames(Bioinformatics)<-c("y",sprintf("x%02d",1:20))

  n <- nrow(Bioinformatics)
  ntrain <- ceiling(0.6*n)
  ntest <- n - ntrain
  Itrain <- sample(1:n,ntrain,replace=FALSE)
  bio.train <- Bioinformatics[Itrain,]
  bio.test <- Bioinformatics[-Itrain,]

  print(mf)
  print("Temps emprat")
  ptm <- proc.time()
  bio.bagging <-bagging(y~.,data=bio.train,mfinal=mf)
  print(proc.time() - ptm)
  bio.bagging.pred <- predict.bagging(bio.bagging,newdata<-bio.test)

  bio.bagging.conf <- bio.bagging.pred$confusion
  #erro <- prob.err(bio.bagging.conf)
  print(bio.bagging.conf)
  print(sum(diag(bio.bagging.conf))/sum(bio.bagging.conf))

}

proves_adabag_astro <- function() {
  a <- seq(2,30,2)
  for(i in a){
    adabag_astro(i)
  }
}
proves_adabag_vehicle <- function() {
  a <- seq(2,30,2)
  for(i in a){
    adabag_vehicle(i)
  }
}
proves_adabag_bio <- function() {
  a <- seq(2,30,2)
  for(i in a){
    adabag_bio(i)
  }
}

##########
#
# RANDOM FOREST
#
##########
randomForest_astro <- function() {
  astroparticle.train<-read.table("svmguide1.train.txt",header=TRUE)
  astroparticle.test<-read.table("svmguide1.test.txt",header=TRUE)
  astroparticle.train <- data.frame(astroparticle.train)
  astroparticle.test <- data.frame(astroparticle.test)
  astroparticle.train$y <- as.factor(astroparticle.train$y)
  astroparticle.test$y <- as.factor(astroparticle.test$y)


  print("Temps emprat")
  ptm <- proc.time()
  astro.rf <- randomForest(y~.,data=astroparticle.train,importance=TRUE,proximity=TRUE)
  print(proc.time() - ptm)
  print(astro.rf)
  astro.pred <-predict(astro.rf,newdata<-astroparticle.test)
  mc<-table("True"=astroparticle.test$y,"Predicted"=astro.pred)
  print(mc)
  print(sum(diag(mc))/sum(mc))

}

randomForest_vehicle <- function() {
  vehicle.train<-read.table("svmguide3.train.txt",header=FALSE,fill=TRUE)
  vehicle.test<-read.table("svmguide3.test.txt",header=FALSE,fill=TRUE)
  colnames(vehicle.train)<-c("y",sprintf("x%02d",1:22))
  colnames(vehicle.test)<-c("y",sprintf("x%02d",1:22))
  vehicle.train$y <- as.factor(vehicle.train$y)
  vehicle.test$y <- as.factor(vehicle.test$y)
  vehicle.train<-vehicle.train[,-23]
  vehicle.test<-vehicle.test[,-23]


  print("Temps emprat")
  ptm <- proc.time()
  vehicle.rf <- randomForest(y~.,data=vehicle.train,importance=TRUE,proximity=TRUE)
  print(proc.time() - ptm)
  print(vehicle.rf)
  vehicle.pred <- predict(vehicle.rf,vehicle.test,type="class")
  #vehicle.pred <- predict(vehicle1,newdata<-vehicle.test,type="response")
  mc<-table("True"=vehicle.test$y,"Predicted"=vehicle.pred)
  print(mc)
}

randomForest_bio <- function(){
  bioinformatics<-scan("svmguide2.txt",what=double(),nmax=21*391)
  bio<-as.data.frame(t(matrix(bioinformatics,nrow=21)))
  bio[,1]<-as.factor(bio[,1])
  colnames(bio)<-c("y",sprintf("x%02d",1:20))
  n <- nrow(bio)
  ntrain <- ceiling(0.6*n)
  ntest <- n - ntrain
  Itrain <- sample(1:n,ntrain,replace=FALSE)
  bio.train <- bio[Itrain,]
  bio.test <- bio[-Itrain,]
  print("Temps emprat")
  ptm <- proc.time()
  bio.rf <- randomForest(y~.,data=bio.train,importance=TRUE,proximity=TRUE)
  print(proc.time() - ptm)
  print(bio.rf)

  bio.pred <- predict(bio.rf,bio.test,type="class")
  #bio.pred <- predict(bio1,newdata<-bio.test,type="response")
  mc<-table("True"=bio.test$y,"Predicted"=bio.pred)
  print(mc)
  print(sum(diag(mc))/sum(mc))
}

llibreries()
