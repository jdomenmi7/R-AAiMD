###
## Aprenentatge Automatic i Mineria de Dades
## Alumna: Judit Domenech Miro
## Practica 3
###

####
# Classificacio lineal per minims quadrats
####

##
# Classificacio lineal per minims quadrats, Dades Spam
##
class_lm_Spam<-function() {
	require(ElemStatLearn)
	## Preparem les dades 
	spam_data <- spam
	spam_data$spam <- as.factor(spam_data$spam)
	y <- spam_data$spam #Guardem la classificacio
	Y <- cbind((y=="email"),(y=="spam")) #Dues columnes email, spam que indiquen true o false segons la classe
	colnames(Y) <- c("Y_email","Y_spam")
	#Treiem la columna spam del data frame i hi afegim les dues columnes creades
	spam2 <- data.frame(spam_data[,-58],Y)
	#Preparem el conjunt de training i el conjunt de test.
	n <- nrow(spam2)
	ntrain <- ceiling(0.6*n)
	ntest <- n - ntrain
	Itrain <- sample(1:n,ntrain,replace=FALSE)
	spam2.train <- spam2[Itrain,]
	spam2.test <- spam2[-Itrain,]
	#Ajustem el model lineal
	spam2.lm1 <-lm(cbind(Y_email,Y_spam)~.,data=spam2.train)
	# Preparem les dades per fer el test treient les columnes Y_email, Y_spam del conjunt de test
	spam2.test.to.predict <- spam2.test[,-(58:59)]
	#Fem la prediccio
	Yhat<-predict(spam2.lm1,newdata<-spam2.test.to.predict)
	#Preparem la matriu de confusio
	apply(Yhat,1,sum)
	Yhat.max <- apply(Yhat,1,max)
	Yhat.ind <- 1*cbind(Yhat[,1]==Yhat.max,Yhat[,2]==Yhat.max)
	Y.test<-Y[-Itrain,]
	C <- t(Y.test)%*%Yhat.ind
	print(C)
	#Calculem les prediccions encertades sumant la diagonal i dividint-la entre el total
	print(sum(diag(C))/sum(C))
	return(C)
}
##
# Classificacio lineal per minims quadrats, Dades Vowel
# El conjunt d'entrenament i el de test ja venen separats
##
class_lm_Vowel <- function() {
	require(ElemStatLearn)
	#PReparacio dades
	data(vowel.train)
	vowel.train$y <- factor(vowel.train$y)
	data(vowel.test)
	vowel.test$y <- factor(vowel.test$y)
	y <-vowel.train$y
	Y <- cbind((y==1),(y==2),(y==3),(y==4),(y==5),(y==6),(y==7),(y==8),(y==9),(y==10),(y==11))
	colnames(Y)<-c("Y1","Y2","Y3","Y4","Y5","Y6","Y7","Y8","Y9","Y10","Y11")
	vowel2.train <- data.frame(vowel.train[,-1],Y)
	vowel2.test <- data.frame(vowel.test)
	#Ajustem el model
	vowel2.lm1 <- lm(cbind(Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8,Y9,Y10,Y11)~.,data=vowel2.train)
	yreal<-vowel.test$y
	Y.test<-cbind((yreal==1),(yreal==2),(yreal==3),(yreal==4),(yreal==5),(yreal==6),(yreal==7),(yreal==8),(yreal==9),(yreal==10),(yreal==11))
	vowel2.test.to.predict <- vowel.test[,-1]
	#Fem la prediccio
	Yhat <- predict(vowel2.lm1,newdata<-vowel2.test.to.predict)
	#Preparem la matriu de confusio
	apply(Yhat,1,sum)
	Yhat.max <- apply(Yhat,1,max)
	Yhat.ind <- 1*cbind(Yhat[,1]==Yhat.max,Yhat[,2]==Yhat.max,Yhat[,3]==Yhat.max,Yhat[,4]==Yhat.max,Yhat[,5]==Yhat.max,
		Yhat[,6]==Yhat.max, Yhat[,7]==Yhat.max,Yhat[,8]==Yhat.max,Yhat[,9]==Yhat.max,Yhat[,10]==Yhat.max,Yhat[,11]==Yhat.max)
	C<-t(Y.test)%*%Yhat.ind
	print(C)
	print(sum(diag(C))/sum(C))
	#En aquest cas ja ens venen separats un conjunt de training i un conjunt de test
}

#####################################################################################################################
#####################################################################################################################

####
# Regressio logistica
####

##
#Regressio logistica, Dades Spam
##
class_reg_log_Spam <- function() {
	require(ElemStatLearn)	
	#Preparacio de les dades i separacio en subconjunt de train i test
	spam_data <- data.frame(spam)
	spam_data$spam <- as.factor(spam_data$spam)
	n <- nrow(spam)
	ntrain <- ceiling(0.6*n)
	ntest <- n - ntrain
	Itrain <- sample(1:n,ntrain,replace=FALSE)
	spam.train <- spam_data[Itrain,]
	spam.test <- spam_data[-Itrain,]
	#Ajustem el model de regressio logistica
	spam.logit1 <-glm(spam~.,data=spam.train,family=binomial)
	#spam.test.to.predict <- spam.test[,-(58:59)]
	#Fem la prediccio
	spam.pred<-predict(spam.logit1,newdata<-spam.test,type="response")
	spam.pred.crisp<-1*(spam.pred>=0.5)
	#Calculem la matriu de confusio
	C<-table("True"=spam.test$spam,"Predicted"=spam.pred.crisp)
	print(C)
	print(sum(diag(C))/sum(C))
	return(C)
}

##
# Regressio logistica dades Vowel
# Segueix l'exemple que es mostra en: http://www.r-bloggers.com/how-to-multinomial-regression-models-in-r/
# En aquest cas ja ens venen separats un conjunt de training i un conjunt de test
##
class_reg_log_Vowel <- function() {
	require(ElemStatLearn)
	require(nnet)
	#Preparacio dades
	data(vowel.train)
	vowel.train$y <- factor(vowel.train$y)	
	data(vowel.test)
	vowel.test$y <- factor(vowel.test$y)
	#Ajustem el model, en aquest cas utilitzan la funcio multinom
	vowel.logit1 <- multinom(y~.,data=vowel.train)

	vowel2.test <- vowel.test[,-1]
	#print(summary(vowel.logit1))
	#Calcula les probabilitats individual i acumulativa
	vowel.probs<-predict(vowel.logit1,newdata=vowel2.test,"probs")
	cum.probs <- t(apply(vowel.probs,1,cumsum))

	vals <-runif(nrow(vowel2.test))
	# Join cumulative probabilities and random draws
    tmp <- cbind(cum.probs,vals)
 
    # For each row, get choice index.
    k <- ncol(vowel.probs)
    ids <- 1 + apply(tmp,1,function(x) length(which(x[1:k] < x[k+1])))
 
	C<-table("True"=vowel.test$y,"Predicted"=ids)
	print(C)
	print(sum(diag(C))/sum(C))
	return(C)
}

####
# Classificacio utilitzant el Discriminador Lineal de Fisher
###

##
# Discriminador Lineal de Fisher, Dades Spam 
##
class_fisher_Spam <- function() {
	require(MASS)
	require(ElemStatLearn)
	#Preparacio dades
	spam_data <- spam
	spam_data$spam <- as.factor(spam_data$spam)
	n <- nrow(spam)
	ntrain <- ceiling(0.6*n)
	ntest <- n - ntrain
	Itrain <- sample(1:n,ntrain,replace=FALSE)
	spam.train <- spam[Itrain,]
	spam.test <- spam[-Itrain,]
	#Ajusta el model
	spam.lda <-lda(spam~.,data=spam.train)
	Y<-spam.test$spam
	#spam.test <- spam.test[,-58] #<- Aparentment no afecta al resultat
	#Fa la prediccio
	spam.pred<-predict(spam.lda,newdata<-spam.test)
	#Matriu de confusio
	C<-table("True"=Y,"Predicted"=spam.pred$class)
	print(C)
	print(sum(diag(C))/sum(C))
	return(C)
}
##
# Discriminador Lineal de Fisher, Dades Vowel 
##
class_fisher_Vowel <- function() {
	require(MASS)
	require(ElemStatLearn)
	#Prepara les dades
	vowel.train$y <- as.factor(vowel.train$y)
	vowel.test$y <- as.factor(vowel.test$y)
	#Ajusta el model
	vowel.lda <- lda(y~.,data=vowel.train)
	#Fa la prediccio
	vowel.pred <- predict(vowel.lda,newdata<-vowel.test)
	#Fa la matriu de confusio
	C<-table("True"=vowel.test$y,"Predicted"=vowel.pred$class)
	print(C)
	print(sum(diag(C))/sum(C))
}
####
# Classificacio utilitzant el metode k-NN
####

##
# KNN, Dades Spam
##
class_knn_Spam <- function(k=3) {
	require(class)
	require(ElemStatLearn)
	#Prepara les  dades i els subconjunts trainig test
	n <- nrow(spam)
	ntrain <- ceiling(0.6*n)
	ntest <- n - ntrain
	Itrain <- sample(1:n,ntrain,replace=FALSE)
	spam.train <- spam[Itrain,]
	spam.test <- spam[-Itrain,]
	cl <- factor(spam.train$spam)
	spam.train <- spam.train[,-58]
	y<-spam.test$spam
	spam.test <- spam.test[,-58]
	#Classifica
	spam.pred <- knn(spam.train,spam.test,cl,k)
	#Matriu de confusio
	C<-table("True"=y,"Predicted"=spam.pred)
	print(C)
	print(sum(diag(C))/sum(C))
	return(C)
}

##
# KNN, Dades Vowel
##
class_knn_Vowel <- function(k=10) {
	require(class)
	require(ElemStatLearn)
	cl<-factor(vowel.train$y)
	Yreal<-vowel.test$y
	vowel.train <- vowel.train[,-1]
	vowel.test <- vowel.test[,-1]
	vowel.pred <- knn(vowel.train,vowel.test,cl,k)
	C<-table("True"=Yreal,"Predicted"=vowel.pred)
	print(C)
	print(sum(diag(C))/sum(C))
	return(C)
}





#################################################################################################################################
#################################################################################################################################
#################################################################################################################################

#
# Funcions utilitzades en el transcurs de la practica per trobar diferents valors de k, i fer diverses proves
#

# Realitza el knn diversos cops per trobar la millor k (o almenys, la millor entre les que es proven)
find_best_k_Spam <-function(){
	mitja_k3 <- 0
	mitja_k10 <- 0
	mitja_k30 <- 0
	mitja_k55 <- 0
	mitja_k100 <- 0
	mitja_k200 <- 0
	for (i in 1:50) {
		C<-class_knn_Spam(3)
		mitja_k3 <- mitja_k3 + sum(diag(C))/sum(C)
	}
	mitja_k3 <- mitja_k3/50.
	i=1
	for (i in 1:50) {
		C<-class_knn_Spam(10)
		mitja_k10 <- mitja_k10 + sum(diag(C))/sum(C)
	}
	mitja_k10 <- mitja_k10/50.
	for (i in 1:50) {
		C<-class_knn_Spam(30)
		mitja_k30 <- mitja_k30 + sum(diag(C))/sum(C)
	}
	mitja_k30 <- mitja_k30/50.
	for (i in 1:50) {
		C<-class_knn_Spam(55)
		mitja_k55 <- mitja_k55 + sum(diag(C))/sum(C)
	}
	mitja_k55 <- mitja_k55/50.
	for (i in 1:50) {
		C<-class_knn_Spam(100)
		mitja_k100 <- mitja_k100 + sum(diag(C))/sum(C)
	}
	mitja_k100 <- mitja_k100/50.
	for (i in 1:50) {
		C<-class_knn_Spam(200)
		mitja_k200 <- mitja_k200 + sum(diag(C))/sum(C)
	}
	mitja_k200 <- mitja_k200/50.
	m_kas <- c("k=3"=mitja_k3,"k=10"=mitja_k10,"k=30"=mitja_k30,"k=55"=mitja_k55,"k=100"=mitja_k100,"k=200"=mitja_k200)
	# 
	#      k=3      k=10      k=30      k=55     k=100     k=200 
	#  0.7935978 0.7737826 0.7486630 0.7389891 0.7249348 0.7126196 
	return(m_kas)
}
find_best_k_Vowel <-function(){
	mitja_k3 <- 0
	mitja_k10 <- 0
	mitja_k15 <- 0
	mitja_k20 <- 0
	mitja_k30 <- 0
	mitja_k55 <- 0
	mitja_k100 <- 0
	mitja_k200 <- 0
	for (i in 1:500) {
		C<-class_knn_Vowel(3)
		mitja_k3 <- mitja_k3 + sum(diag(C))/sum(C)
	}
	mitja_k3 <- mitja_k3/500.
	i=1
	for (i in 1:500) {
		C<-class_knn_Vowel(10)
		mitja_k10 <- mitja_k10 + sum(diag(C))/sum(C)
	}
	mitja_k10 <- mitja_k10/500.
	for (i in 1:500) {
		C<-class_knn_Vowel(15)
		mitja_k15 <- mitja_k15 + sum(diag(C))/sum(C)
	}
	mitja_k15 <- mitja_k15/500.
	for (i in 1:500) {
		C<-class_knn_Vowel(20)
		mitja_k20 <- mitja_k20 + sum(diag(C))/sum(C)
	}
	mitja_k20 <- mitja_k20/500.
	for (i in 1:500) {
		C<-class_knn_Vowel(30)
		mitja_k30 <- mitja_k30 + sum(diag(C))/sum(C)
	}
	mitja_k30 <- mitja_k30/500.
	for (i in 1:500) {
		C<-class_knn_Vowel(55)
		mitja_k55 <- mitja_k55 + sum(diag(C))/sum(C)
	}
	mitja_k55 <- mitja_k55/500.
	for (i in 1:500) {
		C<-class_knn_Vowel(100)
		mitja_k100 <- mitja_k100 + sum(diag(C))/sum(C)
	}
	mitja_k100 <- mitja_k100/500.
	for (i in 1:500) {
		C<-class_knn_Vowel(200)
		mitja_k200 <- mitja_k200 + sum(diag(C))/sum(C)
	}
	mitja_k200 <- mitja_k200/500.
	m_kas <- c("k=3"=mitja_k3,"k=10"=mitja_k10,"k=15"=mitja_k15,"k=20"=mitja_k20,"k=30"=mitja_k30,"k=55"=mitja_k55,"k=100"=mitja_k100,"k=200"=mitja_k200)
	# 
	#      k=3      k=10      k=30      k=55     k=100     k=200 
	#  0.7935978 0.7737826 0.7486630 0.7389891 0.7249348 0.7126196 
	return(m_kas)
}

### 
# Funcio per fer proves, crida 100 vegades class_lm_Spam() i fa la mitja de l'encert
###
call_repeatedly <- function(){
	suma<-0
	for(i in 1:100){
		conf<-class_lm_Spam()
		suma<-suma+sum(diag(conf))/sum(conf)
	}
	print(suma/100.)
}

### 
# Funcio que cride 100 vegades class_reg_log_Vowel i fa la mitja de l'encert
###
repeat_reg_vowel <- function() {
	suma<-0
	for (i in 1:100){
		C<-class_reg_log_Vowel()
		suma<-suma+sum(diag(C))/sum(C)
	}
	print("suma")
	print(suma/100.)
}


print("Minims quadrats SPAM")
class_lm_Spam()
print("Minims quadrats Vowel")
class_lm_Vowel()
print(" ")
print("-----------------------------")
print(" ")
print("Discriminador lineal de Fisher, SPAM")
class_fisher_Spam()
print("Discriminador lineal de Fisher, Vowel")
class_fisher_Vowel()
print(" ")
print("-----------------------------")
print(" ")
print("KNN, SPAM")
class_knn_Spam()
print("KNN, Vowel")
class_knn_Vowel()
print(" ")
print("-----------------------------")
print(" ")
print("Regressio logistica SPAM")
class_reg_log_Spam()
print("Regressio logistica Vowel")
class_reg_log_Vowel()


