###########
## AAiMD
## Judit Domenech Miro
##########

##
# Funcio que carrega les llibreries que es necessiten per utilitzar les altres funcions
llibreries <- function() {
  require(ElemStatLearn)
  require(glmnet)
  require(Matrix)
  require(foreach)
  require(regsel)
  #data(concrete)
  require(rebmix)
  #data(adult)
  #Llibreries per fer cross-validation
  require(boot)
  require(DAAG)

}
##
# hold out per a les dades adult
##
hold_out_adult <- function() {
  #Agafem nomes les observacins completes
  # Es classifiquen en dos grups (guanyen >50k$ o no)
  adult <- adult_preparacio()
  #Subdividim en conjunt de train i de test seleccionant aleatoriament
  #60% de les dades pel train, 40% pel test
  n <- nrow(adult)
	ntrain <- ceiling(0.6*n)
	ntest <- n - ntrain
	Itrain <- sample(1:n,ntrain,replace=FALSE)
  adult.train <- adult[Itrain,]
  adult.test <- adult[-Itrain,]
  #Ajustem el model
  adult.logit1 <- glm(Income~.,data=adult.train,family=binomial)
  #Fem la prediccio
  adult.pred <- predict(adult.logit1,newdata<-adult.test,type="response")
  adult.pred.crisp<-1*(adult.pred>=0.5)
  #La matriu de confusio
  C<-table("True"=adult.test$Income,"Predicted"=adult.pred.crisp)
  print(C)
  print("Error: ")
  #Calculem l'error
	print(1-sum(diag(C))/sum(C))

}

##
# hold out per a les dades  concrete
hold_out_concrete <- function() {
  concrete <- dades_concrete()
  #preparem les dades com a data frame per poder seleccionar utilitzant $
  colnames(concrete)<-c("Cement", "BlastFurnaceSlag","FlyAsh","Water","Superplasticizer","CoarseAggregate","FineAggregate","Age","CompressiveStrength")
  concrete<-as.data.frame(concrete)
  # Separem les dades en train i test aleatoriament
  n <- nrow(concrete)
  ntrain <- ceiling(0.6*n)
	ntest <- n - ntrain
	Itrain <- sample(1:n,ntrain,replace=FALSE)
  concrete.train <- concrete[Itrain,]
  concrete.test <- concrete[-Itrain,]
  #Ajustem el model
  concrete.logit1 <- lm(CompressiveStrength~.,data=concrete.train)
  #Fem la prediccio
  concrete.pred <- predict(concrete.logit1,newdata<-concrete.test, type="response")
  print("MSE error:")
  #Calculem l'error
  print(mean(sqrt((concrete.pred-concrete.test$CompressiveStrength)**2)))

}

##
# K-fold cross-validation dades adult
##
k_fold_cross_adult <- function(k_=10) {
  # Preparem les dades
  adult <- adult_preparacio()
  # Per calcular el temps que tarda en processar
  ptm <- proc.time()
  # ajustem el model
  adult.glm <- glm(Income~.,data=adult,family=binomial)
  # Fa cross-validation, la funcio ja ajusta el model i el test per a cada iteracio
  cv.err.k <- cv.glm(adult,adult.glm,K=k_)$delta #$delta ens indica l'error
  print(proc.time() - ptm) #imprimim el temps
  print("Error (raw estimate) (adjusted estimate) ")
  print(cv.err.k)
}

##
# K-fold cross-validation per a les dades concrete
k_fold_cross_concrete <- function(k_=3) {
  #preparacio de les dades
  concrete <- dades_concrete()
  colnames(concrete)<-c("Cement", "BlastFurnaceSlag","FlyAsh","Water","Superplasticizer","CoarseAggregate","FineAggregate","Age","CompressiveStrength")
  concrete<-as.data.frame(concrete)
  # Ajusta el model i fa la cross-validation
  model <- lm(CompressiveStrength~.,data=concrete)
  cv.err.k <- CVlm(concrete,model,m=k_)
  ## Imprimim el mean square error per a que es vegi mes clar, ja que imprimeix moltes coses
  print("-- Mean Square: ")
  print(attr(cv.err.k,"ms"))

}

##
# Leave one out cross-validation concrete
# k-fold cross per k = num observacions
##
leave_one_out_concrete <- function() {
  concrete <- dades_concrete()
  n <- nrow(concrete)
  colnames(concrete)<-c("Cement", "BlastFurnaceSlag","FlyAsh","Water","Superplasticizer","CoarseAggregate","FineAggregate","Age","CompressiveStrength")
  concrete<-as.data.frame(concrete)
  model <- lm(CompressiveStrength~.,data=concrete)
  cv.err.k <- CVlm(concrete,model,m=n)
  print("-- Mean Square: ")
  print(attr(cv.err.k,"ms"))
}

##
# Leave one out cross-validation dades adult
# Com que es molt costos, s'aplica sobre un subconjunt
##
leave_one_out_adult <- function() {

  adult <- adult_preparacio()

  ptm <- proc.time()
  adult.glm <- glm(Income~.,data=adult,family=binomial)

  adult <- adult[1:100,] ####Tarda hores per processar-ho tot
  #Amb un subconjunt de 10000 observacions tarda gairebe dues hores
  cv.err.k <- cv.glm(adult,adult.glm)
  print(proc.time() - ptm)
#  print("Error")
  print(cv.err.k)
}

##
# Aplicacio de bootstrap per a les dades adult
##
bootstrap_adult <- function() {
  # Prepara les dades
  adult <- adult_preparacio()
  n <- nrow(adult)
  print(n)
  #Selecciona n observacions amb repeticio (n=num observacions)
  indexs <-sample(n,n,replace=TRUE)
  # Conjunt d'entrenament, selecciona els indexs obtinguts aleatoriament
  adult.X1 <-adult[indexs,]
  # Conjunt de test (out-of-bag), selecciona els indexs que no han estat seleccionats
  oob <- adult[-unique(indexs),]
  print("Num. Observacions del conjunt OOB")
  print(nrow(oob))
  #Ajusta el model
  adult.logit1 <- glm(Income~.,data=adult.X1,family=binomial)
  # APlica el model a les dades oob
  adult.pred <- predict(adult.logit1,newdata<-oob,type="response")
  # Fa la prediccio sobre les dades test
  adult.pred.crisp<-1*(adult.pred>=0.5)
  C<-table("True"=oob$Income,"Predicted"=adult.pred.crisp)
  print(C)
  print("Error Bootstrap: ")
  # Calcula l'error bootstrap
  E.boot <- 1-sum(diag(C))/sum(C)
	print(E.boot)
  # Fa la prediccio sobre el conjunt d'entrenament
  adult.pred <- predict(adult.logit1,newdata<-adult.X1,type="response")
  adult.pred.crisp<-1*(adult.pred>=0.5)
  C<-table("True"=adult.X1$Income,"Predicted"=adult.pred.crisp)
  # Calcula l'error E.subst
  E.subst <- 1-sum(diag(C))/sum(C)
  print("Error subst: ")
  print(E.subst)
  print("E.632: ")
  #Aplica la regla 0.632
  print(0.632*E.boot+0.368*E.subst)
}


bootstrap_concrete <- function() {
  concrete <- dades_concrete()
  colnames(concrete)<-c("Cement", "BlastFurnaceSlag","FlyAsh","Water","Superplasticizer","CoarseAggregate","FineAggregate","Age","CompressiveStrength")
  concrete<-as.data.frame(concrete)
  n <- nrow(concrete)
  print("Num Observacions Concrete")
  print(n)
  #Seleccionem els index de les files
  indexs <-sample(n,n,replace=TRUE)
  concrete.X1 <- concrete[indexs,]
  print(nrow(concrete.X1))
  # Seleccionem els indexs de les files que no s'han repetit abans
  oob <- concrete[-unique(indexs),]
  print("OOB row")
  print(nrow(oob))
  concrete.logit1 <- lm(CompressiveStrength~.,data=concrete.X1)
  concrete.pred <- predict(concrete.logit1,newdata<-oob, type="response")
  print("Bootstrap error:")
  E.boot <- mean(sqrt((concrete.pred-oob$CompressiveStrength)**2))
  print(E.boot)
  concrete.logit1 <- lm(CompressiveStrength~.,data=concrete.X1)
  concrete.pred <- predict(concrete.logit1,newdata<-concrete.X1, type="response")
  E.subst <- mean(sqrt((concrete.pred-concrete.X1$CompressiveStrength)**2))
  print("Error subst: ")
  print(E.subst)
  print("E.632: ")
  print(0.632*E.boot+0.368*E.subst)
}

#Prova per contabilitza el temps de fer el predict per una mostra (aprox. 2,30s)
prova <- function() {
  adult <- adult_preparacio()
  train <- adult[-1,]
  test <- adult[1,]
  ptm <- proc.time()
  spam.logit1 <-glm(Income~.,data=train,family=binomial)
  spam.pred<-predict(spam.logit1,newdata<-test,type="response")
  print(proc.time() - ptm)
  spam.pred.crisp<-1*(spam.pred>=0.5)
  #Calculem la matriu de confusio
  C<-table("True"= test$Income,"Predicted"=spam.pred.crisp)
  print(C)
  print(sum(diag(C))/sum(C))
}


# Prepara les dades concrete seguint el fitxer Cocnrete.r
dades_concrete <- function() {
  #problema de regressio de resposta CompressiveStrength
  data(concrete)
  conc.s<-scale(concrete)
  #str(conc.s)
  conc.s.df<-data.frame(conc.s)
  #str(conc.s.df)
  return(conc.s)
}

# Prepara les dades adult seguint el fitxer Adult.r
adult_preparacio <- function() {
  #
  #	Preparaci� de les dades 'adult'
  #
  require(rebmix)
  data(adult)
  adult <- adult[complete.cases(adult), ]
  #
  #
  #
  adult.0<-adult[,-c(1,15)]
  n<-nrow(adult.0)
  #
  #	Eliminem Education, per m�todes lm, glm, lda
  #
  adult.1<-adult.0[,-4]
  #
  #	Eliminem Education.Num, per naive.bayes
  #
  adult.2<-adult.0[,-5]
  #
  #   	Simplifiquem els nivells de Workclass
  #
  levels(adult.1$Workclass)
  #[1] "federal-gov"      "local-gov"        "never-worked"     "private"
  #[5] "self-emp-inc"     "self-emp-not-inc" "state-gov"        "without-pay"
  #
  #	Originalment hi ha 8 nivells. Ho deixem en 4
  #
  g<-adult.1$Workclass
  h<-rep("",length(g))
  h[g=="federal-gov"]<-"funcionari"
  h[g=="local-gov"]<-"funcionari"
  h[g=="state-gov"]<-"funcionari"
  h[g=="never-worked"]<-"aturat"
  h[g=="without-pay"]<-"aturat"
  h[g=="self-emp-inc"]<-"autonom"
  h[g=="self-emp-not-inc"]<-"autonom"
  h[g=="private"]<-"empleat"
  adult.1$Workclass<-as.factor(h)
  adult.2$Workclass<-as.factor(h)
  levels(adult.1$Workclass)
  #[1] "aturat"     "autonom"    "empleat"    "funcionari"
  #
  #	Eliminem la variable Fnlwgt. No queda clar el seu significat. En qualsevol
  #	cas, s'hauria de tractar com el par�metre Weight en lg() i glm(), no com
  #	un predictor.
  #
  adult.1<-adult.1[,-3]
  adult.2<-adult.2[,-3]
  #
  #	Simplifiquem la variable Marital.Status. Originalment hi ha 7 nivells.
  #	Ho deixarem en dos nivells.
  #
  levels(adult.1$Marital.Status)
  #[1] "divorced"              "married-af-spouse"     "married-civ-spouse"
  #[4] "married-spouse-absent" "never-married"         "separated"
  #[7] "widowed"
  #
  g<-adult.1$Marital.Status
  h<-rep("",length(g))
  h[g=="divorced"]<-"single"
  h[g=="married-af-spouse"]<-"married"
  h[g=="married-civ-spouse"]<-"married"
  h[g=="married-spouse-absent"]<-"married"
  h[g=="never-married"]<-"single"
  h[g=="separated"]<-"single"
  h[g=="widowed"]<-"single"
  #
  adult.1$Marital.Status<-as.factor(h)
  adult.2$Marital.Status<-as.factor(h)
  #
  #	Simplifiquem el predictor Occupation. Originalment t� 14 nivells.
  #	Ho deixem en 4 nivells.
  #
  g<-adult.1$Occupation
  levels(g)
  # [1] "adm-clerical"      "armed-forces"      "craft-repair"
  # [4] "exec-managerial"   "farming-fishing"   "handlers-cleaners"
  # [7] "machine-op-inspct" "other-service"     "priv-house-serv"
  #[10] "prof-specialty"    "protective-serv"   "sales"
  #[13] "tech-support"      "transport-moving"
  h<-rep("",length(g))
  h[g=="adm-clerical"]<-"A"
  h[g=="armed-forces"]<-"A"
  h[g=="craft-repair"]<-"B"
  h[g=="farming-fishing"]<-"B"
  h[g=="handlers-cleaners"]<-"B"
  h[g=="machine-op-inspct"]<-"B"
  h[g=="other-service"]<-"B"
  h[g=="priv-house-serv"]<-"B"
  h[g=="transport-moving"]<-"B"
  h[g=="prof-specialty"]<-"C"
  h[g=="protective-serv"]<-"C"
  h[g=="sales"]<-"C"
  h[g=="tech-support"]<-"C"
  h[g=="exec-managerial"]<-"D"
  adult.1$Occupation<-as.factor(h)
  adult.2$Occupation<-as.factor(h)
  #
  #	La variable Relationship la deixem sense canvis, de moment.
  #	Molt possiblement ser� una predictora pobre de la resposta i la descartarem
  #	en l'an�lisi m�s endavant.
  #
  #	Eliminem la variable Race (apart de la correcci� pol�tica). �s una variable
  #	molt no balancejada ~39000 blancs front a nombres molt petits d'altres.
  #
  #
  table(adult.1$Race)
  #
  #amer-indian-eskimo asian-pac-islander              black              other
  #               435               1303               4228                353
  #             white
  #             38903
  adult.1<-adult.1[,-7]
  adult.2<-adult.2[,-7]

  return(adult.1)
}


llibreries()
