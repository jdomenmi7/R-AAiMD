### AAIMD: P1. KNN
## Judit Domenech Miro

## Calculates eucledean distance
distan<-function(a,b){
	d<-sqrt(sum((a-b)^2))
	return(d)
}

###
## X <- Matrix with predictors of the training set
## y <- Known training set classification (groups)
## k <- integer. Number of neighbours
## x <- Predictors of the new item 
## Returns: The group to which the item belongs
knnclass<-function(X,y,k,x){
	#A vector of distances between each row of X and x 
	xdist <- c()
	
	#we calculate the euclidean distance between  for each row of predictors (X) and the predictors of the new item (x)
	nrows <- nrow(X)
	for(i in 1:nrows){
		xdist[i] <- distan(x,X[i,])
	}

	#Sorting the distance list and getting the index
	xsort <-(sort.int(xdist, index.return=TRUE))
	
	# We obtain the groups of the k-nearest neighbours
	groups<-y[xsort$ix[1:k]]

	## We obtain the most repeated group
	y1<-as.integer(names(sort(table(groups),decreasing=TRUE)[1]))
	return(y1) 	
}


###
## X <- Matrix with predictors of the training set
## y <- Known training set classification (groups)
## k <- integer. Number of neighbours
## Xt <- Matrix with predictors of the test set
## yt <- Known test set classification (groups)
## Returns: Confusion matrix and probability of error estimation
knnclass.crossval<-function(X,y,k,Xt,yt) {
	# We create a 10x10 martix of 0s
	C<-matrix(0, nrow =10, ncol=10)
	
	## The row of the matrix is the predicted value, and the column the real value
	## for each predictor of the test set, we use knn to obtain the predicted group
	## we go to the position of the matrix C[predicted-group,real-group] and add 1
	numrows <- nrow(Xt)
	for (i in 1:numrows){
		y.val<-knnclass(X,y,k,Xt[i,])
		C[y.val+1,yt[i]+1] <- C[y.val+1,yt[i]+1] + 1
	}
	
	#The matrix' diagonal gives the elements correctly classificated,
	est_correct = sum(diag(C)) 
	
	#prob_error=1-prob_encert
	prob_error = 1 - est_correct/numrows
	
	#Set the nameof the rows and columns
	rownames(C) <- paste("ypred",c(0:9))
	colnames(C) <- paste("y",c(0:9))
	retorna=list("matriu confusio: " = C,"probabilitat error:"= prob_error)
	return(retorna)
}

###
## X <- Matrix with predictors of the training set
## y <- Known training set classification (groups)
## k <- integer. Number of neighbours
## x <- Predictors of the new item 
## Returns: predicted y for the new item
knnreg<-function(X,y,k,x){
	#A vector of distances between each row of X and x 
	xdist <- c()
	
	#we calculate the euclidean distance between  for each row of predictors (X) and the predictors of the new item (x)
	nrows <- nrow(X)
	for(i in 1:nrows){
		xdist[i] <- distan(x,X[i,])
	}

	#Sorting the distance list and getting the index
	xsort <-(sort.int(xdist, index.return=TRUE))
	
	# We obtain the groups of the k-nearest neighbours
	grups<-y[xsort$ix[1:k]]

	# Returns the average of the k-nearest neighbours
	y.pred<-mean(grups)
	return(y.pred)
}

###
## X <- Matrix with predictors of the training set
## y <- Known training set classification (groups)
## k <- integer. Number of neighbours
## Xt <- Matrix with predictors of the test set
## yt <- Known test set classification (groups)
## Returns: Mean square error
knnreg.crossval<-function(X,y,k,Xt,yt){
	### Calculates the sum of squared difference between preddicted y and real y
	er.quad <- 0
	numrows <- nrow(Xt)
	for (i in 1:numrows){
		y.val<-knnreg(X,y,k,Xt[i,])
		er.quad <- er.quad + (y.val - yt[i])^2

	}
	##REturns mean
	return(er.quad/numrows)
}

##############
### Functions to prepare the data and test the knn-functions
test_knn_crossval <- function(){
	## Train
	library('ElemStatLearn')
	data(zip.train)
	X0 <- zip.train[]
	X <- X0[1:800,2:257]
	y <- X0[1:800,1]
	data(zip.test)
	X.test <- zip.test[1:500,2:257]
	y.test <- zip.test[1:500,1]
	
	return(knnclass.crossval(X,y,15,X.test,y.test))
}


test_knn_crossval_reg <- function(){
	## Train
	library('ElemStatLearn')
	data(zip.train)
	X0 <- zip.train[]
	X <- X0[1:800,2:257]
	y <- X0[1:800,1]
	data(zip.test)
	X.test <- zip.test[1:500,2:257]
	y.test <- zip.test[1:500,1]
	
	return(knnreg.crossval(X,y,15,X.test,y.test))
}


#### Crida a funcions
print("Conjunt entrenament: 800 files  conjunt test: 500 files")
print("Knn classificacio: ")
clasi <- test_knn_crossval()
print(clasi)
print("Knn reg:")
regre <- test_knn_crossval_reg()
print(regre)