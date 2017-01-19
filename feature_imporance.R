require(xgboost)
require(methods)

train = read.csv('train.csv',header=TRUE,stringsAsFactors = F)
test = read.csv('test.csv',header=TRUE,stringsAsFactors = F)
train = train[,-1]
test = test[,-1]

y = train[,ncol(train)]
y = gsub('Class_','',y)
y = as.integer(y)-1 #xgboost take features in [0,numOfClass)

x = rbind(train[,-ncol(train)],test)
x = as.matrix(x)
x = matrix(as.numeric(x),nrow(x),ncol(x))
trind = 1:length(y)
teind = (nrow(train)+1):nrow(x)

# Set necessary parameter
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 9,
              "nthread" = 8)

# Run Cross Valication
cv.nround = 50
bst.cv = xgb.cv(param=param, data = x[trind,], label = y, 
                nfold = 3, nrounds=cv.nround)

# Train the model
nround = 50
bst = xgboost(param=param, data = x[trind,], label = y, nrounds=nround)

xgb.plot.tree( model = bst, n_first_tree = 1)

# Compute feature importance matrix
importance_matrix <- xgb.importance( model = bst)

# Nice graph
xgb.plot.importance(importance_matrix[1:10,])



