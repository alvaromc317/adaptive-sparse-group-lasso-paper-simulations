# CODE FOR USING SPLS IN R ######################################################

rm(list=ls())
gc()

install_github("matt-sutton/bigsgPLS", host = "https://api.github.com")

if(!require(pacman)){install.packages("pacman"); library(pacman)}
pacman::p_load(devtools, mlbench, spls, sgPLS)

library(bigsgPLS)
library(sgPLS)
library(spls)
setwd("D:/Documentos_2/GoogleDrive/Doctorado/Tesis/Codigo/Python/Quantile_regression/analysis/quantile_asgl_paper_simulations/sim_625_sparse_t_error_spls")


# EJEMPLO DE USO ################################################################

data("BostonHousing")
x = as.matrix(BostonHousing[, -c(2, 4, 9, 10, 14)])
y = as.matrix(BostonHousing[,14], ncol=1)
y2 = BostonHousing[,14]
total_variance_in_x = sum(diag(var(x)))
n= dim(x)[1]

# PAQUETE MIXOMICS: MAL##########################################################

mixomics_model = spls(X=x,Y=y, mode = 'regression', ncomp=9)

plotIndiv(mixomics_model)
plotVar(mixomics_model)
ex = explained_variance(mixomics_model$X, mixomics_model$variates$X, ncomp=10)
cumsum(ex)

# Este paquete es una chapuza que no permite estimar los valores de los parametros
# y que proporciona una explicacion de la variabilidad explicada por las componentes pls
# pero no respecto a la matriz original x sino a las propias componentes de forma que
# con tantas componentes como variables, recuperas el 100% de la variabilidad, lo que es falso.

# PAQUETE sgPLS: MAL##########################################################

index = c(1, 3, 6, 9)
ncomp=2

model.sPLS <- sPLS(X=x, Y=y, ncomp=ncomp, mode="regression")
# loadings = loadings
p_sgpls = model.sPLS$loadings$X
# Scores = variates
t_sgpls = model.sPLS$variates$X

betahat = predict(object=model.sPLS, newdata=x[1:5,])$B.hat[,,1]

model.sgPLS <- sgPLS(X=x, Y=y, ncomp=ncomp, ind.block.x = index, 
                     keepX= rep(ncol(x), ncomp), alpha.x=0.1)
model.sgPLS$loadings$X

# No se como controlar la sparsidad de las soluciones

# PAQUETE bigsgPLS ##############################################################

# Ver paquete sgPLS (en cran) para detalles de la funcion. El paquete big esta
# hecho por los mismos pero adaptado al big data

index = c(1, 3, 6, 9, 12, 13)
spls_1 = bigsgpls(X=x, Y=y, regularised="none", H=5, alpha.x=0.5, case=4, ind.block.x = index)

# loadings = loadings
p_sgpls = spls_1$loadings$X
# Scores = variates
t_sgpls = spls_1$variates$X

pls_1 = bigsgpls(X=x, Y=y, regularised="none", H=5, case=4)
p_pls = pls_1$loadings$X
t_pls = pls_1$variates$X

sum(abs(p_sgpls-p_pls)>1e-3)
sum(abs(t_sgpls-t_pls)>1e-3)
# PAQUETE spls ##################################################################

# El parametro eta (entre 0 y 1) controla el nivel de sparsidad.
# El parametro k es el numero de componentes pls

spls_1 = spls(x=x, y=y2, eta =0.9, K = 2)
spls_2 = spls(x=x, y=y2, eta =0.99, K = 9)

p_spls1 = spls_1$projection
p_spls2 = spls_2$projection

sum(abs(p_spls1 - p_spls2) > 1e-3)

spls_1$betahat

# Porque si le pongo k=9 las componentes dejan de ser sparse? Porque el beta se hace sparse?

##################################################################################

# Usaremos paquete spls para estimar los 

for(index in 0:49)
  {
  x = as.matrix(read.table(file=glue('sim_625_sparse_t_error_spls_data/', index, '_x.txt'), header = FALSE, sep = ",", dec = "."))
  y = as.matrix(read.table(file=glue('sim_625_sparse_t_error_spls_data/', index, '_y.txt'), header = FALSE, sep = ",", dec = "."))
  train_index = as.vector(as.matrix(read.table(file=glue('sim_625_sparse_t_error_spls_data/', index, '_train_index.txt'), header = FALSE, sep = ",", dec = "."))) + 1
  
  x_train = x[train_index,]
  y_train = y[train_index,]
  
  spls_1 = spls(x=x_train, y=y_train, eta =0.7, K = 30)
  
  tmp_weight = spls_1$betahat
  write.table(x=tmp_weight, file = glue('sim_625_sparse_t_error_spls_data/', index, '_tmp_weight.txt'),row.names=F, col.names=F)
  }



