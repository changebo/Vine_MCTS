library(CopulaModel)

d <- 15
n <- 500
n_sample <- 30

# ar2acf.phi=function(ph1,ph2,lagmax) {
# 	if(ph1+ph2>=1 | ph2-ph1>=1 | ph2>=1 | ph2<= -1) { return(0) }
# 	r1=ph1/(1-ph2)
# 	rv=rep(0,lagmax)
# 	# yule-walker equations
# 	rv[1]=r1
# 	rv[2]=ph1*r1+ph2
# 	for (k in 3:lagmax) { rv[k]=ph1*rv[k-1]+ph2*rv[k-2] }
# 	#cat(rv,"\n")
# 	toeplitz(c(1,rv))
# }

ar2_corr <- function(d) {
	D <- Dvinearray(d)
	M <- matrix(0, d, d)
	M[2:d, 1] <- runif(d-1, 0.2, 0.3)
	M[3:d, 2] <- runif(d-2, 0.5, 0.6)
	reg2cor(M, D)
}

# Exact AR2 correlation matrix
for (i in 1:n_sample) {
	set.seed(i)

	rmat <- ar2_corr(d)

    filename <- paste0("../corr/simAR2_d_", d, "_id_", i, ".csv")
    write(n, filename)
	write.table(
	  format(rmat, digits = 4),
	  filename,
	  append = TRUE,
	  quote = FALSE,
	  row.names = FALSE,
	  col.names = FALSE,
	  sep = ", "
	)

}

# Perturbed correlation matrix
for (i in 1:n_sample) {
	set.seed(i + 1)

	rmat <- ar2_corr(d)
	rmat_chol <- chol(rmat)
	z <- matrix(rnorm(n*d),n,d)
	y <- z %*% rmat_chol
	rmat_sample <- cor(y)

    filename <- paste0("../corr/simPerturbAR2_d_", d, "_id_", i, ".csv")
    write(n, filename)

	write.table(
	  format(rmat_sample, digits = 4),
	  filename,
	  append = TRUE,
	  quote = FALSE,
	  row.names = FALSE,
	  col.names = FALSE,
	  sep = ", "
	)
}