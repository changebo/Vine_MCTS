# correlation matrix of normal scores for DAX GARCH-filtered data
# daily 2011-2012, in dax1112gf.RData (CopulaModel/7chapter/08sect3/)

load("dax1112gf.RData")
# ls()
# [1] "dax1112gf" "label"

zdat = dax1112gf$zscore
colnames(zdat) = label
rmat = cor(zdat)
n = nrow(zdat) # 511
d = ncol(zdat) # 29

dim_vec <- c(8, 10, 15, 20)
n_sample <- 100

file.remove(Sys.glob("../corr/dax_d_*"))

for (subset_size in dim_vec) {
  set.seed(1)
  
  for (i in 1:n_sample) {
    filename <- paste0("../corr/dax_d_", subset_size, "_id_", i, ".csv")
    write(n, filename)
    
    ind <- sample(d, subset_size)
    write.table(
      rmat[ind, ind],
      filename,
      append = TRUE,
      quote = FALSE,
      row.names = FALSE,
      col.names = FALSE,
      sep = ", "
    )
  }
}
