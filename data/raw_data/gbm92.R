# From STAT 521 course project

load("gbm92.RData")

rmat = rmat.raw
n = nrow(gbmsub) # 558
d = ncol(gbmsub) # 92

dim_vec <- c(8, 10, 15, 20, 30, 50)

file.remove(Sys.glob("../corr/gbm_d_*"))

for (subset_size in dim_vec) {
  if (subset_size > 20) {
    n_sample <- 10
  } else {
    n_sample <- 100
  }

  set.seed(1)
  for (i in 1:n_sample) {
    filename <- paste0("../corr/gbm_d_", subset_size, "_id_", i, ".csv")
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
