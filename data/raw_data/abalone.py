import pandas

df = pandas.read_csv("abalone.data", header=None, usecols=range(1, 9))

f = open("../corr/abalone.csv", "w")
f.write(str(df.shape[0]) + "\n")
f.close()

df.corr(method="spearman").to_csv(
    "../corr/abalone.csv", header=False, index=False, mode="a"
)
