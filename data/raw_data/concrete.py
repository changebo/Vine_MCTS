import pandas

df = pandas.read_csv("concrete.csv", header=0, usecols=range(0, 8))

f = open("../corr/concrete.csv", "w")
f.write(str(df.shape[0] - 1) + "\n")
f.close()

df.corr(method="spearman").to_csv(
    "../corr/concrete.csv", header=False, index=False, mode="a"
)
