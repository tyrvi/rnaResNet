library(hash)
source("utility.R")

# functions with CamelCase, variables dot (.) separated.

keys   = c("bladder", "breast", "cervix", "colon", "liver", "prostate", "stomach",
           "thyroid", "uterus", "blca",    "brca",   "cesc",   "coad",  "lihc",
           "prad",     "stad",    "thca",    "ucec",   "lung", "luad", "lusc")
values = c("bladder", "breast", "cervix", "colon", "liver", "prostate", "stomach",
           "thyroid", "uterus", "bladder", "breast", "cervix", "colon", "liver",
           "prostate", "stomach", "thyroid", "uterus", "lung", "lung", "lung")
tissue.map = hash(keys=keys, values=values)

data.path = "~/ws/research/rnaResNet/data/raw/unnormalized"

file.names = c("breast-rsem-fpkm-gtex.txt", "prostate-rsem-fpkm-gtex.txt", "thyroid-rsem-fpkm-gtex.txt", "brca-rsem-fpkm-tcga.txt", "prad-rsem-fpkm-tcga.txt", "thca-rsem-fpkm-tcga.txt")

# load data
data = FileMultiMerge(file.names, data.path)

# rename rows to Hugo_Symbol and remove the Hugo_Symbol and
# Entrez_Gene_Id columns
rownames(data) = data[, 1]
data[, 1:2] = NULL

# remove rows with low mean expression levels.
data = data[which(rowMeans(data[,]) > 0.01), ]

data.log = log2(data + 1)
