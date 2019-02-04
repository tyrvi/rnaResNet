library(hash)
source("utility.R")

# functions with CamelCase, variables dot (.) separated.

read.path = "~/ws/research/rnaResNet/data/raw/unnormalized"

file.names = c("breast-rsem-fpkm-gtex.txt",
               "prostate-rsem-fpkm-gtex.txt",
               "thyroid-rsem-fpkm-gtex.txt",
               "brca-rsem-fpkm-tcga.txt",
               "prad-rsem-fpkm-tcga.txt",
               "thca-rsem-fpkm-tcga.txt")

# load data
data = FileMultiMerge(file.names, data.path)

# rename rows to Hugo_Symbol and remove the Hugo_Symbol and
# Entrez_Gene_Id columns
rownames(data) = data[, 1]
data[, 1:2] = NULL

# remove rows with low mean expression levels.
data = data[which(rowMeans(data[,]) > 0.01), ]
data = log2(data + 1)
data = DoPCA(data)[[1]]
# Add study and tissue columns.
data = AddStudyTissue(data)
# splits the data by study to unsplit do unsplit(data, data$study)
# where data$study is the factor of studies from the original complete
# data frame.
data = split(data, data$study)

# write GTEX and TCGA data
write.path = "~/ws/research/rnaResNet/data/interim"
files = c(GTEX = "<NAME OF GTEX FILE HERE>", # <NAME OF GTEX FILE HERE>
          TCGA = "<NAME OF TCGA FILE HERE>") # <NAME OF TCGA FILE HERE>
write.table(data$GTEX, file = paste(write.path, files[["GTEX"]], sep = "/"))
write.table(data$TCGA, file = paste(write.path, files[["TCGA"]], sep = "/"))

