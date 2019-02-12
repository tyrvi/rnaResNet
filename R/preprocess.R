## read.path = "~/ws/research/rnaResNet/data/raw/unnormalized"
## write.path = "~/ws/research/rnaResNet/data/interim"

## files.read = c("breast-rsem-fpkm-gtex.txt",
##                "prostate-rsem-fpkm-gtex.txt",
##                "thyroid-rsem-fpkm-gtex.txt",
##                "brca-rsem-fpkm-tcga.txt",
##                "prad-rsem-fpkm-tcga.txt",
##                "thca-rsem-fpkm-tcga.txt")

          
## files = c(GTEX = "<NAME OF GTEX OUTPUT FILE HERE>", ## <NAME OF GTEX OUTPUT FILE>
##           TCGA = "<NAME OF TCGA OUTPUT FILE HERE>") ## <NAME OF TCGA OUTPUT FILE>

require(yaml, quietly = TRUE)
source("utility.R")

PreprocessAndWrite = function(read.path, write.path, files.read, files.write, type = "log") {
    data = Preprocess(read.path, files.read, type = type)
    ## splits the data by study to unsplit do unsplit(data, data$study)
    ## where data$study is the factor of studies from the original complete
    ## data frame.
    data = split(data, data$study)

    ## write GTEX and TCGA data
    cat("Writing files...\n")
    print(paste(write.path, files, sep="/"))
    write.table(data$GTEX, file = paste(write.path, files.write[["GTEX"]], sep = "/"))
    write.table(data$TCGA, file = paste(write.path, files.write[["TCGA"]], sep = "/"))
    cat("Done.\n")
}


Preprocess = function(read.path, files.read, type = "log") {
    ## load data
    cat("Loading data...\n")
    print(paste(read.path, files.read, sep="/"))
    data = FileMultiMerge(files.read, read.path)
    cat("Done.\n")
    ## rename rows to Hugo_Symbol and remove the Hugo_Symbol and
    ## Entrez_Gene_Id columns
    rownames(data) = data[, 1]
    data[, 1:2] = NULL

    ## remove rows with low mean expression levels.
    cat("Removing tissues with low expression levels... ")
    data = data[which(rowMeans(data[,]) > 0.01), ]
    cat("Done.\n")

    cat("Log transforming data...")
    data = log2(data + 1)
    cat("Done.\n")

    if (identical(type, "qn")) {
        ## Do quantile normalization.
        cat("Applying quantile normalization per gene... ")
        data = DoQN(data)
        cat("Done.\n")
    }

    cat("Doing PCA... ")
    data = DoPCA(data)[[1]]
    ## Add study and tissue columns.
    data = AddStudyTissue(data)
    cat("Done.\n")

    return(data)
}

ProcessCombat = function(data) {
    cat("Doing ComBat... \n")
    data = DoCombat(data)
    cat("Done.\n")

    return(data)
}

ProcessAndWriteCombat = function(data) {
    data = ProcessCombat(data)
    data = split(as.data.frame(t(data)), GetMetaDataFrame(t(data))$study)

    ## write GTEX and TCGA data
    cat("Writing files...\n")
    print(paste(write.path, files, sep="/"))
    write.table(as.data.frame(t(data$GTEX)), file = paste(write.path, files.write[["GTEX"]], sep = "/"))
    write.table(as.data.frame(t(data$TCGA)), file = paste(write.path, files.write[["TCGA"]], sep = "/"))
    cat("Done.\n")
}

## functions with CamelCase, variables dot (.) separated.
args = commandArgs(trailingOnly = TRUE)

if (length(args) == 0) {
    cfg = yaml.load_file("config.yml")
} else {
    cfg = yaml.load_file(args[1])
}

read.path = cfg$input$path
files.read = cfg$input$files

write.path = cfg$output$path
files.write = c(
    GTEX = cfg$output$GTEX,
    TCGA = cfg$output$TCGA
)

if (cfg$write == TRUE) {
    PreprocessAndWrite(read.path, write.path, files.read, files.write, type = cfg$type)
} else {
    data = Preprocess(read.path, files.read, type = cfg$type)
}


