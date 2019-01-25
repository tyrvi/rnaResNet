library(reshape)
library(gplots)
library(ops)
library(calibrate)
library(biomaRt)
library(sva)
library(ggplot2)
library(org.Hs.eg.db) # for transferring gene identifiers
library(data.table) # for collapsing transcript RPKMs
library(Rtsne)

# read gtex data
gtex <- read.delim("../data/GTEx-3-Tissues-N.txt", sep="\t")
gtexcolumns <- colnames(gtex)
gtex.fpkms <- gtex[,gtexcolumns[2:length(gtexcolumns)]]
colnames(gtex.fpkms)[1] <- "Gene"

# read tcga data
tcga <- read.delim("../data/TCGA-3-Tissues-N.txt", sep="\t")
tcgacolumns <- colnames(tcga)
tcga.genes <- tcga[,"Gene"]
tcga[,"Gene"] <- gsub("\\|.*", "", tcga.genes)
tcga.fpkms <- tcga

# merge gtex and tcga data
fpkms <- merge(gtex.fpkms, tcga.fpkms, by="Gene")

# remove duplicates
temp <- fpkms[!duplicated(fpkms[,1]),]
gene <- temp[,1]
f <- temp[,2:ncol(temp)]
rownames(f) <- gene

# remove values which are close to zero
f.nozero <- f[-which(rowMeans(f[,])<=0.01),]

# get counts of genes
f.gtex.bladder <- length(grep("GTEX.*bladder", colnames(f.nozero)))
f.gtex.prostate <- length(grep("GTEX.*prostate", colnames(f.nozero)))
f.gtex.thyroid <- length(grep("GTEX.*thyroid", colnames(f.nozero)))
f.tcga.bladder <- length(grep("TCGA.*bladder", colnames(f.nozero)))
f.tcga.prostate <- length(grep("TCGA.*prostate", colnames(f.nozero)))
f.tcga.thyroid <- length(grep("TCGA.*thyroid", colnames(f.nozero)))
f.gtex.total <- f.gtex.bladder + f.gtex.prostate + f.gtex.thyroid
f.tcga.total <- f.tcga.bladder + f.tcga.prostate + f.tcga.thyroid

# shapes in plots
gtex.shape <- 0
tcga.shape <- 19

# counts of samples
f.counts <- c(f.gtex.total, f.tcga.total, f.gtex.bladder, f.gtex.prostate, f.gtex.thyroid, f.tcga.bladder, f.tcga.prostate, f.tcga.thyroid)

# vector of colors
f.colors <- c(rep("indianred",f.counts[3]), rep("dodgerblue",f.counts[4]), rep("forestgreen",f.counts[5]),
              rep("indianred",f.counts[6]), rep("dodgerblue",f.counts[7]), rep("forestgreen",f.counts[8]))
# vector of shapes
shapes <- c(rep(gtex.shape,f.counts[1]),rep(tcga.shape,f.counts[2]))

annotation = data.frame(TissueType=factor(rep(c("bladder", "prostate", "thyroid", "bladder", "prostate", "thyroid"), 
                                              c(f.gtex.bladder, f.gtex.prostate, f.gtex.thyroid, f.tcga.bladder, f.tcga.prostate, f.tcga.thyroid))), 
                        Study=factor(rep(c("GTEX", "TCGA"), c(f.gtex.total, f.tcga.total))))
rownames(annotation) = colnames(f.nozero)

#tmp <- t(f.nozero[1:nrow(f.nozero),1:ncol(f.nozero)])

tsne_f <- Rtsne(t(f.nozero[1:nrow(f.nozero),1:ncol(f.nozero)]), initial_dims=100, perplexity=15, max_iter = 1500)

# figure 2 apply log transform = log(val) + 1
pseudo <- 1
f.log <- log2(f.nozero + pseudo)

tsne_f_log <- Rtsne(t(f.log[1:nrow(f.log),1:ncol(f.log)]), initial_dims=100, perplexity=15, max_iter = 1500)

# figure 3 apply ComBat
meta <- data.frame(study=c(rep("GTEx",f.counts[1]), rep("TCGA",f.counts[2])), tissue=c(rep("Bladder",f.counts[3]), rep("Prostate",f.counts[4]), rep("Thyroid",f.counts[5]), rep("Bladder",f.counts[6]), rep("Prostate",f.counts[7]), rep("Thyroid",f.counts[8])))
batch <- meta$study
design <- model.matrix(~1, data=meta)
combat <- ComBat(dat=f.log, batch=batch, mod=design, par.prior=TRUE)

tsne_f_combat <- Rtsne(t(combat[1:nrow(combat),1:ncol(combat)]), initial_dims=70, perplexity=40, max_iter = 1500)

# plot before any modifications
plot(tsne_f$Y, pch=shapes, cex=0.7, cex.lab=1, col=f.colors, main="T-SNE no processing")
legend("bottomright", legend=c("Bladder","Prostate","Thyroid"),col=c("indianred", "dodgerblue", "forestgreen"),cex=1,pch=20,ncol=(3),bty="n")
legend("bottomleft",legend=c("GTEx","TCGA"),col="black",pch=c(gtex.shape,tcga.shape),ncol=2, bty="n")

# plot after log transform
plot(tsne_f_log$Y, pch=shapes, cex=0.7, cex.lab=1, col=f.colors, main="T-SNE log transformation")
legend("bottomright", legend=c("Bladder","Prostate","Thyroid"),col=c("indianred", "dodgerblue", "forestgreen"),cex=1,pch=20,ncol=(3),bty="n")
legend("bottomleft",legend=c("GTEx","TCGA"),col="black",pch=c(gtex.shape,tcga.shape),ncol=2, bty="n")

# plot after combat
plot(tsne_f_combat$Y, pch=shapes, cex=0.7, cex.lab=1, col=f.colors, main="T-SNE ComBat")
legend("bottomright", legend=c("Bladder","Prostate","Thyroid"),col=c("indianred", "dodgerblue", "forestgreen"),cex=1,pch=20,ncol=(3),bty="n")
legend("bottomleft",legend=c("GTEx","TCGA"),col="black",pch=c(gtex.shape,tcga.shape),ncol=2, bty="n")


# quantile normalized data
data <- read.delim("../data/combined-DataMatrix-QuantNormed.txt", sep = "\t")
datacol <- colnames(data)
# remove breast samples
data[,datacol[grep(".*breast", colnames(data))]] <- list(NULL)
data.genes <- data[,"Gene"]

qn <- data[,2:length(data)]
rownames(qn) <- data.genes
qn.nozero <- qn


qn.gtex.bladder <- length(grep("GTEX.*bladder", colnames(qn.nozero)))
qn.gtex.prostate <- length(grep("GTEX.*prostate", colnames(qn.nozero)))
qn.gtex.thyroid <- length(grep("GTEX.*thyroid", colnames(qn.nozero)))
qn.tcga.bladder <- length(grep("TCGA.*bladder", colnames(qn.nozero)))
qn.tcga.prostate <- length(grep("TCGA.*prostate", colnames(qn.nozero)))
qn.tcga.thyroid <- length(grep("TCGA.*thyroid", colnames(qn.nozero)))
qn.gtex.total <- qn.gtex.bladder + qn.gtex.prostate + qn.gtex.thyroid
qn.tcga.total <- qn.tcga.bladder + qn.tcga.prostate + qn.tcga.thyroid

qn.counts <- c(qn.gtex.total, qn.tcga.total, qn.gtex.bladder, qn.gtex.prostate, qn.gtex.thyroid, qn.tcga.bladder, qn.tcga.prostate, qn.tcga.thyroid)
qn.colors <- c(rep("indianred",qn.counts[3]), rep("dodgerblue",qn.counts[4]), rep("forestgreen",qn.counts[5]),
               rep("indianred",qn.counts[6]), rep("dodgerblue",qn.counts[7]), rep("forestgreen",qn.counts[8]))


tsne_qn <- Rtsne(t(qn.nozero[1:nrow(qn.nozero),1:ncol(qn.nozero)]))

pseudo <- 1
qn.log <- log2(qn.nozero + pseudo)

tsne_qn_log <- Rtsne(t(qn.log[1:nrow(qn.log),1:ncol(qn.log)]))

qn.combat <- qn.log[-which(rowVars(qn.log[,])==0),]

meta <- data.frame(study=c(rep("GTEx",qn.counts[1]), rep("TCGA",qn.counts[2])), tissue=c(rep("Bladder",qn.counts[3]), rep("Prostate",qn.counts[4]), rep("Thyroid",qn.counts[5]), rep("Bladder",qn.counts[6]), rep("Prostate",qn.counts[7]), rep("Thyroid",qn.counts[8])))
batch <- meta$study
design <- model.matrix(~1, data=meta)
qn_combat <- ComBat(dat=qn.combat, batch=batch, mod=design, par.prior=TRUE)

tsne_qn_combat <- Rtsne(t(qn_combat[1:nrow(qn_combat),1:ncol(qn_combat)]))

# plot before any modifications
plot(tsne_qn$Y, pch=shapes, cex=0.7, cex.lab=1, col=qn.colors, main="T-SNE QN no processing")
legend("bottomright", legend=c("Bladder","Prostate","Thyroid"),col=c("indianred", "dodgerblue", "forestgreen"),cex=1,pch=20,ncol=(3),bty="n")
legend("bottomleft",legend=c("GTEx","TCGA"),col="black",pch=c(gtex.shape,tcga.shape),ncol=2, bty="n")

# plot after log transform
plot(tsne_qn_log$Y, pch=shapes, cex=0.7, cex.lab=1, col=qn.colors, main="T-SNE QN log transformation")
legend("bottomright", legend=c("Bladder","Prostate","Thyroid"),col=c("indianred", "dodgerblue", "forestgreen"),cex=1,pch=20,ncol=(3),bty="n")
legend("bottomleft",legend=c("GTEx","TCGA"),col="black",pch=c(gtex.shape,tcga.shape),ncol=2, bty="n")

# plot after combat
plot(tsne_qn_combat$Y, pch=shapes, cex=0.7, cex.lab=1, col=qn.colors, main="T-SNE QN ComBat")
legend("bottomright", legend=c("Bladder","Prostate","Thyroid"),col=c("indianred", "dodgerblue", "forestgreen"),cex=1,pch=20,ncol=(3),bty="n")
legend("bottomleft",legend=c("GTEx","TCGA"),col="black",pch=c(gtex.shape,tcga.shape),ncol=2, bty="n")