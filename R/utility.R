require(Matrix)
require(igraph)
require(gmodels)
require(ggplot2)
require(sva)
require(RANN)
require(reshape)
require(preprocessCore)
## require(tsne)


rnaObj <- setClass("rnaObj", slots =
                     c(data = "data.frame", tissue = "vector", study = "vector", counts = "data.frame", meta = "data.frame",
                       pca.scores = "data.frame", pca.load = "data.frame", pca.meta = "data.frame", percentage = "vector"))


setGeneric("initialize", function(object, min.expression = 0.01, keep.tissue = c(), log = FALSE, qn = FALSE, normalize = FALSE, center = TRUE, scale = TRUE) 
  standardGeneric("initialize"))
setMethod("initialize", "rnaObj",
          function(object, min.expression = 0.01, keep.tissue = c(), log = FALSE, qn = FALSE, normalize = FALSE, center = TRUE, scale = TRUE) {
            print("Initializing S4 object")
            tmp = object@data
            print(dim(tmp))
            
            # remove unwanted tissues
            if (length(keep.tissue) != 0) {
              print("removing unwanted tissues")
              cols.use = unlist(lapply(keep.tissue, function(x) colnames(tmp)[grep(x, colnames(tmp))]))
              tmp = tmp[, cols.use]
              print(dim(tmp))
              rm(cols.use)
            }
            
            # gene filtering
            # remove genes with row means less than min expression level
            print("removing genes below mean expression level")
            tmp = tmp[which(rowMeans(tmp[,]) > min.expression), ]
            print(dim(tmp))
            
            # if log then take log of data
            if (log) {
              print("applying log2 transformation")
              tmp = log2(tmp + 1)
            }
            
            # if normalize is true then do quantile normalization
            if (qn || normalize) {
              if (qn) {
                print("applying quantile normalization per gene")
                tmp.rownames = rownames(tmp)
                tmp.colnames = colnames(tmp)
                tmp = normalize.quantiles(as.matrix(tmp))
                tmp = data.frame(tmp, row.names = tmp.rownames)
                colnames(tmp) = tmp.colnames
                rm(tmp.rownames, tmp.colnames)
              } else {
                print("applying z-score")
                tmp = t(scale(t(tmp),  center = center, scale = scale))
              }
            }
            
            object@data = data.frame(tmp)
            rm(tmp)
            
            print("calculating tissue counts")
            object@study = factor(unlist(lapply(colnames(object@data), function(x) unlist(strsplit(x, "\\.")[1][1])[1])))
            object@tissue = factor(unlist(lapply(colnames(object@data), function(x) unlist(strsplit(x, "\\.")[1][1])[2])))
            object@meta = data.frame(study = object@study, tissue = object@tissue)
            
            study.table = table(object@study)
            tissue.table = table(object@tissue)
            
            total.gtex = study.table[1]
            total.tcga = study.table[2]
            
            gtex.tissues = colnames(object@data)[grep("GTEX", colnames(object@data))]
            gtex.tissues = lapply(gtex.tissues, function(x) unlist(strsplit(x, "\\.")[1][1]))
            
            tcga.tissues = colnames(object@data)[grep("TCGA", colnames(object@data))]
            tcga.tissues = lapply(tcga.tissues, function(x) unlist(strsplit(x, "\\.")[1][1]))
            
            count.gtex = as.vector(table(unlist(lapply(gtex.tissues, function (x) x[2]))))
            count.tcga = as.vector(table(unlist(lapply(tcga.tissues, function (x) x[2]))))
            
            unique.tissues = levels(object@tissue)
            
            count = data.frame(GTEX = count.gtex, TCGA = count.tcga, row.names = unique.tissues)
            total.row = data.frame(GTEX = total.gtex, TCGA = total.tcga, row.names = c("total"))
            count = rbind(count, total.row)
            
            object@counts = count
            rm(count, total.gtex, total.tcga, count.gtex, count.tcga, unique.tissues, total.row, study.table, tissue.table, gtex.tissues, tcga.tissues)
            
            return(object)
          })


setGeneric("doPCA", function(object, pcs.store = 100) standardGeneric("doPCA"))
setMethod("doPCA", "rnaObj",
          function(object, pcs.store=100) {
            data.use = object@data
            
            pca.obj = fast.prcomp(t(data.use), center = FALSE, scale = FALSE)
            
            object@pca.scores = data.frame(pca.obj$x[ ,1:pcs.store])
            object@pca.load = data.frame(pca.obj$rotation[ , 1:pcs.store])
            
            object@percentage = (100*(pca.obj$sdev)^2 / sum(pca.obj$sdev^2))[1:pcs.store]
            
            pca.meta = object@pca.scores
            pca.meta[, 'study'] = object@study
            pca.meta[, 'tissue'] = object@tissue
            
            pca.meta = arrange.vars(pca.meta, c("study"=1, "tissue"=2))
            object@pca.meta = pca.meta
            
            rm(pca.obj, pca.meta)

            return(object)
          })

setGeneric("scatterPlot", function(object, title = "") standardGeneric("scatterPlot"))
setMethod("scatterPlot", "rnaObj",
          function(object, title = "") {
            tissue = factor(object@tissue)
            study = factor(object@study)
            
            ggplot(object@pca.scores, aes(x = PC1, y = PC2, shape = study, colour = tissue)) + 
              geom_point(size = 2) + scale_color_hue(l = 55) + 
              theme_bw() + 
              guides(colour = guide_legend(order = 2), shape = guide_legend(order = 2)) +
              ggtitle(title) + 
              theme(plot.title = element_text(hjust = 0.5, lineheight=.8, face="bold"))
          })


##arrange df vars by position
##'vars' must be a named vector, e.g. c("var.name"=1)
arrange.vars = function(data, vars){
  ##stop if not a data.frame (but should work for matrices as well)
  stopifnot(is.data.frame(data))
  
  ##sort out inputs
  data.nms = names(data)
  var.nr = length(data.nms)
  var.nms = names(vars)
  var.pos = vars
  ##sanity checks
  stopifnot( !any(duplicated(var.nms)), 
             !any(duplicated(var.pos)) )
  stopifnot( is.character(var.nms), 
             is.numeric(var.pos) )
  stopifnot( all(var.nms %in% data.nms) )
  stopifnot( all(var.pos > 0), 
             all(var.pos <= var.nr) )
  
  ##prepare output
  out.vec = character(var.nr)
  out.vec[var.pos] = var.nms
  out.vec[-var.pos] = data.nms[ !(data.nms %in% var.nms) ]
  stopifnot( length(out.vec)==var.nr )
  
  ##re-arrange vars by position
  data = data[ , out.vec]
  return(data)
}

# Loads and merges RNA data from a list of file names and paths
FileMultiMerge = function(file.names, path) {
  tissueTypes = c("bladder", "breast", "prostate", "thyroid")
  path.names = lapply(file.names, function(file.name) {
    paste(path, file.name, sep = "/")
  })
  
  data.list = lapply(path.names, function(file.name) {
    read.table(file = file.name, header = TRUE)
  })
  
  data = Reduce(function(x, y) {
    merge(x, y, by=c('Hugo_Symbol', 'Entrez_Gene_Id'))
  }, data.list)
  
  return(data)
}


# Loads and merges RNA data from all files in given path
PathMultiMerge = function(path) {
  file.names = list.files(path = path, full.names = TRUE)
  data.list = lapply(file.names, function(x) {
    read.table(file = x, header = TRUE)
  })
  
  data = Reduce(function(x, y) {
    merge(x, y, by=c('Hugo_Symbol', 'Entrez_Gene_Id'))
  }, data.list)
  
  return(data)
}
