require(gmodels, quietly = TRUE)
require(sva, quietly = TRUE)
require(preprocessCore, quietly = TRUE)
require(hash, quietly = TRUE)

# global map of tissues from file names to "real" tissue name.
keys   = c("bladder", "breast", "cervix", "colon", "liver",
           "prostate", "stomach", "thyroid", "uterus", "blca", "brca",
           "cesc", "coad", "lihc", "prad", "stad", "thca", "ucec",
           "lung", "luad", "lusc")
values = c("bladder", "breast", "cervix", "colon", "liver",
           "prostate", "stomach", "thyroid", "uterus", "bladder",
           "breast", "cervix", "colon", "liver", "prostate",
           "stomach", "thyroid", "uterus", "lung", "lung", "lung")
tissue.map = hash(keys=keys, values=values)

## arrange df vars by position
## 'vars' must be a named vector, e.g. c("var.name"=1)
ArrangeVars = function(data, vars){
  ## stop if not a data.frame (but should work for matrices as well)
  stopifnot(is.data.frame(data))
  
  ## sort out inputs
  data.nms = names(data)
  var.nr = length(data.nms)
  var.nms = names(vars)
  var.pos = vars
  ## sanity checks
  stopifnot(!any(duplicated(var.nms)), !any(duplicated(var.pos)))
  stopifnot(is.character(var.nms), is.numeric(var.pos))
  stopifnot(all(var.nms %in% data.nms))
  stopifnot(all(var.pos > 0), all(var.pos <= var.nr))
  
  ## prepare output
  out.vec = character(var.nr)
  out.vec[var.pos] = var.nms
  out.vec[-var.pos] = data.nms[!(data.nms %in% var.nms)]
  stopifnot(length(out.vec)==var.nr)
  
  ## re-arrange vars by position
  data = data[, out.vec]
  return(data)
}

## Calculate PCA on data to prep for input to NN.
DoPCA = function(data, pcs.store = 100) {
    pca.obj = fast.prcomp(t(data), center = FALSE, scale = FALSE)
    
    pcs.store = min(pcs.store, dim(pca.obj$x)[2])

    pca.scores = data.frame(pca.obj$x[, 1:pcs.store])
    pca.load = data.frame(pca.obj$rotation[, 1:pcs.store])
    percentage = (100*(pca.obj$sdev)^2 / sum(pca.obj$sdev^2))[1:pcs.store]

    return(list(pca.scores, pca.load, percentage))
}

DoQN = function(data) {
    data.rownames = rownames(data)
    data.colnames = colnames(data)
    data = normalize.quantiles(as.matrix(data))
    data = data.frame(data, row.names = data.rownames)
    colnames(data) = data.colnames

    return(data)
}

DoCombat = function(data) {
    meta = GetMetaDataFrame(data)

    batch = meta$study
    design = model.matrix(~1, data = meta)
    print(data)
    combat = ComBat(dat = data, batch = batch, mod = design, par.prior = TRUE)

    return(combat)
}

# Takes in pca data and adds columns specifying the source and tissue
# of the row.
AddStudyTissue = function(data) {
    study = factor(unlist(lapply(rownames(data), function(x) {
        unlist(strsplit(x, "\\.")[1][1])[1]
    })))
    tissue = factor(unlist(lapply(rownames(data), function(x) {
        unlist(strsplit(x, "\\.")[1][1])[2]
    })))

    data[, 'study'] = study
    data[, 'tissue'] = tissue
    data = ArrangeVars(data, c("study"=1, "tissue"=2))

    return(data)
}

GetMetaDataTable = function(data) {
    if ("study" %in% colnames(data) && "tissue" %in% colnames(data)) {
        return(table(data$study, data$tissue))
    }

    if (grepl("GTEX", colnames(data)[1]) || grepl("TCGA", colnames(data)[1])) {
        study = factor(unlist(lapply(colnames(data), function(x) {
            unlist(strsplit(x, "\\.")[1][1])[1]
        })))
        tissue = factor(unlist(lapply(colnames(data), function(x) {
            unlist(strsplit(x, "\\.")[1][1])[2]
        })))
    } else {
        study = factor(unlist(lapply(rownames(data), function(x) {
            unlist(strsplit(x, "\\.")[1][1])[1]
        })))
        tissue = factor(unlist(lapply(rownames(data), function(x) {
            unlist(strsplit(x, "\\.")[1][1])[2]
        })))
    }

    return(table(study, tissue))
}

GetMetaDataFrame = function(data) {
    if ("study" %in% colnames(data) && "tissue" %in% colnames(data)) {
        ## return(table(data$study, data$tissue))
        return(data.frame(study = data$study, tissue = data$tissue))
    }

    if (grepl("GTEX", colnames(data)[1]) || grepl("TCGA", colnames(data)[1])) {
        study = factor(unlist(lapply(colnames(data), function(x) {
            unlist(strsplit(x, "\\.")[1][1])[1]
        })))
        tissue = factor(unlist(lapply(colnames(data), function(x) {
            unlist(strsplit(x, "\\.")[1][1])[2]
        })))
    } else {
        study = factor(unlist(lapply(rownames(data), function(x) {
            unlist(strsplit(x, "\\.")[1][1])[1]
        })))
        tissue = factor(unlist(lapply(rownames(data), function(x) {
            unlist(strsplit(x, "\\.")[1][1])[2]
        })))
    }

    return(data.frame(study = study, tissue = tissue))
}

# Loads and merges RNA data from a list of file names and paths
FileMultiMerge = function(file.names, path) {
    ## path.names = lapply(file.names, function(file.name) {
    ##     paste(path, file.name, sep = "/")
    ## })
  
    data.list = lapply(file.names, function(file.name) {
        file.path = paste(path, file.name, sep = "/")

        s = strsplit(file.name, "-")[[1]]
        study = toupper(strsplit(s[4], "\\.")[[1]][1])
        tissue = tissue.map[[s[1]]]
        # The columns will be named by the source of the data
        # e.g. TCGA or GTEX followed by the tissue type.
        study.tissue = paste(study, tissue, sep = ".")
        ## TODO: change nrows so we read all the rows.
        d = read.table(file = file.path, header = TRUE, nrows = 1000)
        len = length(d)
        colnames(d)[3:len] = make.names(rep(c(study.tissue), len-2), unique = TRUE)

        return(d)
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
