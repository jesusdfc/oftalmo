library(ggplot2)
library(gridExtra)
args = commandArgs(trailingOnly=TRUE)


#LOAD THE DATASET
method <- args[1]
message('Loading file with method ',method)
excel_df <- read.table(file=paste0(getwd(),'/best_conf/nested_cv/',method,'_best_conf_6_6.tsv'), sep='\t', header = TRUE, row.names = NULL)

#--------------------> FIRST TABLE, TEST BALANCED ACCURACY AS A FUNCTION OF THE FILE NAME (VARS) <------------------------#
message('Generating first figure')
table_1_df <- aggregate(excel_df[['test_balacc']],excel_df['file_name'],mean)
colnames(table_1_df) <- c('Variables','Mean Test Balanced Accuracy')
table_1_df['Max Test Balanced Accuracy'] <- list('Max Test Balanced Accuracy' = aggregate(excel_df[['test_balacc']],excel_df['file_name'],max)$x)
table_1_df['Min Test Balanced Accuracy'] <- list('Min Test Balanced Accuracy' = aggregate(excel_df[['test_balacc']],excel_df['file_name'],min)$x)

table_1_df[['Variables']] <- sapply(table_1_df[['Variables']],function(x) substr(as.character(x),1,nchar(as.character(x))-7))
#Reduce it to only contain the variables that have not been dropped
table_1_df[['Variables']] <- sapply(table_1_df[['Variables']],function(x){
  
  split_x <- stringr::str_split(x,'_')[[1]]
  
  new_x <- c()
  for (i in seq_along(split_x)){
    
    if (i%%2){
      
      if (split_x[i+1]=='False'){
        new_x <- c(new_x,split_x[i])
      }
      
    }
    
  }
  
  return(paste0(new_x,collapse='_'))
  
})

table_1_df <- table_1_df[order(table_1_df$'Max Test Balanced Accuracy',decreasing=TRUE),]

#pdf(paste0(getwd(),'/figures/',method,'/table_testbalacc_filename.pdf'),height=40,width=15)
#grid.table(as.data.frame(table_1_df),rows=NULL)
#dev.off()


#----------------------->SECOND TABLE, SAME BUT ALSO GROUPING BY PREDICTION METHOD<---------------------------#
message('Generating second figure')
table_2_df <- aggregate(excel_df[['test_balacc']],excel_df[c('file_name','method')],mean)
colnames(table_2_df) <- c('Variables','Method','Mean Test Balanced Accuracy')
table_2_df['Max Test Balanced Accuracy'] <- list('Max Test Balanced Accuracy' = aggregate(excel_df[['test_balacc']], excel_df[c('file_name','method')],max)$x)
table_2_df['Min Test Balanced Accuracy'] <- list('Min Test Balanced Accuracy' = aggregate(excel_df[['test_balacc']], excel_df[c('file_name','method')],min)$x)

table_2_df[['Variables']] <- sapply(table_2_df[['Variables']],function(x) substr(as.character(x),1,nchar(as.character(x))-7))
#Reduce it to only contain the variables that have not been dropped
table_2_df[['Variables']] <- sapply(table_2_df[['Variables']],function(x){
  
  split_x <- stringr::str_split(x,'_')[[1]]
  
  new_x <- c()
  for (i in seq_along(split_x)){
    
    if (i%%2){
      
      if (split_x[i+1]=='False'){
        new_x <- c(new_x,split_x[i])
      }
      
    }
    
  }
  
  return(paste0(new_x,collapse='_'))
  
})

table_2_df <- table_2_df[order(table_2_df$'Max Test Balanced Accuracy',decreasing=TRUE),]

#pdf(paste0(getwd(),'/figures/',method,'/table_testbalacc_filename_method.pdf'),height=120,width=15)
#grid.table(as.data.frame(table_2_df),rows=NULL)
#dev.off()


#------------------------->THIRD TABLE, SAME BUT ALSO GROUPING BY PREDICTION METHOD + Test Area<------------------------#
message('Generating third figure')
table_3_df <- aggregate(excel_df[['test_balacc']],excel_df[c('file_name','method')],mean)
colnames(table_3_df) <- c('Variables','Method','Mean Test Balanced Accuracy')
table_3_df['Max Test Balanced Accuracy'] <- list('Max Test Balanced Accuracy' = aggregate(excel_df[['test_balacc']],
                                                                                          excel_df[c('file_name','method')],max)$x)
table_3_df['Min Test Balanced Accuracy'] <- list('Min Test Balanced Accuracy' = aggregate(excel_df[['test_balacc']],
                                                                                          excel_df[c('file_name','method')],min)$x)
#ADD AREA ACC
table_3_df['Mean Test Area Accuracy'] <- list('Mean Test Area Accuracy' = aggregate(excel_df[['test_area_acc']],
                                                                                          excel_df[c('file_name','method')],mean)$x)
table_3_df['Min Test Area Accuracy'] <- list('Min Test Area Accuracy' = aggregate(excel_df[['test_area_acc']],
                                                                                          excel_df[c('file_name','method')],min)$x)
table_3_df['Max Test Area Accuracy'] <- list('Max Test Area Accuracy' = aggregate(excel_df[['test_area_acc']],
                                                                                          excel_df[c('file_name','method')],max)$x)
#ADD REJECTION RATE
table_3_df['Mean Test Area Rejection Rate'] <- list('Mean Test Area Rejection Rate' = aggregate(excel_df[['test_area_reject_rate']],
                                                                                          excel_df[c('file_name','method')],mean)$x)
table_3_df['Min Test Area Rejection Rate'] <- list('Min Test Area Rejection Rate' = aggregate(excel_df[['test_area_reject_rate']],
                                                                                          excel_df[c('file_name','method')],min)$x)
table_3_df['Max Test Area Rejection Rate'] <- list('Max Test Area Rejection Rate' = aggregate(excel_df[['test_area_reject_rate']],
                                                                                          excel_df[c('file_name','method')],max)$x)

#ADD SCORE
table_3_df['Max Score'] <- list('Max Score' = aggregate(excel_df[['score']], excel_df[c('file_name','method')],max)$x)

#REFORMAT VARIABLES
table_3_df[['Variables']] <- sapply(table_3_df[['Variables']],function(x) substr(as.character(x),1,nchar(as.character(x))-7))
#Reduce it to only contain the variables that have not been dropped
table_3_df[['Variables']] <- sapply(table_3_df[['Variables']],function(x){
  
  split_x <- stringr::str_split(x,'_')[[1]]
  
  new_x <- c()
  for (i in seq_along(split_x)){
    
    if (i%%2){
      
      if (split_x[i+1]=='False'){
        new_x <- c(new_x,split_x[i])
      }
      
    }
    
  }
  
  return(paste0(new_x,collapse='_'))
  
})

#Order by Max Score
table_3_df <- table_3_df[order(table_3_df$'Max Score',decreasing=TRUE),]

pdf(paste0(getwd(),'/figures/',method,'/table_testbalacc_area_acc_filename_method_orderby_score.pdf'),height=110,width=30)
grid.table(as.data.frame(table_3_df),rows=NULL)
dev.off()

#Order by Max Test Balanced Accuracy
table_3_df <- table_3_df[order(table_3_df$'Max Test Balanced Accuracy',decreasing=TRUE),]

pdf(paste0(getwd(),'/figures/',method,'/table_testbalacc_area_acc_filename_method_orderby_testbalacc.pdf'),height=110,width=30)
grid.table(as.data.frame(table_3_df),rows=NULL)
dev.off()
