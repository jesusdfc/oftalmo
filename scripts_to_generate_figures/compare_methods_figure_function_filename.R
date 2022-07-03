library(ggplot2)
library(gridExtra)
args = commandArgs(trailingOnly=TRUE)


#LOAD THE DATASET
Model <- args[1]
message('Loading file with Model ',Model)
excel_df <- read.table(file=paste0(getwd(),'/best_conf/nested_cv/with_ROC/',Model,'_best_conf_6_6.tsv'), sep='\t', header = TRUE, row.names = NULL)

#--------------------> FIRST TABLE, VALIDATION BALANCED ACCURACY AS A FUNCTION OF THE FILE NAME (VARS) <------------------------#

# message('Generating first figure')
# table_1_df <- aggregate(excel_df[['Val.BalAcc']],excel_df['FileName'],mean)
# colnames(table_1_df) <- c('Variables','Mean Val Balanced Accuracy')
# table_1_df['Max Val Balanced Accuracy'] <- list('Max Val Balanced Accuracy' = aggregate(excel_df[['Val.BalAcc']],excel_df['FileName'],max)$x)
# table_1_df['Min Val Balanced Accuracy'] <- list('Min Val Balanced Accuracy' = aggregate(excel_df[['Val.BalAcc']],excel_df['FileName'],min)$x)

# table_1_df[['Variables']] <- sapply(table_1_df[['Variables']],function(x) substr(as.character(x),1,nchar(as.character(x))-7))
# #Reduce it to only contain the variables that have not been dropped
# table_1_df[['Variables']] <- sapply(table_1_df[['Variables']],function(x){
  
#   split_x <- stringr::str_split(x,'_')[[1]]
  
#   new_x <- c()
#   for (i in seq_along(split_x)){
    
#     if (i%%2){
      
#       if (split_x[i+1]=='False'){
#         new_x <- c(new_x,split_x[i])
#       }
      
#     }
    
#   }
  
#   return(paste0(new_x,collapse='_'))
  
# })

# table_1_df <- table_1_df[order(table_1_df$'Max Val Balanced Accuracy',decreasing=TRUE),]

# pdf(paste0(getwd(),'/figures/',Model,'/summary_tables/table_valbalacc_filename.pdf'),height=40,width=15)
# grid.table(as.data.frame(table_1_df),rows=NULL)
# dev.off()


#-----------------------> SECOND TABLE, SAME BUT ALSO GROUPING BY PREDICTION Model <---------------------------#

# message('Generating second figure')
# table_2_df <- aggregate(excel_df[['Val.BalAcc']],excel_df[c('FileName','Model')],mean)
# colnames(table_2_df) <- c('Variables','Model','Mean Val Balanced Accuracy')
# table_2_df['Max Val Balanced Accuracy'] <- list('Max Val Balanced Accuracy' = aggregate(excel_df[['Val.BalAcc']], excel_df[c('FileName','Model')],max)$x)
# table_2_df['Min Val Balanced Accuracy'] <- list('Min Val Balanced Accuracy' = aggregate(excel_df[['Val.BalAcc']], excel_df[c('FileName','Model')],min)$x)

# table_2_df[['Variables']] <- sapply(table_2_df[['Variables']],function(x) substr(as.character(x),1,nchar(as.character(x))-7))
# #Reduce it to only contain the variables that have not been dropped
# table_2_df[['Variables']] <- sapply(table_2_df[['Variables']],function(x){
  
#   split_x <- stringr::str_split(x,'_')[[1]]
  
#   new_x <- c()
#   for (i in seq_along(split_x)){
    
#     if (i%%2){
      
#       if (split_x[i+1]=='False'){
#         new_x <- c(new_x,split_x[i])
#       }
      
#     }
    
#   }
  
#   return(paste0(new_x,collapse='_'))
  
# })

# table_2_df <- table_2_df[order(table_2_df$'Max Val Balanced Accuracy',decreasing=TRUE),]

# pdf(paste0(getwd(),'/figures/',Model,'/summary_tables/table_valbalacc_filename_Model.pdf'),height=120,width=15)
# grid.table(as.data.frame(table_2_df),rows=NULL)
# dev.off()

#-------------------------> THIRD TABLE, SAME BUT ALSO GROUPING BY PREDICTION Model + Val Area <------------------------#

message('Generating third figure')
table_3_df <- aggregate(excel_df[['Val.BalAcc']], excel_df[c('FileName','Model')],mean)
colnames(table_3_df) <- c('Variables','Model', 'Mean ValBAC')

#ADD BALANCED ACCURACY
table_3_df['Max ValBAC'] <- list('Max ValBAC' = aggregate(excel_df[['Val.BalAcc']], excel_df[c('FileName','Model')],max)$x)
table_3_df['Min ValBAC'] <- list('Min ValBAC' = aggregate(excel_df[['Val.BalAcc']], excel_df[c('FileName','Model')],min)$x)

#ADD AUAC
table_3_df['Mean ValAUAC'] <- list('Mean ValAUAC' = aggregate(excel_df[['Val.AUAC']], excel_df[c('FileName','Model')],mean)$x)
table_3_df['Min ValAUAC'] <- list('Min ValAUAC' = aggregate(excel_df[['Val.AUAC']], excel_df[c('FileName','Model')],min)$x)
table_3_df['Max ValAUAC'] <- list('Max ValAUAC' = aggregate(excel_df[['Val.AUAC']], excel_df[c('FileName','Model')],max)$x)

#ADD AURC
table_3_df['Mean ValAURC'] <- list('Mean ValAURC' = aggregate(excel_df[['Val.AURC']], excel_df[c('FileName','Model')],mean)$x)
table_3_df['Min ValAURC'] <- list('Min ValAURC' = aggregate(excel_df[['Val.AURC']], excel_df[c('FileName','Model')],min)$x)
table_3_df['Max ValAURC'] <- list('Max ValAURC' = aggregate(excel_df[['Val.AURC']], excel_df[c('FileName','Model')],max)$x)

#ADD AUROC
table_3_df['Mean ValAUROC'] <- list('Mean ValAUROC' = aggregate(excel_df[['Val.AUROC']], excel_df[c('FileName','Model')],mean)$x)
table_3_df['Min ValAUROC'] <- list('Min ValAUROC' = aggregate(excel_df[['Val.AUROC']], excel_df[c('FileName','Model')],min)$x)
table_3_df['Max ValAUROC'] <- list('Max ValAUROC' = aggregate(excel_df[['Val.AUROC']], excel_df[c('FileName','Model')],max)$x)

#ADD Score
table_3_df['Max ValScore'] <- list('Max ValScore' = aggregate(excel_df[['Val.Score']], excel_df[c('FileName','Model')],max)$x)



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

#Order by MAX SCORE
table_3_df <- table_3_df[order(table_3_df$'Max ValScore',decreasing=TRUE),]

pdf(paste0(getwd(),'/figures/',Model,'/summary_tables/table_summary_orderby_ValScore.pdf'),height=110,width=25)
grid.table(as.data.frame(table_3_df),rows=NULL)
dev.off()

#Order by MAX ValBAC
table_3_df <- table_3_df[order(table_3_df$'Max ValBAC',decreasing=TRUE),]

pdf(paste0(getwd(),'/figures/',Model,'/summary_tables/table_summary_orderby_valBAC.pdf'),height=110,width=25)
grid.table(as.data.frame(table_3_df),rows=NULL)
dev.off()

#Order by MAX ValAUROC
table_3_df <- table_3_df[order(table_3_df$'Max ValAUROC',decreasing=TRUE),]

pdf(paste0(getwd(),'/figures/',Model,'/summary_tables/table_summary_orderby_valAUROC.pdf'),height=110,width=25)
grid.table(as.data.frame(table_3_df),rows=NULL)
dev.off()
