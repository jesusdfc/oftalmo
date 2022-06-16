library(ggplot2)
args = commandArgs(trailingOnly=TRUE)

method <- args[1]
message('Loading file with method ',method)
excel_df <- read.table(file=paste0(getwd(),'/best_conf/nested_cv/',method,'_best_conf_6_6.tsv'),sep='\t',header = TRUE,row.names = NULL)


#FIRST FIGURE, TEST BALANCED ACCURACY AS A FUNCTION OF THE SELECTED METHOD 'RF','SVM','XGBOOST'
message('Generating Test Balanced Accuracy as a function of the selected method "RF", "SVM" or "XGBOOST"')
pdf(paste0(getwd(),'/figures/',method,'/testbalacc_selectedmethod.pdf'))
ggplot(excel_df,aes(x=method))+
       theme_bw() +
       geom_violin(aes(y=test_balacc,fill=method),scale='width')
dev.off()

#SECOND FIGURE, TRAIN BALANCED ACCURACY AS A FUNCTION OF THE SELECTED METHOD 'RF','SVM','XGBOOST'
message('Generating Train Balanced Accuracy as a function of the selected method "RF", "SVM" or "XGBOOST"')
pdf(paste0(getwd(),'/figures/',method,'/trainbalacc_selectedmethod.pdf'))
ggplot(excel_df,aes(x=method))+
  theme_bw() +
  geom_violin(aes(y=train_balacc,fill=method),scale='width')
dev.off()

#THIRD FIGURE, TEST AREA REJECT RATE AS A FUNCTION OF THE SELECTED METHOD 'RF','SVM','XGBOOST'
message('Generating Test Area Rejection Rate as a function of the selected method "RF", "SVM" or "XGBOOST"')
pdf(paste0(getwd(),'/figures/',method,'/testarearejectrate_selectedmethod.pdf'))
ggplot(excel_df,aes(x=method))+
  theme_bw() +
  geom_violin(aes(y=test_area_reject_rate,fill=method),scale='width')
dev.off()

#FOURTH FIGURE, TEST AREA ACCURACY AS A FUNCTION OF THE SELECTED METHOD 'RF','SVM','XGBOOST'
message('Generating Test Area Accuracy as a function of the selected method "RF", "SVM" or "XGBOOST"')
pdf(paste0(getwd(),'/figures/',method,'/test_area_acc_selectedmethod.pdf'))
ggplot(excel_df,aes(x=method))+
  theme_bw() +
  geom_violin(aes(y=test_area_acc,fill=method),scale='width')
dev.off()


#FIFTH FIGURE, TEST ACCURACY THRESHOLD AS A FUNCTION OF THE SELECTED METHOD 'RF','SVM','XGBOOST'
message('Generating Test Acuraccy with Thresholds as a function of the selected method "RF", "SVM" or "XGBOOST"')
#Make category for test_acc_th_{0.55-0.95}
test_acc_th_df <- c()

for (item in colnames(excel_df)[18:26]){
  
    test_acc_th_df <- rbind(test_acc_th_df,data.frame(method = excel_df[['method']], 
                                                      threshold = rep(item,nrow(excel_df)),
                                                      value = excel_df[[item]]))
}

# pdf(paste0(getwd(),'/figures/',method,'/test_acc_th_selectedmethod_boxplot.pdf'))
# ggplot(test_acc_th_df,aes(x=threshold,y=value,fill=method))+
# theme_bw() +
# geom_boxplot()+
# theme(axis.text.x = element_text(angle = 45))
# dev.off()


