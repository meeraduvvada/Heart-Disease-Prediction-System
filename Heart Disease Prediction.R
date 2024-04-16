library(tidyverse)
library(dslabs)
library(dplyr)
library(caret)
library(lubridate)
library(tidytext)
library("RColorBrewer")
library(randomForest)
library(tictoc)
library(e1071)
library(ggpubr)
heart_disease_data <- read_csv("heart_statlog_cleveland_hungary_final1.csv")
heart_disease_data %>% head()

heart_disease_data %>% summarize(n_age = n_distinct(age), n_sex = n_distinct(sex),
                                 n_chestpaintype = n_distinct(chestpaintype), n_restingbps = n_distinct(restingbps),
                                 n_cholesterol = n_distinct(cholesterol), n_fastingbloodsugar = n_distinct(fastingbloodsugar),
                                 n_restingecg = n_distinct(restingecg), n_maxheartrate = n_distinct(maxheartrate),
                                 n_exerciseangina = n_distinct(exerciseangina), n_oldpeak = n_distinct(oldpeak),
                                 n_STslope = n_distinct(STslope), n_target = n_distinct(target))
rlang::last_error()

heart_disease_data %>% group_by(age, target) %>% summarize(count = n()) %>%
  ggplot() + geom_bar(aes(age, count,   fill = as.factor(target)), stat = "Identity") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 90, size = 10)) + 
  ylab("Count") + xlab("Age") + labs(fill = "Condition")

heart_disease_data %>% filter(target == 1) %>% group_by(age, chestpaintype) %>% summarise(count = n()) %>%
  ggplot() + geom_bar(aes(age, count,   fill = as.factor(chestpaintype)),stat = "Identity") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 90, size = 10)) + 
  ylab("Count") + xlab("Age") + labs(fill = "Condition") + 
  ggtitle("Age vs. Count (disease only) for various chest pain conditions") +
  scale_fill_manual(values=c("red", "blue", "green", "black"))                                                                   

options(repr.plot.width = 20, repr.plot.height = 8) 

heart_disease_data %>%ggballoonplot(x = "age", y = "sex",
                                     size = "cholesterol", size.range = c(5, 30), fill = "target",show.label = FALSE,
                                     ggtheme = theme_bw()) + scale_fill_viridis_c(option = "C") + theme(axis.text.x = element_text(angle = 90, size = 10)) +ggtitle("Age vs. Sex Map") + labs(fill = "Condition")


options(repr.plot.width = 20, repr.plot.height = 8) 
heart_disease_data %>% ggballoonplot(x = "age", y = "chestpaintype",
                                     size = "cholesterol", size.range = c(5, 30), fill = "sex",show.label = FALSE,
                                     ggtheme = theme_bw()) +
  scale_fill_viridis_c(option = "C") + 
  theme(axis.text.x = element_text(angle = 90, size = 10)) +
  ggtitle("Age vs. Chest Pain Map") + labs(fill = "sex")


#disease prediction
set.seed(2020, sample.kind = "Rounding")
# Divide into train and validation dataset
test_index <- createDataPartition(y = heart_disease_data$target, times = 1, p = 0.2, list= FALSE)
train_set <- heart_disease_data[-test_index, ]
validation <- heart_disease_data[test_index, ]

# Converting the dependent variables to factors
train_set$target <- as.factor(train_set$target)
validation$target <- as.factor(validation$target)


################################
# LDA Analysis
###############################

lda_fit <- train(target ~ ., method = "lda", data = train_set)
lda_predict <- predict(lda_fit, validation)
confusionMatrix(lda_predict, validation$target)

################################
# QDA Analysis
###############################

qda_fit <- train(target ~ ., method = "qda", data = train_set)
qda_predict <- predict(qda_fit, validation)
confusionMatrix(qda_predict, validation$target)

#knn
ctrl <- trainControl(method = "cv", verboseIter = FALSE, number = 5)
knnFit <- train(target ~ ., 
                data = train_set, method = "knn", preProcess = c("center","scale"),
                trControl = ctrl , tuneGrid = expand.grid(k = seq(1, 20, 2)))

plot(knnFit)
toc()

knnPredict <- predict(knnFit,newdata = validation )
knn_results <- confusionMatrix(knnPredict, validation$target )

knn_results


# SVM
############################

ctrl <- trainControl(method = "cv", verboseIter = FALSE, number = 5)

grid_svm <- expand.grid(C = c(0.01, 0.1, 1, 10, 20))

tic(msg= " Total time for SVM :: ")
svm_fit <- train(target ~ .,data = train_set,
                 method = "svmLinear", preProcess = c("center","scale"),
                 tuneGrid = grid_svm, trControl = ctrl)

plot(svm_fit)
toc()
svm_predict <- predict(svm_fit, newdata = validation)
svm_results <- confusionMatrix(svm_predict, validation$target)

svm_results

# RF
############################
control<- trainControl(method = "cv", number = 5, verboseIter = FALSE)
grid <-data.frame(mtry = seq(1, 10, 2))
tic(msg= " Total time for rf :: ")
rf_fit <- train(target ~ ., method = "rf", data = train_set, ntree = 20, trControl = control,
                tuneGrid = grid)

plot(rf_fit)
toc()
rf_predict <- predict(rf_fit, newdata = validation)

rf_results <- confusionMatrix(rf_predict, validation$target)

rf_results

# GBM

gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 10, 25, 30),
                        n.trees = c(5, 10, 25, 50),
                        shrinkage = c(0.1, 0.2, 0.3,  0.4, 0.5),
                        n.minobsinnode = 20)

tic(msg= " Total time for GBM :: ")
gbm_fit <- train(target ~ ., method = "gbm", data = train_set,  trControl = control, verbose = FALSE,
                 tuneGrid = gbmGrid)

plot(gbm_fit)
toc()
gbm_predict <- predict(gbm_fit, newdata = validation)

gbm_results <- confusionMatrix(gbm_predict, validation$target)

gbm_results
