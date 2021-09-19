
# Credit Card Fraud Detection
  Train the model to detect the credit card if fraud or not with deep learning

# Data (From Kaggle)
  ## Imblance data
   * SMOTE
   
  ## Data Clip
   * clip the data if datapoint is outlier( mean + 2 * std ) against the "Amount" feature  
   
# Model
  * Use Encoder-Decoder + Classifier
  * Learning rate warm-up with 5 epochs
  
# Result
  AUC : 0.99
  Fraud Recall : 0.948
  
  ![image](https://github.com/ChingHuanChiu/Credit-Card-Fraud-Detection/blob/master/img/credict card detection.png)
  

    