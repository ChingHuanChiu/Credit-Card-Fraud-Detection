
# Credit Card Fraud Detection
  Train the model to detect the credit card if fraud or not with deep learning

# What have I Done:

  ## Data (From Kaggle)

    * Imblance data
      * SMOTE
   
# Model
  
  * Use structure of Encoder-Decoder  + Classifier:
    the main idea here is try to transform the origin dataset to a different dimension space let the dataset becomes "linear separable"

  ![image](https://github.com/ChingHuanChiu/Credit-Card-Fraud-Detection/blob/master/img/model.png)
  
  * Learning rate warm-up with 3 epochs
  
# Result
  AUC : 0.99
  
  Fraud Recall : 0.93
  
  ![image](https://github.com/ChingHuanChiu/Credit-Card-Fraud-Detection/blob/master/img/credict%20card%20detection.png)
  

    