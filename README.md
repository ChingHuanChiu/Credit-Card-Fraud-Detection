
# Credit Card Fraud Detection
  Train the model to detect the credit card if fraud or not with supervised model and unsupervised model

   
# Model
  ## Supervised 
  * Encoder-Decoder  + Classifier:
    the main idea here is try to transform the origin dataset to a different dimension space let the dataset becomes "linear separable"

    ![image](https://github.com/ChingHuanChiu/Credit-Card-Fraud-Detection/blob/master/img/model.png)
  
  ## Unsupervised
  * Encoder-Decoder
    ![image](https://github.com/ChingHuanChiu/Credit-Card-Fraud-Detection/blob/master/img/unsupervised_model.png)
  
# Result
  ## Supervised
  * Confusion matrix

    ![image](https://github.com/ChingHuanChiu/Credit-Card-Fraud-Detection/blob/master/img/supervised_cnf_matrix.png)

  * Score

    * F1 Score : 0.95
    * Fraud Recall : 0.915
  
  ## Unsupervised
    * 評估方法, 步驟如下 : 

    1. 將測試資料進行模型預測
    2. 計算原始資料與預測後的資料計算均方誤差(MSE)
    3. 計算所有資料的MSE的平均值與標準差
    4. 將落於均值正負1倍標準差外的資料視為異常值
    5. 與真實標籤做評估
  
  * 一倍標準差外的資料圖

    ![image](https://github.com/ChingHuanChiu/Credit-Card-Fraud-Detection/blob/master/img/unsupervised_1std.png)


  * Confusion matrix

    ![image](https://github.com/ChingHuanChiu/Credit-Card-Fraud-Detection/blob/master/img/unsupervised_cnf_matrix.png)

  * Score

    * F1 Score : 0.86
    * Fraud Recall : 0.64
 

    