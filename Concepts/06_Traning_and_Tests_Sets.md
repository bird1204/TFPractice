# Training and Test Sets

## Splitting Data

在前一章，我們已經知道什麼是訓練集、測試集了

![](https://developers.google.com/machine-learning/crash-course/images/PartitionTwoSets.svg?authuser=1&refresh=1)

我們還要確認測試集具備以下條件：

* 資料量要足夠大，大到具備統計意義
* 要能夠代表整個數據集，也就是說不要選擇與訓練集具備不同特徵的資料測試

最後一件事：**永遠不要使用測試集訓練 model**

如果發現你的評估結果超級好，就有可能是因為不小心用測試集訓練 model 了