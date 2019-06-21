# Validation Set

## Another Partition

我們現在知道要把資料分成兩個部分了，理想的情況下整個流程會長這樣：

![](https://developers.google.com/machine-learning/crash-course/images/WorkflowWithTestSet.svg)

但這樣真的是可行的方式嗎？
試著想想看，我 train 了一個 model 然後用 Test Set 評估後，發現不夠好，接著調整了一些參數，再用 Test Set 評估一次，持續迭代這個過程，直到評估的結果是好的～

阿...這樣不就變相的把 Test Set 拿來 Train 了嗎？應該會造成 over fitting 的問題吧～

這邊有個方法，另外切一個 Validation Set

![](https://developers.google.com/machine-learning/crash-course/images/PartitionThreeSets.svg)

然後我們會得到更合適的 workflow

![](https://developers.google.com/machine-learning/crash-course/images/WorkflowWithValidationSet.svg)

小提醒：  
測試集、驗證集再重複的使用下，會產生一定的磨損，對相同的資料進行越多的超參數調整，能夠在新資料得到好預測的機會就越低。但是驗證集的磨損會比測試集低得多

用更多的新資料替換 Test Set, Validation Set 是一個好方式，如果沒有辦法，直接重新訓練也是一個方式～
