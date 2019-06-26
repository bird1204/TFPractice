# Classification

## Thresholding

邏輯迴歸回傳可能機率，我們可以直接使用回傳的可能機率，或是轉換成二元分類的預測結果。

如果一個邏輯迴歸 model ，預測特定 E-mail 有 0.9995 個可能性是 spam ，那他很有可能真的是 spam；反之，預測特定 E-mail 有 0.0003 個可能性是 spam，那他很有可能不是 spam。但是預測值如果是 0.6 的 E-mail 呢？

為了要把邏輯迴歸值對應到二元結果，就必須定義一個 classification threshold ( 也被稱為 decision threshold )。如果預測值高於 threshold 那就會被辨識為 spam；低於 threshold 則不是 spam。 

threshold 可以很間單的設為 0.5 ，但實際上 threshold 要依據問題作調整


## True vs. False; Positive vs. Negative

We can summarize our "wolf-prediction" model using a 2x2 confusion matrix that depicts all four possible outcomes:

我們直接以一個 2x2 的矩陣來說明"狼來了"的預測結果，做個總結：

|          |                                          True                                         |                                              False                                              |
|:--------:|:-------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| Positive |   True Positive (TP):<br> 現實: 有狼<br> 牧羊人說: "有狼"<br> 結果: 牧羊人是英雄<br>  | False Positive (FP):<br> 現實: 沒有狼<br> 牧羊人說: "有狼"<br> 結果: 村民很生氣被牧羊人吵醒<br> |
| Negative | False Negative (FN):<br> 現實: 有狼<br> 牧羊人說: "沒有狼"<br> 結果: 狼把羊吃光了<br> |        True Negative (TN):<br> 現實: 沒有狼<br> 牧羊人說: "沒有狼"<br> 結果: 沒事兒～<br>       |

## Accuracy

Accuracy 是其中一種用來衡量分類模型的指標，非正式地，accuracy 是用來評斷 model 有沒有正確預測的分數，形式上，我們有這個公式：

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtext>Accuracy</mtext>
  <mo>=</mo>
  <mfrac>
    <mtext>Number of correct predictions</mtext>
    <mtext>Total number of predictions</mtext>
  </mfrac>
</math>

用在二元分類上，accuracy 可以被看成下面的公式：

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtext>Accuracy</mtext>
  <mo>=</mo>
  <mfrac>
    <mrow>
      <mi>T</mi>
      <mi>P</mi>
      <mo>+</mo>
      <mi>T</mi>
      <mi>N</mi>
    </mrow>
    <mrow>
      <mi>T</mi>
      <mi>P</mi>
      <mo>+</mo>
      <mi>T</mi>
      <mi>N</mi>
      <mo>+</mo>
      <mi>F</mi>
      <mi>P</mi>
      <mo>+</mo>
      <mi>F</mi>
      <mi>N</mi>
    </mrow>
  </mfrac>
</math>

假設 model 預測 100 次"狼來了"的結果如下：

|          | True | False  |
|:--------:|:----:|:------:|
| Positive | 1    | 1      |
| Negative | 8    | 90     |

我們會得到 accuray = 0.91：

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtext>Accuracy</mtext>
  <mo>=</mo>
  <mfrac>
    <mrow>
      <mi>T</mi>
      <mi>P</mi>
      <mo>+</mo>
      <mi>T</mi>
      <mi>N</mi>
    </mrow>
    <mrow>
      <mi>T</mi>
      <mi>P</mi>
      <mo>+</mo>
      <mi>T</mi>
      <mi>N</mi>
      <mo>+</mo>
      <mi>F</mi>
      <mi>P</mi>
      <mo>+</mo>
      <mi>F</mi>
      <mi>N</mi>
    </mrow>
  </mfrac>
  <mo>=</mo>
  <mfrac>
    <mrow>
      <mn>1</mn>
      <mo>+</mo>
      <mn>90</mn>
    </mrow>
    <mrow>
      <mn>1</mn>
      <mo>+</mo>
      <mn>90</mn>
      <mo>+</mo>
      <mn>1</mn>
      <mo>+</mo>
      <mn>8</mn>
    </mrow>
  </mfrac>
  <mo>=</mo>
  <mn>0.91</mn>
</math>

雖然 accuracy 看起來有不錯的結果，但是 9 個有狼的情況，卻有 8 個沒有被預測出來。當我們在處理 class-imbalanced 的資料集時，只使用 accuracy 並沒辦法說明完整的故事，就像上面這個例子一樣，接下來我們會介紹更適合用來衡量 class-imbalanced 的資料集的指標：precision and recall

## Precision and Recall

**Precision** 旨在回答：

````
有多少比例的 positive 預測是正確的？
````
公式如下：

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtext>Precision</mtext>
  <mo>=</mo>
  <mfrac>
    <mrow>
      <mi>T</mi>
      <mi>P</mi>
    </mrow>
    <mrow>
      <mi>T</mi>
      <mi>P</mi>
      <mo>+</mo>
      <mi>F</mi>
      <mi>P</mi>
    </mrow>
  </mfrac>
</math>

|          | True | False  |
|:--------:|:----:|:------:|
| Positive | 1    | 1      |
| Negative | 8    | 90     |

回到上一節的例子，我們的 precision 是 0.5，也就是當 model 預測"狼來了"的時候，準確率是 50% ：

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtext>Precision</mtext>
  <mo>=</mo>
  <mfrac>
    <mrow>
      <mi>T</mi>
      <mi>P</mi>
    </mrow>
    <mrow>
      <mi>T</mi>
      <mi>P</mi>
      <mo>+</mo>
      <mi>F</mi>
      <mi>P</mi>
    </mrow>
  </mfrac>
  <mo>=</mo>
  <mfrac>
    <mn>1</mn>
    <mrow>
      <mn>1</mn>
      <mo>+</mo>
      <mn>1</mn>
    </mrow>
  </mfrac>
  <mo>=</mo>
  <mn>0.5</mn>
</math>

**Recall** 旨在回答：

````
有多少比例的 actual positives 預測是正確的？
````

公式如下：

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtext>Recall</mtext>
  <mo>=</mo>
  <mfrac>
    <mrow>
      <mi>T</mi>
      <mi>P</mi>
    </mrow>
    <mrow>
      <mi>T</mi>
      <mi>P</mi>
      <mo>+</mo>
      <mi>F</mi>
      <mi>N</mi>
    </mrow>
  </mfrac>
</math>

回到上一節的例子，我們的 Recall 是 0.11，也就是 model 預測到了 11% 的"狼來了" ：

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtext>Recall</mtext>
  <mo>=</mo>
  <mfrac>
    <mrow>
      <mi>T</mi>
      <mi>P</mi>
    </mrow>
    <mrow>
      <mi>T</mi>
      <mi>P</mi>
      <mo>+</mo>
      <mi>F</mi>
      <mi>N</mi>
    </mrow>
  </mfrac>
  <mo>=</mo>
  <mfrac>
    <mn>1</mn>
    <mrow>
      <mn>1</mn>
      <mo>+</mo>
      <mn>8</mn>
    </mrow>
  </mfrac>
  <mo>=</mo>
  <mn>0.11</mn>
</math>

#### A Tug of War

要全面的評估 model 的效果，必須**同時**使用 precision 和 recall，不幸地是，這兩者會影響彼此的消長，甚至可能呈現反比關係，透過下圖，我們試著更詳細的說明：

![](https://developers.google.com/machine-learning/crash-course/images/PrecisionVsRecallBase.svg)

我們會得到：

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtext>Precision</mtext>
  <mo>=</mo>
  <mfrac>
    <mrow>
      <mi>T</mi>
      <mi>P</mi>
    </mrow>
    <mrow>
      <mi>T</mi>
      <mi>P</mi>
      <mo>+</mo>
      <mi>F</mi>
      <mi>P</mi>
    </mrow>
  </mfrac>
  <mo>=</mo>
  <mfrac>
    <mn>8</mn>
    <mrow>
      <mn>8</mn>
      <mo>+</mo>
      <mn>2</mn>
    </mrow>
  </mfrac>
  <mo>=</mo>
  <mn>0.8</mn>
</math>
<br><br>
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtext>Recall</mtext>
  <mo>=</mo>
  <mfrac>
    <mrow>
      <mi>T</mi>
      <mi>P</mi>
    </mrow>
    <mrow>
      <mi>T</mi>
      <mi>P</mi>
      <mo>+</mo>
      <mi>F</mi>
      <mi>N</mi>
    </mrow>
  </mfrac>
  <mo>=</mo>
  <mfrac>
    <mn>8</mn>
    <mrow>
      <mn>8</mn>
      <mo>+</mo>
      <mn>3</mn>
    </mrow>
  </mfrac>
  <mo>=</mo>
  <mn>0.73</mn>
</math>

接著我們調整 classification threhold 的位置：

![](https://developers.google.com/machine-learning/crash-course/images/PrecisionVsRecallRaiseThreshold.svg)

我們會得到：

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtext>Precision</mtext>
  <mo>=</mo>
  <mfrac>
    <mrow>
      <mi>T</mi>
      <mi>P</mi>
    </mrow>
    <mrow>
      <mi>T</mi>
      <mi>P</mi>
      <mo>+</mo>
      <mi>F</mi>
      <mi>P</mi>
    </mrow>
  </mfrac>
  <mo>=</mo>
  <mfrac>
    <mn>7</mn>
    <mrow>
      <mn>7</mn>
      <mo>+</mo>
      <mn>1</mn>
    </mrow>
  </mfrac>
  <mo>=</mo>
  <mn>0.88</mn>
</math>
<br><br>
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtext>Recall</mtext>
  <mo>=</mo>
  <mfrac>
    <mrow>
      <mi>T</mi>
      <mi>P</mi>
    </mrow>
    <mrow>
      <mi>T</mi>
      <mi>P</mi>
      <mo>+</mo>
      <mi>F</mi>
      <mi>N</mi>
    </mrow>
  </mfrac>
  <mo>=</mo>
  <mfrac>
    <mn>7</mn>
    <mrow>
      <mn>7</mn>
      <mo>+</mo>
      <mn>4</mn>
    </mrow>
  </mfrac>
  <mo>=</mo>
  <mn>0.64</mn>
</math>

接著我們繼續把 classification threhold 往左移：

![](https://developers.google.com/machine-learning/crash-course/images/PrecisionVsRecallLowerThreshold.svg)

我們得到：

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtext>Precision</mtext>
  <mo>=</mo>
  <mfrac>
    <mrow>
      <mi>T</mi>
      <mi>P</mi>
    </mrow>
    <mrow>
      <mi>T</mi>
      <mi>P</mi>
      <mo>+</mo>
      <mi>F</mi>
      <mi>P</mi>
    </mrow>
  </mfrac>
  <mo>=</mo>
  <mfrac>
    <mn>9</mn>
    <mrow>
      <mn>9</mn>
      <mo>+</mo>
      <mn>3</mn>
    </mrow>
  </mfrac>
  <mo>=</mo>
  <mn>0.75</mn>
</math>
<br><br>
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtext>Recall</mtext>
  <mo>=</mo>
  <mfrac>
    <mrow>
      <mi>T</mi>
      <mi>P</mi>
    </mrow>
    <mrow>
      <mi>T</mi>
      <mi>P</mi>
      <mo>+</mo>
      <mi>F</mi>
      <mi>N</mi>
    </mrow>
  </mfrac>
  <mo>=</mo>
  <mfrac>
    <mn>9</mn>
    <mrow>
      <mn>9</mn>
      <mo>+</mo>
      <mn>2</mn>
    </mrow>
  </mfrac>
  <mo>=</mo>
  <mn>0.82</mn>
</math>

已經有很多被開發來衡量 precision 和 recall 的方式，[F1 scroe](https://en.wikipedia.org/wiki/F1_score)是其中之一

## ROC Curve and AUC

#### ROC curve
具體概念是：我們對每個可能的 classification threshold 進行評估，並觀察相應 classification threshold 的 True Positive Rate ( Recall ) 和 False Positive Rate :

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>T</mi>
  <mi>P</mi>
  <mi>R</mi>
  <mo>=</mo>
  <mfrac>
    <mrow>
      <mi>T</mi>
      <mi>P</mi>
    </mrow>
    <mrow>
      <mi>T</mi>
      <mi>P</mi>
      <mo>+</mo>
      <mi>F</mi>
      <mi>N</mi>
    </mrow>
  </mfrac>
</math>
<br><br>
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>F</mi>
  <mi>P</mi>
  <mi>R</mi>
  <mo>=</mo>
  <mfrac>
    <mrow>
      <mi>F</mi>
      <mi>P</mi>
    </mrow>
    <mrow>
      <mi>F</mi>
      <mi>P</mi>
      <mo>+</mo>
      <mi>T</mi>
      <mi>N</mi>
    </mrow>
  </mfrac>
</math>

所以我們 ROC 找出來，得到一個曲線：

![](https://developers.google.com/machine-learning/crash-course/images/ROCCurve.svg)

#### AUC: Area Under the ROC Curve

然後我們借助曲線下的面積來找出那一個點，不然直接從線上找出那一點太沒效率了。

![](https://developers.google.com/machine-learning/crash-course/images/AUC.svg)

AUC 提供了所有可能的 classification threshold 的總效能指標，其中一種解釋 AUC 的方式是： model 針對隨機 positive 樣本的排序的可能性會比針對隨機 negative 樣本的可能性還高

![](https://developers.google.com/machine-learning/crash-course/images/AUCPredictionsRanked.svg)

AUC 表示 actual positive 位於 actual negative 的機率；其範圍是 0 ~ 1 ，預測 100% 錯誤的 AUC 是 0.0 ；反之則為 1.1 。

因為下面兩個原因，我們認為 AUC 可以使用：

1. AUC 是 **規模不變的 ( scale-invariant )**. 他衡量被排序過後的預測位置，而非絕對值。
2. AUC 是 **classification-threshold-invariant**. 他衡量預測的品質，而非 threshold

但是上述兩點也需注意，可能影響我們要不要使用 AUC :

* 我們有時候不希望 scale-invariant : 我們已經做得夠好了，但是 AUC 不這樣顯示。
* 我們有時候不希望 classification-threshold-invariant : 沒有人希望狼來了的時候沒有被通知，所以我們可能寧願選擇不要給我壞結果的誤報 ( 會讓 FP 很高，但卻是我們要的 )

## Prediction Bias

邏輯迴歸預測應該要沒有偏差，也就是 :

````
預測的平均應該要 ≈ 觀察的平均

````
**prediction bias** 是我們的兩個平均之間，距離了多遠？

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtext>prediction bias</mtext>
  <mo>=</mo>
  <mtext>average of predictions</mtext>
  <mo>&#x2212;<!-- − --></mo>
  <mtext>average of labels in data set</mtext>
</math>

Prediction Bias 不為零，透露出 model 中一定有某些錯誤，因為這意味著 model 在 positive label 出現的頻率上是錯誤的。

舉例來說，我們知道只有 1% 的 E-mail 是 spam，因此 model 在預測的時候也應該得到只有 1% 的 E-mail 是 spam，如果預測出來有 20% 是 spam ，我們可以推斷出現了 Prediction Bias

可能會造成 prediction bias 的原因：

* 不完整的 feature set
* 吵雜的 data set
* Machine learning 過程有問題
* 已經是 bias 的 training sample
* regularization 做過頭

你可能會想要用一些學習後處理，來導正  prediction bias ，或者是說，透過增加一個 calibration layer 來減少 prediction bias，但是因為某些理由，這個方法並不好：

* 治標不治本
* 需要因為各種因素，不斷更新模型

````
注意：一個好模型的 bias 通常會趨近於 0
````

#### Bucketing and Prediction Bias

邏輯迴歸預測 0 ~ 1 之間的值，可是所有 labeled example 不是 0 就是 1 。 因此在測試 prediction bias 的時候，不能用一個 example 的判斷，必須針對一"桶 ( bucket ) " example。也就是說邏輯迴歸在預測 bias 的時候必須要同時比較 predicted value 和 observed values 

分 bucket 的方式有 :

* 把目標的預測值做線性區分
* 分位數

假設每個 bucket 有 1000 筆資料，然後我們會得到下圖：

* X 軸表示平均預測值
* Y 軸表示實際預測值

````
兩個值都取過 log
````

![](https://developers.google.com/machine-learning/crash-course/images/BucketingBias.svg)

這邊我們會發現有幾個點，是離群值，為什麼會這樣呢？

* training set 不足以代表那個點
* 這個點比其他點還更複雜
* regularization 做過頭


