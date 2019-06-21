# Generalization

## Peril of Overfitting

接下來我們來看看 overfitting 會造成什麼後果
這個章節，我們主要討論 Generalization，為了讓大家看了更有感覺，接下來會由幾張圖做呈現：

* 藍點是生病的樹
* 橘點是健康的樹

接下來我們看第一張圖：

![](https://developers.google.com/machine-learning/crash-course/images/GeneralizationA.png)

想像一個 model 用這些資料 train 出一個 Loss 非常低的預測 ( 你可以自己畫圖試試看 )，畫完的話，會得到下面這張圖：

![](https://developers.google.com/machine-learning/crash-course/images/GeneralizationB.png)

## Low loss, but still a bad model?

太好了，我們訓練出一個 Loss 超級低的 model，接下來我們丟新的資料進去做預測吧！

![](https://developers.google.com/machine-learning/crash-course/images/GeneralizationC.png?authuser=1&refresh=1)

shit, 訓練出來的 model 根本沒有辦法辨識新的資料啊！

這個情況就是所謂的 overfit，一個 overfitting 的 model 在訓練資料的預測表現會顯得非常好，卻沒有辦法處理新的資料，造成 overfit 的原因是 model 比他所需要的還要更複雜。**一個好的 ML 必須用最簡單的方式去 fitting 訓練資料。**

ML 的目標是準確的預測真實世界的新資料，很不幸地，model 看不到全部的真相，而是從測試資料去推敲真實世界的樣貌，如果 model 很透徹的對應到樣本，那我們要怎麼相信 model 對於前所未見的資料能夠有好的預測呢？

來自 14 世紀的 William of Ockham，他是史上第一位 ML 理論家，提出了 Ockham's razor - 如無必要，勿增實體，套進 ML 的詞彙：

````
ML 的模型越簡單，就越不可能是因為訓練樣本的特性，而得到好的預測結果
````

在現代，圍繞這個觀念，我們提出了 generalization bounds - 影響一個 model 是否有能力 generalize 到新資料的因素：

* model 的複雜度
* model 在訓練資料的效能

在實務上，我們會把整個資料集分成：

* 訓練集 - 用來 train
* 測試集 - 用來 test

在幾個前提之下，擁有好的測試集預測結果就是一個好 model 的指標：

* 測試集夠大
* 不可以一在的使用相同的測試集作弊

## The ML fine print

對於 ML，還是有幾點注意事項，三點基本假設：

1. 我们要以**獨立**且**一致** ( i.i.d ) 的方式從该分布**隨機**抽取樣本
2. 分布是平穩 ( stationary ) 的，不隨時間產生變化
3. 我們總是從同一個分布提取樣本

在實務上，我們有時候會違反這些假設，舉例來說：

* 假設 model 要選擇哪個廣告要曝光：則 i.i.d 就會因為有些 user 已經看過廣告，而被違反
* 假設資料集包含了整年的銷售資料：則stationary 就會因為有分淡旺季，而被違反

當我們知道被違反的時候，要特別注意各項指標的改變
