# Representation

這整件事佔了 ML 工作的 75% ，而且攸關成敗

## Feature Engineering

以前在寫程式的時候，我們專注在程式碼上面；但是在 ML 的世界裡，我們要專注在 Representation。一種透過增加或改進 features 來磨練 model 的方式

#### Mapping Raw Data to Features

一般來說，我們拿到 Feature Data 的機會很少；大部分的時間都是拿到 Raw Data ，在下圖我們可以看到，左半邊是從資料來源拿到的 Raw Data，右半邊則是 **Feature vector**，意指包含左邊 Raw data 的浮點數值集合

由於 feature 往往需要乘上 model 的權重，所以我們需要把 feature 都轉換成實數向量才行

![](https://developers.google.com/machine-learning/crash-course/images/RawDataToFeatureVector.svg)

#### Mapping numeric values
因為整數和浮點數的資料已經可以做加權，就不需要特別處理；就把 6 變成 6.0 就好啦～

![](https://developers.google.com/machine-learning/crash-course/images/FloatingPointFeatures.svg)

#### Mapping categorical values

Categorical features 有一個可能值的離散集合。舉例：可能有一個叫 road_name 的 feature 包含 {'忠孝東路', '仁愛路', '信義路', '和平東路'}

因為 model 無法為字串做加權，我們就要透過 feature engineering 把字串轉換成數值，我們會做一個索引表，賦予這些分類索引值，並且加上一個"其他"，當作 **OOV (out-of-vocabulary) bucket** :

* 忠孝東路 : 0
* 仁愛路 : 1
* 信義路 : 2
* 和平東路 : 3
* 其他 : 4

但是，如果我們直接把這些值放進 model，卻會導致一些可能產生問題的限制：

* 我們會學到所有的路都只有同一個權重：但不同的路應該有不同的權重 ( 南陽街比信義路還要便宜 )
* 我們將無法處理 road_name 有很多值的情況：OO 路與 XX 路交叉口

為了解開這些限制，我們可以為每個 categorical features 都做成二元向量：

* 對應到的填 1
* 其他都填 0

這個向量的長度等同於上述索引表中索引值的數量，當只有單個值為 1 的時候，我們稱為 **one-hot encoding**，有多個的時候就稱為 **multi-hot encoding**

如下圖，把 street_name 變成 one-hot encoding 吧！

![](https://developers.google.com/machine-learning/crash-course/images/OneHotEncoding.svg)

one-hot encoding 也可以應用在我們**不想直接加乘**的數值 feature ，像是郵遞區號


#### Sparse Representaion

當 value = 1 的資料很少，零很多的時候，我們可以用 Sparse Representaion - 只儲存非零值的資料 ( 存在索引表 )

## Qualities of Good Features

我們已經知道要怎麼轉換成 feature vector 了，但這只是工作的一部分，現在我們來看看什麼才是好的 feature

#### Avoid rarely used discrete feature values

一個好的 feature values 必須具有非零值，而且在 dataset 裡出現五次以上， 
不然就應該過濾掉

<pre style='background-color:#cdebdd;'><span style="float:right;">✔</span>house_location: Taipei</pre>

是唯一值，沒辦法訓練

<pre style='background-color:#f7d2ce;'><span style="float:right;">✘</span>unique_house_id: TP0012837</pre>

#### Prefer clear and obvious meanings

應該要具有明確而且清晰的意義，這樣我們才能做檢查

<pre style='background-color:#cdebdd;'><span style="float:right;">✔</span>user_age: 18</pre>

<pre style='background-color:#f7d2ce;'><span style="float:right;">✘</span>user_age: 277<br>user_age: 87329421</pre>

#### Don't mix "magic" values with actual data

為了更容易調整和推理，不要使用魔術值，作為替代方案，使用額外的 boolean feature 來表示

<pre style='background-color:#cdebdd;'><span style="float:right;">✔</span>watch_time: 1.082<br>watch_time: 1.2<br>watch_time_is_defined: 1.0</pre>

<pre style='background-color:#f7d2ce;'><span style="float:right;">✘</span>watch_time: -1.0</pre>

在原始 feature 中，可能出現這些 magic values ：

* 有範圍限制 ( 或離散 ) 的變數：觀看 John Wick 的時間一定是從 1 到 電影長度 
* 對於連續變量，要確保遺失的值不會影響到 model，這邊我們就取 feature's data 的平均值


#### Account for upstream instability

feature 不應該隨著時間產生變化，這個變化可能來自於其他的 model 

雖然這邊需要做 one-hot encoding，但是我們可以確保這個值的意義不變

<pre style='background-color:#cdebdd;'><span style="float:right;">✔</span>city_id: "br/sao_paulo"</pre>

但是這個例子，可能隨著不同的資料來源，而有不同的含義：

<pre style='background-color:#f7d2ce;'><span style="float:right;">✘</span>interred_city_cluster: 219</pre>


## Cleaning Data 

想像一下，現在有個鳳梨田，鳳梨田裡不會都是好鳳梨，參雜著中等鳳梨、還可以賣的鳳梨、生病的鳳梨，這時候在把鳳梨賣出去之前，農工們要先把鳳梨分類好，甚至對一些還有機會賣出去的鳳梨做點小加工

作為一個 ML 工程師，我們就是那些農工，要把鳳梨分好，即便只是參雜少量生病的鳳梨，也可能危害到整個鳳梨田

#### Scaling feature values

**Scaling** 意味著把自然數範圍 ( 100 ~ 900 ) 轉換成具備**標準範圍**的浮點數 feature ( 0 ~ 1 或 -1 ~ +1 )，在有多個 feature 要處理的情況下，這麼做會得到幾項好處：

* 幫助 gradient descent 更有效率
* 幫助避免 NaN 陷阱
* 幫助 model 對每個 feature 學習到適當的權重

標準化範圍的方式有兩種：

1. [feature 最小值, feature 最大值] 按照比例轉換成 [-1, +1]
2. 找到每個 feature 值的 [Z score](https://zh.wikipedia.org/wiki/標準分數) : <math>
  <mi>s</mi>
  <mi>c</mi>
  <mi>a</mi>
  <mi>l</mi>
  <mi>e</mi>
  <mi>d</mi>
  <mi>v</mi>
  <mi>a</mi>
  <mi>l</mi>
  <mi>u</mi>
  <mi>e</mi>
  <mo>=</mo>
  <mo stretchy="false">(</mo>
  <mi>v</mi>
  <mi>a</mi>
  <mi>l</mi>
  <mi>u</mi>
  <mi>e</mi>
  <mo>&#x2212;</mo>
  <mi>m</mi>
  <mi>e</mi>
  <mi>a</mi>
  <mi>n</mi>
  <mo stretchy="false">)</mo>
  <mrow class="MJX-TeXAtom-ORD">
    <mo>/</mo>
  </mrow>
  <mi>s</mi>
  <mi>t</mi>
  <mi>d</mi>
  <mi>d</mi>
  <mi>e</mi>
  <mi>v</mi>
  <mo>.</mo>
</math>，使用 Z score 會讓大部分的結果在 -3 ~ +3 之間

![z score](https://upload.wikimedia.org/wikipedia/commons/b/bb/Normal_distribution_and_scales.gif)

#### Handling extreme outliers

我們不採用不理性的離群值，下面這張圖是來自 [California Housing data set](https://developers.google.com/machine-learning/crash-course/california-housing-data-description?authuser=1&refresh=1) 的 roomPerPerson，從這張圖可以看到大部分的人都有 2 ~ 4 個房間，可以有極少數的人有 50 個房間

![](https://developers.google.com/machine-learning/crash-course/images/ScalingNoticingOutliers.svg)

我們要怎麼減少最大限度地這些極端的異常值呢？有幾種方式：

1. 為每一個值取 log ，可惜在這裡效果不是很好
	![](https://developers.google.com/machine-learning/crash-course/images/ScalingLogNormalization.svg)
2. 直接定義一個上限值，並把超過上限的值都當成上限值；狀況好多了，但是右邊多了一座小山  
	![](https://developers.google.com/machine-learning/crash-course/images/ScalingClipping.svg?authuser=1&refresh=1)
3. Binning - 切成小區塊處理

##### Binning

假設我們現在要找到緯度對加州房價的影響，我們發現房價與緯度的高低並沒有線性關聯，但是在特定的區間內，就可以發現維度與房價有很強的關聯，也就是說個別的緯度對於房價可能是很好的預測指標

 ( LA 大概在北緯 34 度；SF 則是在北緯 38 度 ) 
![](https://developers.google.com/machine-learning/crash-course/images/ScalingBinningPart1.svg)

那接下來，我們把維度分成幾個 "bins"，如下圖

![](https://developers.google.com/machine-learning/crash-course/images/ScalingBinningPart2.svg)

所以現在我們新得到了 11 個 feature ( LatitudeBin1, LatitudeBin2, ... LatitudeBin11 )，再透過 one-hot encoding 轉成 feature vector，SF 房子的維度就會變成 :

````
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
```` 

多虧了 Binning ，我們現在可以為不同的緯度給予不同的權重啦～

為了簡單起見，在這個例子，我們用整數來分割，但是為了更準確的辨識，我們可以把每個箱子切得更小，但是前提是資料量夠多。

另外一個方式是用分位數 ( 十分位數、百分位數、中位數等等 ) 來分割，這個方式可以讓每個資料集數量都一樣，而且也不用擔心離群職

##### Scrubbing

可惜，在現實情況我們的資料常常不可靠，因為：

* 被忽略的值 : admin 忘記輸入屋齡
* 重複的樣本 : server 上傳同樣的 log 兩次
* 壞 label : admin 把別墅標籤成公寓
* 壞 feature values : 房價的欄位填成不合理的值

前兩項可以寫程式簡單解決，但是後兩項就很麻煩，除了**畫成圖 ( 直條圖或散佈圖 )**來檢查之外，我們還有其他方式：

* 最大值和最小值
* 平均值和中位數
* 標準差
* 考慮建立一個針對離群 feature 的檢查表，檢查他們的數量

##### Know your data

遵守規則：

* 要知道你的資料應該長什麼樣子
* 驗證資料符合期望 ( 或解釋為什麼不一樣 )
* 重複檢查資料與其他來源是否一致 ( dashboard )
* 謹慎處理資料與程式。好的資料造就好的 ML

分成三步：

1. Visualize : 畫圖
2. Debug : 重複的樣本？缺值？離群值？資料有沒有跟 Dashboard 一致？Trianing Data 和 Validation Data 類似？
3. Monitor : feature 的分位數，一段時間的樣本數量？