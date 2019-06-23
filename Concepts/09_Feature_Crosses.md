# Feature Crosses

## Encoding Nonlinearity

還記得這張圖嗎？

* 藍點：生病的樹
* 橘點：健康的樹

在這張圖，我們可以輕易的找到一條線，用來區分生病的樹和健康的樹，
即便不是 100% 準確，但仍然有很好的效果

![](https://developers.google.com/machine-learning/crash-course/images/LinearProblem1.png)

可是如果情況變成下圖呢？我們就沒辦法找到那一條線了，不管怎麼畫都不會得到好的預測效果，也就是說這是一個非線性問題

![](https://developers.google.com/machine-learning/crash-course/images/LinearProblem2.png)

為了要解決非線性問題，我們會定義一個 additional feature，也可以稱為 synthetic feature 或 feature cross :

<math display="block">
  <msub>
    <mi>x</mi>
    <mn>3</mn>
  </msub>
  <mo>=</mo>
  <msub>
    <mi>x</mi>
    <mn>1</mn>
  </msub>
  <msub>
    <mi>x</mi>
    <mn>2</mn>
  </msub>
</math>

然後我們把這個新定義的 <math>
  <msub>
    <mi>x</mi>
    <mn>3</mn>
  </msub>
</math> 也當做其他的 feature，則新的公式會變成：  

<math display="block">
  <mi>y</mi>
  <mo>=</mo>
  <mi>b</mi>
  <mo>+</mo>
  <msub>
    <mi>w</mi>
    <mn>1</mn>
  </msub>
  <msub>
    <mi>x</mi>
    <mn>1</mn>
  </msub>
  <mo>+</mo>
  <msub>
    <mi>w</mi>
    <mn>2</mn>
  </msub>
  <msub>
    <mi>x</mi>
    <mn>2</mn>
  </msub>
  <mo>+</mo>
  <msub>
    <mi>w</mi>
    <mn>3</mn>
  </msub>
  <msub>
    <mi>x</mi>
    <mn>3</mn>
  </msub>
</math>
現在演算法也可以學到 <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msub>
    <mi>w</mi>
    <mn>3</mn>
  </msub>
</math> 的權重啦 ~

#### Kinds of feature crosses

我們可以建立不同種類的 feature crosses，舉例來說 :

* ``[A X B]`` : 透過兩個 feature 值相乘而構成的 feature crosses
* ``[A x B x C x D x E]`` : 透過五個 feature 值相乘而構成的 feature crosses
* ``[A X A]`` : 透過對單個 feature 值做平方而構成的 feature crosses
* 當 A 跟 B 是 boolean 值 (經過 one-hot encoding) 的時候，產生的 cross 可能非常地稀疏

## Crossing One-Hot Vectors

在前一節，我們集中討論關於獨立浮點數 feature 的 feature-crossing。可是在實務上，ML 的學習 model 沒什麼機會可以直接針對連續性 feature 做 cross，更常出現的情況是：針對 one-hot feature vectors 的 feature cross

我們把 one-hot feature 的 feature crosses 當成 logical conjunctions。如果我們要預測房價，則可能會針對緯度與房子坪數做交叉，因為房子的價錢同時與位置和大小有關：

````
北緯30度 AND 30坪
````

另外一個例子，假設我們進一步的經度和緯度都做 Binning ，各自產生了包含五個元素的 one-hot vector

````
binned_latitude = [0, 0, 0, 1, 0]
binned_longitude = [0, 1, 0, 0, 0]
````

在假設我們定義了 binned_latitude, binned_longitude 的feature cross：

````
binned_latitude X binned_longitude
````

這個 feature cross 是一個有 25 個元素的 one-hot vector ( 24 個 0 和 1 個 1 ) ，這一個 1 在 cross 中表達了經緯度的 logical conjunctions。讓 model 可以學習有關這個 conjunction 的特定關係

換一個比較"人話"的方式說明，現在假設我們有一些指定範圍的經度和緯度：

````
binned_latitude(lat) = [
  0  < lat <= 10
  10 < lat <= 20
  20 < lat <= 30
]

binned_longitude(lon) = [
  0  < lon <= 15
  15 < lon <= 30
]
````

建立 feature cross ，意味著我們會得到：

````
binned_latitude_X_longitude(lat, lon) = [
  0  < lat <= 10 AND 0  < lon <= 15
  0  < lat <= 10 AND 15 < lon <= 30
  10 < lat <= 20 AND 0  < lon <= 15
  10 < lat <= 20 AND 15 < lon <= 30
  20 < lat <= 30 AND 0  < lon <= 15
  20 < lat <= 30 AND 15 < lon <= 30
]
````

現在我們來試試看用兩個 feature 來預測狗主人對狗狗的滿意度：

* 狗狗的行為 ( 吠叫、哭叫、撒嬌...等等 )
* 發生的時間

如果我們對這兩個 feature 做 feature cross :

````
[behavior type X time of day]
````

然後我們終究會得到一個比任何一個 feature 都強大的預測能力，舉例來說：
『在下午 5 點 ( 主人剛回到家時 ) 的哭叫』將可能是一個對於狗主人滿意度的正向預測；『在凌晨 1 點 ( 主人睡覺時 ) 的哭叫』將可能是一個對於狗主人滿意度的反向預測