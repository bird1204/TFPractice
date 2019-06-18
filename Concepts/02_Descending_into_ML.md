# Descending into ML

## Linear Regression - 線性迴歸
線性回歸，就是找**斜率**：  

<math display="block">
  <mi>y</mi>
  <mo>=</mo>
  <mi>m</mi>
  <mi>x</mi>
  <mo>+</mo>
  <mi>b</mi>
</math>

以下圖為例：
![Google Example 1](https://developers.google.com/machine-learning/crash-course/images/CricketLine.svg?authuser=1&refresh=1)
<math display="block">
  <mi>y</mi>
  <mo>=</mo>
  <mi>m</mi>
  <mi>x</mi>
  <mo>+</mo>
  <mi>b</mi>
</math>
在這邊 y, m, x, b 的意義（ 如果還不懂就 Google ）：  

* y : 攝氏溫度 - 要求的值
* m : 斜率
* x : 每分鐘蟋蟀啾聲的次數 - 輸入的值
* b : y 的截距

但是按照在 ML 裡的慣例，這個等式會變成：

<math display="block">
  <msup>
    <mi>y</mi>
    <mo>&#x2032;</mo>
  </msup>
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
</math>

在這邊 y, m, x, b 的意義（ 如果還不懂就 Google ）：  

* <math><msup>
    <mi>y</mi>
    <mo>&#x2032;</mo>
  </msup></math> : 要預測的 label
* <math><mi>b</mi></math> : y 的截距，有時候可以對照到 <math><msub><mi>w</mi><mn>0</mn></msub></math>
* <math><msub>
    <mi>w</mi>
    <mn>1</mn>
  </msub></math> : feature 1 的權重，一樣是斜率 <math><mi>m</mi></math> 的概念
* <math><msub>
    <mi>x</mi>
    <mn>1</mn>
  </msub></math> : feature
  
在只有一個 feature 的情況下，只需要替換 <math><msub><mi>x</mi><mn>1</mn></msub></math> 就可以找到 <math><msup><mi>y</mi><mo>&#x2032;</mo></msup></math> ，而更複雜的 model 可能意味著有複數個 feature 及各個 feature 的權重，這時候公式也要隨著更動：

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msup>
    <mi>y</mi>
    <mo>&#x2032;</mo>
  </msup>
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

## Traning and Loss

**Traning** 一個 model，其實就是用 labeled example 找出那條線 ( 上面那條藍色的線 )，這個過程我們稱為 **empirical risk minimization**  
**Loss** 則可以視為壞預測的處罰，是那條線與實際情況的誤差


從下圖來看，左圖是高 Loss 的模型，右圖是低 Loss 的模型
![Google Example 2](https://developers.google.com/machine-learning/crash-course/images/LossSideBySide.png?authuser=1&refresh=1)

為了解決這個問題，損失函數登場！！！

### 常見的函數：Squared loss ( a.k.a <math><msub><mi>L</mi><mn>2</mn></msub></math>loss )

想法是這樣子：

````
  = the square of the difference between the label and the prediction  
  = (observation - prediction(x))2
  = (y - y')2
````

**均方誤差 Mean square error (MSE)** 是指参數估計值與參數真值之差平方的期望值：

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>M</mi>
  <mi>S</mi>
  <mi>E</mi>
  <mo>=</mo>
  <mfrac>
    <mn>1</mn>
    <mi>N</mi>
  </mfrac>
  <munder>
    <mo>&#x2211;<!-- ∑ --></mo>
    <mrow class="MJX-TeXAtom-ORD">
      <mo stretchy="false">(</mo>
      <mi>x</mi>
      <mo>,</mo>
      <mi>y</mi>
      <mo stretchy="false">)</mo>
      <mo>&#x2208;<!-- ∈ --></mo>
      <mi>D</mi>
    </mrow>
  </munder>
  <mo stretchy="false">(</mo>
  <mi>y</mi>
  <mo>&#x2212;<!-- − --></mo>
  <mi>p</mi>
  <mi>r</mi>
  <mi>e</mi>
  <mi>d</mi>
  <mi>i</mi>
  <mi>c</mi>
  <mi>t</mi>
  <mi>i</mi>
  <mi>o</mi>
  <mi>n</mi>
  <mo stretchy="false">(</mo>
  <mi>x</mi>
  <mo stretchy="false">)</mo>
  <msup>
    <mo stretchy="false">)</mo>
    <mn>2</mn>
  </msup>
</math>

其中：

* <math xmlns="http://www.w3.org/1998/Math/MathML"><mo stretchy="false">(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo stretchy="false">)</mo></math> 是 example 裡：
	* <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>x</mi>
</math> 是模型用來預測的一組 features
	* <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>y</mi>
</math> 是 example 的 label
* <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>p</mi>
  <mi>r</mi>
  <mi>e</mi>
  <mi>d</mi>
  <mi>i</mi>
  <mi>c</mi>
  <mi>t</mi>
  <mi>i</mi>
  <mi>o</mi>
  <mi>n</mi>
  <mo stretchy="false">(</mo>
  <mi>x</mi>
  <mo stretchy="false">)</mo>
</math> 是 features 中權重跟截距的組合
* <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>D</mi>
</math> 是包含 label 的資料集
* <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>N</mi>
</math> 是 <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>D</mi>
</math> 的數量

雖然 MSE 是常見的損失函數，但他不是唯一的損失函數，也不一定適合套用在所有的情況


# 練習 MSE

求下圖的 MSE 為多少？

![Google Example 3](https://developers.google.com/machine-learning/crash-course/images/MCEDescendingIntoMLLeft.png?authuser=1&refresh=1)

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>M</mi>
  <mi>S</mi>
  <mi>E</mi>
  <mo>=</mo>
  <mfrac>
    <mrow>
      <msup>
        <mn>0</mn>
        <mn>2</mn>
      </msup>
      <mo>+</mo>
      <msup>
        <mn>1</mn>
        <mn>2</mn>
      </msup>
      <mo>+</mo>
      <msup>
        <mn>0</mn>
        <mn>2</mn>
      </msup>
      <mo>+</mo>
      <msup>
        <mn>1</mn>
        <mn>2</mn>
      </msup>
      <mo>+</mo>
      <msup>
        <mn>0</mn>
        <mn>2</mn>
      </msup>
      <mo>+</mo>
      <msup>
        <mn>1</mn>
        <mn>2</mn>
      </msup>
      <mo>+</mo>
      <msup>
        <mn>0</mn>
        <mn>2</mn>
      </msup>
      <mo>+</mo>
      <msup>
        <mn>1</mn>
        <mn>2</mn>
      </msup>
      <mo>+</mo>
      <msup>
        <mn>0</mn>
        <mn>2</mn>
      </msup>
      <mo>+</mo>
      <msup>
        <mn>0</mn>
        <mn>2</mn>
      </msup>
    </mrow>
    <mn>10</mn>
  </mfrac>
  <mo>=</mo>
  <mn>0.4</mn>
</math>
