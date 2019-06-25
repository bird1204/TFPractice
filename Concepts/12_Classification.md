# Logistic Regression

## Calculating a Probability
有很多問題是需要輸出可能性的，邏輯迴歸在處理這類問題上非常有效率！
現在假設我們有一個邏輯迴歸的 model 是用來預測狗狗在半夜會叫幾次的 :  

````
p(bark | night)
````

假設 model 預測是 0.05 次，我們會得到狗主人在一年內可能被嚇醒 18 次 :

````
startled = p(bark | night) * nights
18 ~= 0.05 * 365
````

很多情況，我們會把邏輯回歸用在二元分類的問題，正確的預測 feature 的label 屬於兩個可能 label 中的哪一個

所以邏輯迴歸 model 是怎麼把 output 都落在 0 ~ 1 呢？ 使用 sigmoid 函數，產生具有相同特徵的 output : 

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>y</mi>
  <mo>=</mo>
  <mfrac>
    <mn>1</mn>
    <mrow>
      <mn>1</mn>
      <mo>+</mo>
      <msup>
        <mi>e</mi>
        <mrow class="MJX-TeXAtom-ORD">
          <mo>&#x2212;<!-- − --></mo>
          <mi>z</mi>
        </mrow>
      </msup>
    </mrow>
  </mfrac>
</math>

sigmoid 可以畫出這張圖 : 
![](https://developers.google.com/machine-learning/crash-course/images/SigmoidFunction.png)

如果``z``是邏輯迴歸 model 在線性 layer 的 output，那 sigmoid(z) 就會落在 0 ~ 1 之間 : 

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msup>
    <mi>y</mi>
    <mo>&#x2032;</mo>
  </msup>
  <mo>=</mo>
  <mfrac>
    <mn>1</mn>
    <mrow>
      <mn>1</mn>
      <mo>+</mo>
      <msup>
        <mi>e</mi>
        <mrow class="MJX-TeXAtom-ORD">
          <mo>&#x2212;<!-- − --></mo>
          <mo stretchy="false">(</mo>
          <mi>z</mi>
          <mo stretchy="false">)</mo>
        </mrow>
      </msup>
    </mrow>
  </mfrac>
</math>

其中 : 

* y' 邏輯迴歸 model 針對特定 example 的 ouput
* z is b + w1x1 + w2x2 + ... wNxN
	* The w 是 model 學到的權重； b 是截距 
	* The x 是 特定 example 的 feature 值

注意 z 也被稱為對數概率 ( *log-odds* ) :

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>z</mi>
  <mo>=</mo>
  <mi>l</mi>
  <mi>o</mi>
  <mi>g</mi>
  <mo stretchy="false">(</mo>
  <mfrac>
    <mi>y</mi>
    <mrow>
      <mn>1</mn>
      <mo>&#x2212;<!-- − --></mo>
      <mi>y</mi>
    </mrow>
  </mfrac>
  <mo stretchy="false">)</mo>
</math>

![](https://developers.google.com/machine-learning/crash-course/images/LogisticRegressionOutput.svg)

現在來做做看，假設 :

* b = 1
* w1 = 2
* w2 = -1
* w3 = 5

然後有三個 feature :

* x1 = 0
* x2 = 10
* x3 = 2

因為 log-odds :

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
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

我們得到 :


```` 
(1) + (2)(0) + (-1)(10) + (5)(2) = 1 
````

最終答案等於 0.731 :

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msup>
    <mi>y</mi>
    <mo>&#x2032;</mo>
  </msup>
  <mo>=</mo>
  <mfrac>
    <mn>1</mn>
    <mrow>
      <mn>1</mn>
      <mo>+</mo>
      <msup>
        <mi>e</mi>
        <mrow class="MJX-TeXAtom-ORD">
          <mo>&#x2212;<!-- − --></mo>
          <mo stretchy="false">(</mo>
          <mn>1</mn>
          <mo stretchy="false">)</mo>
        </mrow>
      </msup>
    </mrow>
  </mfrac>
  <mo>=</mo>
  <mn>0.731</mn>
</math>

## Loss and Regularization

在前面我們學到線性迴歸的損失函數是 squared loss ，現在告訴你，邏輯迴歸的是 **Log Loss** :

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtext>Log Loss</mtext>
  <mo>=</mo>
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
  <mo>&#x2212;<!-- − --></mo>
  <mi>y</mi>
  <mi>log</mi>
  <mo>&#x2061;<!-- ⁡ --></mo>
  <mo stretchy="false">(</mo>
  <msup>
    <mi>y</mi>
    <mo>&#x2032;</mo>
  </msup>
  <mo stretchy="false">)</mo>
  <mo>&#x2212;<!-- − --></mo>
  <mo stretchy="false">(</mo>
  <mn>1</mn>
  <mo>&#x2212;<!-- − --></mo>
  <mi>y</mi>
  <mo stretchy="false">)</mo>
  <mi>log</mi>
  <mo>&#x2061;<!-- ⁡ --></mo>
  <mo stretchy="false">(</mo>
  <mn>1</mn>
  <mo>&#x2212;<!-- − --></mo>
  <msup>
    <mi>y</mi>
    <mo>&#x2032;</mo>
  </msup>
  <mo stretchy="false">)</mo>
</math>

其中：

 * <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mo stretchy="false">(</mo>
  <mi>x</mi>
  <mo>,</mo>
  <mi>y</mi>
  <mo stretchy="false">)</mo>
  <mo>&#x2208;<!-- ∈ --></mo>
  <mi>D</mi>
</math> 是包含了成對 labeled examples 的資料集
 pairs.
 
 * <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>y</mi>
</math> labeled example 的 label，因為是邏輯回歸 <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>y</mi>
</math> 要在 0 ~ 1
 
 * <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msup>
    <mi>y</mi>
    <mo>&#x2032;</mo>
  </msup>
</math> is feature <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>x</mi>
</math> 的預測值，也是 0 ~ 1

#### Regularization in Logistic Regression

Regularization 在邏輯回歸極其重要，如果沒有正規化，在高維度的損失將趨近於零，所以常用有兩種方式來抑制複雜度：L2, 提早停止
