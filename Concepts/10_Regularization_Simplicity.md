# Regularization Simplicity

## Regularization for Simplicity: L₂ Regularization

下圖的曲線，代表了迭代學習與 Loss 的關係：

![](https://developers.google.com/machine-learning/crash-course/images/RegularizationTwoLossFunctions.svg)

訓練越多次，Train Data 越準，可以 Validation Data 的 Loss 卻變高了，這個現象就是 model overfitting 了，防止 overfitting 有兩個方式：

1. 強制暫停：在 Validation Data 最準的時候暫停 ( 很難做到 )
2. 懲罰過於複雜的 model

不再只是單純的做最小化 Loss ( empirical risk minimization ) - 獲取正確的樣本 :

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtext>minimize(Loss(Data|Model))</mtext>
</math>

現在我們還要懲罰複雜的 model :

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtext>minimize(Loss(Data|Model) + complexity(Model))</mtext>
</math>

我們的演算法，現在有兩個功能: **loss term**, **regularization term**
在 ML 速成裡，我們用兩種方式來衡量一個 model 的複雜度 :  

* model 中所有 feature 的權重
* model 中所有非零權重 feature 的數量

如果複雜度是 model 中所有 feature 的權重，那權重的絕對值越高， model 越複雜，而且可以用  **L₂ regularization** 來量化複雜度 :

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msub>
    <mi>L</mi>
    <mn>2</mn>
  </msub>
  <mtext>&#xA0;regularization term</mtext>
  <mo>=</mo>
  <mrow class="MJX-TeXAtom-ORD">
    <mo stretchy="false">|</mo>
  </mrow>
  <mrow class="MJX-TeXAtom-ORD">
    <mo stretchy="false">|</mo>
  </mrow>
  <mi mathvariant="bold-italic">w</mi>
  <mrow class="MJX-TeXAtom-ORD">
    <mo stretchy="false">|</mo>
  </mrow>
  <msubsup>
    <mrow class="MJX-TeXAtom-ORD">
      <mo stretchy="false">|</mo>
    </mrow>
    <mn>2</mn>
    <mn>2</mn>
  </msubsup>
  <mo>=</mo>
  <mrow class="MJX-TeXAtom-ORD">
    <msubsup>
      <mi>w</mi>
      <mn>1</mn>
      <mn>2</mn>
    </msubsup>
    <mo>+</mo>
    <msubsup>
      <mi>w</mi>
      <mn>2</mn>
      <mn>2</mn>
    </msubsup>
    <mo>+</mo>
    <mo>.</mo>
    <mo>.</mo>
    <mo>.</mo>
    <mo>+</mo>
    <msubsup>
      <mi>w</mi>
      <mi>n</mi>
      <mn>2</mn>
    </msubsup>
  </mrow>
</math>

舉例來說，一個線性 model 有下列權重：

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mo fence="false" stretchy="false">{</mo>
  <msub>
    <mi>w</mi>
    <mn>1</mn>
  </msub>
  <mo>=</mo>
  <mn>0.2</mn>
  <mo>,</mo>
  <msub>
    <mi>w</mi>
    <mn>2</mn>
  </msub>
  <mo>=</mo>
  <mn>0.5</mn>
  <mo>,</mo>
  <msub>
    <mi>w</mi>
    <mn>3</mn>
  </msub>
  <mo>=</mo>
  <mn>5</mn>
  <mo>,</mo>
  <msub>
    <mi>w</mi>
    <mn>4</mn>
  </msub>
  <mo>=</mo>
  <mn>1</mn>
  <mo>,</mo>
  <msub>
    <mi>w</mi>
    <mn>5</mn>
  </msub>
  <mo>=</mo>
  <mn>0.25</mn>
  <mo>,</mo>
  <msub>
    <mi>w</mi>
    <mn>6</mn>
  </msub>
  <mo>=</mo>
  <mn>0.75</mn>
  <mo fence="false" stretchy="false">}</mo>
</math>

L₂ regularization 的結果是 26.915

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msubsup>
    <mi>w</mi>
    <mn>1</mn>
    <mn>2</mn>
  </msubsup>
  <mo>+</mo>
  <msubsup>
    <mi>w</mi>
    <mn>2</mn>
    <mn>2</mn>
  </msubsup>
  <mo>+</mo>
  <msubsup>
    <mi mathvariant="bold-italic">w</mi>
    <mn mathvariant="bold">3</mn>
    <mn mathvariant="bold">2</mn>
  </msubsup>
  <mo>+</mo>
  <msubsup>
    <mi>w</mi>
    <mn>4</mn>
    <mn>2</mn>
  </msubsup>
  <mo>+</mo>
  <msubsup>
    <mi>w</mi>
    <mn>5</mn>
    <mn>2</mn>
  </msubsup>
  <mo>+</mo>
  <msubsup>
    <mi>w</mi>
    <mn>6</mn>
    <mn>2</mn>
  </msubsup>
</math><br>
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mo>=</mo>
  <msup>
    <mn>0.2</mn>
    <mn>2</mn>
  </msup>
  <mo>+</mo>
  <msup>
    <mn>0.5</mn>
    <mn>2</mn>
  </msup>
  <mo>+</mo>
  <msup>
    <mn mathvariant="bold">5</mn>
    <mn mathvariant="bold">2</mn>
  </msup>
  <mo>+</mo>
  <msup>
    <mn>1</mn>
    <mn>2</mn>
  </msup>
  <mo>+</mo>
  <msup>
    <mn>0.25</mn>
    <mn>2</mn>
  </msup>
  <mo>+</mo>
  <msup>
    <mn>0.75</mn>
    <mn>2</mn>
  </msup>
</math><br>
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mo>=</mo>
  <mn>0.04</mn>
  <mo>+</mo>
  <mn>0.25</mn>
  <mo>+</mo>
  <mn mathvariant="bold">25</mn>
  <mo>+</mo>
  <mn>1</mn>
  <mo>+</mo>
  <mn>0.0625</mn>
  <mo>+</mo>
  <mn>0.5625</mn>
</math><br>
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mo>=</mo>
  <mn>26.915</mn>
</math>

我們可以看到 <math xmlns="http://www.w3.org/1998/Math/MathML">
  <msub>
    <mi>w</mi>
    <mn>3</mn>
  </msub>
</math> 超級重的，佔了 25

## Lambda

實務上，我們用的將值乘上 Lambda ( 也稱為 regularization rate ) :

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtext>minimize(Loss(Data|Model)</mtext>
  <mo>+</mo>
  <mi>&#x03BB;<!-- λ --></mi>
  <mtext>&#xA0;complexity(Model))</mtext>
</math>

執行 L₂ regularization 會得到這些效果 :

* 鼓勵權重趨近於 0
* 鼓勵權重的平均趨近於 0 ( 變成常態分佈 )

lambda 值越高，圖形會越集中，反之；那我們要如何選擇 lambda 值呢？目標是簡單與 train data 拟合間的平衡：

* 太高 -> 太簡單 -> 資料不足
* 太低 -> 太複雜 -> over-fitting