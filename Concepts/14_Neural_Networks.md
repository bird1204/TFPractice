# Neural Networks

## Structure

我們已經知道了下面這張圖不是線性問題

![](https://developers.google.com/machine-learning/crash-course/images/FeatureCrosses1.png)

非線性問題的意思是沒辦法用 <math xmlns="http://www.w3.org/1998/Math/MathML">
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
</math> 進行準確的預測，換言之，decision surface 並不是一條線。在之前的章節，我們講 feature cross 視為一種可能解決非線性問題的方法

現在假設我們有這樣的 dataset :

![](https://developers.google.com/machine-learning/crash-course/images/NonLinearSpiral.png)

很明顯沒辦法用任何 linear model 解決這題，這時候神經網路上場～

為了要了解 neural networks 如何協助解決非線性問題，我們先把 linear model 畫成這張圖 : 

![](https://developers.google.com/machine-learning/crash-course/images/linear_net.svg)

每個藍點表示一個 input feature，綠點表示 inputs 的加權總和

我們要怎麼讓這個 model 進化到可以解決 nonlinear problems 呢？

### Hidden Layers

我們加上 Hidden Layers 當作中間值，hidden layer 中的黃點表示藍點 inputs 的加權總和

![](https://developers.google.com/machine-learning/crash-course/images/1hidden.svg)

現在這個 model 是線性的嗎？是的——output 還是 inputs 的線性組合

那我們再加一層 hidden layers

![](https://developers.google.com/machine-learning/crash-course/images/2hidden.svg)

現在這個 model 是線性的嗎？是的——即便我們添加了任意多的 hidden layer，所有線性函數的組合依然是線性函數

### Activation Functions

為了解決非線性問題，我們應該要直接引入非線性

![](https://developers.google.com/machine-learning/crash-course/images/activation.svg)

現在我們加入了 Activation Functions，加進 layer 會有更多影響。在非線性上堆疊非線性，讓我們的 model 在 input 與 預測的 output 間擁有更複雜的關係。簡單來說，每一個 layer 都從 raw inputs 有效地學習更複雜、更高級的功能

### Common Activation Functions

**sigmoid** 把加權和轉換成 0 ~ 1 的值

<math display="block">
  <mi>F</mi>
  <mo stretchy="false">(</mo>
  <mi>x</mi>
  <mo stretchy="false">)</mo>
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
          <mi>x</mi>
        </mrow>
      </msup>
    </mrow>
  </mfrac>
</math>

![](https://developers.google.com/machine-learning/crash-course/images/sigmoid.svg)

**rectified linear unit activation function ( ReLU )** 通常比 smooth function 更好一些，也更好計算

<math display="block">
  <mi>F</mi>
  <mo stretchy="false">(</mo>
  <mi>x</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mi>m</mi>
  <mi>a</mi>
  <mi>x</mi>
  <mo stretchy="false">(</mo>
  <mn>0</mn>
  <mo>,</mo>
  <mi>x</mi>
  <mo stretchy="false">)</mo>
</math>

![](https://developers.google.com/machine-learning/crash-course/images/relu.svg)

事實上，所有數學函數都可以用來當 activation function 。假設 <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>&#x03C3;<!-- σ --></mi>
</math> 代表 activation function，則求網路中的節點值的公式 : 

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>&#x03C3;<!-- σ --></mi>
  <mo stretchy="false">(</mo>
  <mi mathvariant="bold-italic">w</mi>
  <mo>&#x22C5;<!-- ⋅ --></mo>
  <mi mathvariant="bold-italic">x</mi>
  <mo>+</mo>
  <mi>b</mi>
  <mo stretchy="false">)</mo>
</math>

TensorFlow 提供 out-of-the-box 支援這種 activation function。不過還是推薦使用 ReLU

### Summary

現在我們的 model 終於有了 neural network 的樣子了：

* 一組 node，類似神經元，按照 layer 組織
* 一組 weight 表示每個 neural network 層與下層的連結
* 一組 bias， 每個 node 有一個
* activation function，轉換 layer 中每個節點的輸出，不同的 layer 可以有不同的 activation function