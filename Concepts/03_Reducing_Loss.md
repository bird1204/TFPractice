# Reducing Loss

## An Iterative Approach

我們在 02_Descending_into_ML 已經瞭解了 Loss 的觀念，現在我們來看看 ML 如何迭代地減少損失

還記得『猜售價』怎麼玩嗎～  
魔王會問：請問橡皮擦賣多少錢？
這時候我們要隨便猜一個數字，接著魔王會告訴我們：數字要猜多一點還是少一點  
猜**最少次**而且猜到**正確售價**的人獲勝！

長大以後，我們都知道玩這個遊戲有幾個心法：

1. 第一次要盡可能猜準
2. 剛開始猜的數字間隔要夠大，用來找到有可能區間
3. 找到區間以後，猜數字的間隔要越來越小

````
嘗試找到一個最有效率地猜售價的方式
````

下面這張圖，就顯示了 ML 演算法用來訓練模型的 iterative trial-and-error 流程
![GoogleExample](https://developers.google.com/machine-learning/crash-course/images/GradientDescentDiagram.svg?authuser=1&refresh=1)

還記得 ML 的公式嗎？ 
 
<math display="block"><msup><mi>y</mi><mo>&#x2032;</mo></msup><mo>=</mo><mi>b</mi><mo>+</mo><msub><mi>w</mi><mn>1</mn></msub><msub><mi>x</mi><mn>1</mn></msub></math>

我們的 <math><mi>b</mi></math> 和 <math>
  <msub>
    <mi>w</mi>
    <mn>1</mn>
  </msub>
</math> 的初始值應該要設多少呢？在普遍的線性迴歸問題裡，初始值並沒有這麼重要，我們可以隨機假設，這邊就假設：

* <math><mi>b</mi><mo>=</mo><mn>0</mn></math>
* <math><msub><mi>w</mi><mn>1</mn></msub><mo>=</mo><mn>0</mn></math> 

再假設第一個 feature (<math><msub><mi>x</mi><mn>1</mn></msub></math>) 的值是 10，我們會得到：

````
  y' = 0 + 0(10)
  y' = 0
````

當 model 在計算損失的時候，其實是用**損失函數**來處理的，假設使用  squared loss function 來計算，我們就會需要兩個 input :

* y' : model 對 feature x 的預測
* y : 與 feature x 對應的正確 label

最後我們會進入 Compute parameter updates 的階段，意味著 ML 系統已經找到了 <math><mi>b</mi><mo>=</mo><mn>0</mn></math> 和 <math><msub><mi>w</mi><mn>1</mn></msub></math> 的值

現在 ML 已經完成了第一次迭代，接下來他持續的迭代並試著找到損失最低的參數，大多數的情況，我們會持續的迭代，直到整體的損失停止變化或變得很緩慢為止，這種情況我們稱 model 已經 converged.

## 梯度下降 - Gradient Descent
在 Iterative Approach 我們為了解釋流程創造名詞：Compute parameter updates，現在我們要用更實質的東西來取代他

如果我們有時間計算出所有的損失，就可能會得到下面這張圖：
![Google example 2](https://developers.google.com/machine-learning/crash-course/images/convex.svg)

也就是說，一定會有個最小值 ( 凸點 )，或者是說**斜率為零**的點，可是計算所有損失再找到這個點太沒有效率了，在 ML 中，常常使用 **gradient descent** 來解決這個問題：

1. 一樣隨便選一個點，並計算該點的梯度(損失的梯度等於曲線的斜率)
2. 梯度是向量，具有方向性、大小；我們往負梯度的方向挑另外一個點
3. 持續 1, 2 直到找到最小值

## Learning Rate

我們學會了梯度下降，可以要怎麼知道下一個點要離這個點多遠呢？Learning Rate 就是我們用來告訴 ML 下一個點要跨多遠的參數，舉例來說，梯度 2.5、Learning Rate 0.01 的情況下，下一個點會距離 0.025

Hyperparameters 是工程師在機器學習演算法中調整的旋鈕。大多數機器學習工程師花費相當多的時間來調整學習率。如果你選擇的 Learning Rate 太小，學習時間會太長；如果 Learning Rate 太大，則可能跨過最小值；調整到恰到好處才是正道 - 如果損失的梯度很小，則 Learning Rate 可以大一點；如果損失的梯度很大，則 Learning Rate 就要小一點。

一維的問題的建議值是： <math xmlns="http://www.w3.org/1998/Math/MathML">
  <mfrac>
    <mn>1</mn>
    <mrow>
      <mi>f</mi>
      <mo stretchy="false">(</mo>
      <mi>x</mi>
      <msup>
        <mo stretchy="false">)</mo>
        <mo>&#x2033;</mo>
      </msup>
    </mrow>
  </mfrac>
</math>  
二維或更多維度的問題建議值則是：[Hessian](https://zh.wikipedia.org/wiki/黑塞矩陣) 的倒數

## Stochastic Gradient Descent (SGD)

在前面，我們都假設已經處理了所有的資料，但是實務上這樣的方式還是但沒有效率了，那如果我們能夠從平均值得到正確的梯度，以減少大部分運算呢？

在更新參數的時候:

* GD 我們是一次用全部訓練集的數據去計算損失函數的梯度就更新一次參數。
* SGD 就是一次跑一個樣本或是小批次(mini-batch)樣本然後算出一次梯度或是小批次梯度的平均後就更新一次，那這個樣本或是小批次的樣本是隨機抽取的，所以才會稱為隨機梯度下降法。
* mini-batch SGD 是 GD 跟 SGD 的折衷，隨機選擇 10 ~ 1,000 個 example 做處理