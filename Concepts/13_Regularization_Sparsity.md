# Regularization: Sparsity

## L₁ Regularization

稀疏向量通常包含很多維度，在建立 feature cross 的時候會產生更多維度，這可能導致 model 變得過於巨大，而且需要大量的 RAM

在高維度的稀疏向量中，鼓勵權重在可能的情況下降低到 0 是一件好事。把權重下降到 0 基本上等同於從 model 中將對應的 feature 移除。這樣的方法可以節省 RAM 而且減少 model 中的 noise

舉例來說，假設我們現在有一個涵蓋全球住房的數據，想要推測地區與房價的關係。
可是全球有七成的面積是海洋，也就是有七成的經緯度是不需要被儲存的，鼓勵這些無意義的權重為 0 ，可以避免我們在訓練期間支付額外的成本

我們可以用正規化，或者是說我們已經學到正規化負責處理這件事～
不幸的是，先前學到的 L2 Regularization 只負責把權重變小，而不是把權重歸零

另外一種想法是 L0 Regularization，懲罰 model 中非零系数值的數量，這種方法直觀，可惜沒有凸性、難以優化，因此沒有辦法用實務上。

然而，我們還有 L₁ Regularization ，有凸性的 L0 ~

### L1 vs L2 regularization.

L2 and L1 懲罰的方式不同 :

* L2 懲罰 <math>
 <msup>
  <mi>weight</mi>
  <mn>2</mn>
 </msup>
</math>
* L1 懲罰 <math>
  <mo>|</mo>
  <mrow>
  	<mi>weight</mi>
  </mrow>
  <mo>|</mo>
</math>

因此，L2 and L1 也有不同的 derivatives :

* L2 的是 <math>
  <mn>2</mn>
  <mo>*</mo>
  <mi>weight</mi>
</math>
* L1 的是 <math>
  <mrow>
  	<mi>k</mi>
  </mrow>
</math> ( 常數，其值與權重無關 )

我們可以把 L2 的 derivatives 想成消除 x% 的力，就算做了一百萬次的 99% 也不會等於零；L1 的 derivatives 想成消除 |x| 的力，因此在減到負數的時候會變成 0

L₁ Regularization - 懲罰所有權重的絕對值 - 對於寬
model 是非常有效的。

注意：這些敘述適用於一維模型


