# Training Neural Networks

## Best Practices

本章解釋反向傳播 ( backpropagation ) 的失敗案例，以及正規化神經網路的常用方法

### Failure Cases

有很多讓反向傳播錯誤的方式

##### Vanishing Gradients

較低層 ( 靠近 inputs ) 的梯度可以變得很小, 在 deep network 裡，計算這些梯度可能會導致得到許多小項的乘積

當低層的梯度消失時 ( 趨近於 0 )，這些 layer 的訓練會變得很慢，或是根本沒效

把 ReLu 當作 activation function 可以防止這件事

##### Exploding Gradients

如果網路中的權重非常大，則會導致我們得到許多大項的乘積。在這種情況下，我們可能會遇到 Exploding Gradients : 梯度太大，大到無法收斂

Batch normalization 可以防止 Exploding Gradients，也可以降低 learning rate

##### Dead ReLU Units

一旦 ReLU 的權重和小於 0, ReLU 就會卡住

降低 learning rate 可以防止這件事

### Dropout Regularization

還有一種正規化的方式叫做 dropout ，對 NN 很有用，是透過隨機的 "droping out" 一些 unit 達成的。我們可以定義 0 - 1 之間的捨棄比例：

* 0.0 = 不執行Dropout Regularization
* 1.0 = 通通丟掉，model學不到任何東西
* 在0~1之間會比較有用