# -*- coding: UTF-8 -*-
# 訓練一個能夠分辨服裝種類的神經網路
from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# 載入內含 10 個種類, 70,000筆灰階服裝圖片的資料集，其中 60,000 拿來訓練，10,000 拿來測試
# 註：實際上分成四個檔案，可以想像成四個大圖檔，每個圖檔包含很多的小圖
# 註：因為這個檔案內含在 TF 裡，所以直接呼叫即可
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 從 mnist 的 github 可以知道服裝分類的對照
# https://github.com/zalandoresearch/fashion-mnist
# 宣告對應的 class_name
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Explore the data
# 會得到 (60000, 28, 28), 有 60,000 張 28*28 的圖
print(train_images.shape)
# 會得到 60000
print(len(train_labels))
# 每個 label 都介於 0-9
print(train_labels)
# 會得到 (10000, 28, 28), 有 60,000 張 28*28 的圖
print(test_images.shape)
# 會得到 10000
print(len(test_labels))
# 每個 label 都介於 0-9
print(test_labels)

# Preprocess the data
# 要把小圖切出來，先檢查每個小圖的pixel，用第一張圖檢查
# plt.show() 之後會發現每個服裝小圖pixel 都介於 0-255
# plt.figure()
# plt.imshow(train_images[1])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# 把圖片切出來
train_images = train_images / 255.0
test_images = test_images / 255.0

# 顯示前 25 張，檢查切得對不對，確認後開始訓練
plt.figure(figsize=(10,10))
for i in range(25):
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(train_images[i], cmap=plt.cm.binary)
  plt.xlabel(class_names[train_labels[i]])

plt.show()

# Setup the layers
# layer 直翻為層，用途是對資料集加上一層『處理』，並且希望加上這層處理，我們可以得到有意義的結果
# 大部分的 layer, 都可以再加上參數調整

# keras.layers.Flatten：把圖片的格式從二維陣列(28 by 28 pixels)轉成一維陣列(28*28=784 pixels)
# 註：對應到以前做資料整理心法：降維
# 第一個 keras.layers.Dense：128個節點，activation 使用regular
# 第二個 keras.layers.Dense：10個節點，activation 使用softmax，這個回傳一個長度為 10, 總和為 1 的陣列，用來表示屬於哪個分類的可能性
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compile the model
# 設定好 layer 的下一步就是 compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
# 訓練三步驟： 餵資料給model > 讓 model 看得懂圖片跟label的關係 > 用測試資料預測結果
model.fit(train_images, train_labels, epochs=5)
# 訓練完之後會發現有 loss: 0.2939 - accuracy: 0.8912

# Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc) # Test accuracy: 0.8648

# Make predictions
# 預測所有 test data 的種類，並取第一張做檢查
predictions = model.predict(test_images)
print('預測結果:', predictions[0])
print('取最有可能的預測結果:', np.argmax(predictions[0]))
print('回頭檢查 test 的種類:', test_labels[0], '(', class_names[9], ')')


# 接下來我們還可以畫圖
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(img, cmap=plt.cm.binary)
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# 看看前 15 張的結果
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)

plt.show()

# 預測其中一張圖的服裝種類
# 做一個新的測試資料集，裡面放一張圖
img = (np.expand_dims(test_images[0],0))
predictions_single = model.predict(img)
print('predict image is', class_names[np.argmax(predictions_single[0])], 'from result:', predictions_single )

plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()
