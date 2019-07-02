# -*- coding: UTF-8 -*-
# 線性問題旨在預測一個連續值，有別於分類測試的預測是哪一個種類 ( 不懂去看 concept )
# 這次我們的資料是 Auto MPG, 1970 ~ 1980 間的汽車資料
# 為了畫圖，我們需要安裝新的 lib
# pip install -q seaborn

from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 這次的資料集是來自 UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/index.php)
# 所以我們要自己下載資料囉～
dataset_path = keras.utils.get_file("auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
dataset_path


# 把資料匯入，使用 panda
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin'] 
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.tail()

# Clean the data

# 檢查資料狀態
dataset.isna().sum()
# 發現有些 unknown value
# MPG 0 
# Cylinders 0 
# Displacement 0 
# Horsepower 6 
# Weight 0 
# Acceleration 0 
# Model Year 0 
# Origin 0 
# dtype: int64

# 把錯誤資料刪掉
dataset = dataset.dropna()

# "Origin" 欄位是真的國家，把這邊整理一下，變成 one-hot
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
dataset.tail()
# 會發現多了 USA, EUROPE, JAPAN

# Split the data into train and test
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Inspect the data
# 看個圖，還有統計數字
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
# <seaborn.axisgrid.PairGrid at 0x7f4b8be6a160>

train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
train_stats

# Split features from labels
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

# Normalize the data
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


# Build the model
def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)
  ])
  optimizer = tf.keras.optimizers.RMSprop(0.001)
  model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model

model = build_model()


# Inspect the model
model.summary()

# 先用前面十筆測測看囉
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
example_result

# Train the model
# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])

# train 完看一下訓練結果
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

# 畫圖看結果
def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()

plot_history(history)
# 發現訓練結果在 100 次之後就沒有進步，甚至變壞，所以我們要在模型沒有進步的時候停止訓練

model = build_model()
# 每 10 次檢查一次
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)

# 好勒～我們現在可以用 test data 試試看
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))
# Testing set Mean Abs Error:  1.95 MPG

# Make predictions
test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
plt.show()

# 在這個範例，我們使用了幾個方式來處理線性問題
# 1. Mean Squared Error (MSE) 是常用於線性問題的損失函數
# 2. 同上，Mean Absolute Error (MAE)是常用於線性問題的評估指標
# 3. 當輸入值是數字且具有不同範圍時，將每個 feature 都獨立地變成相同範圍的值 ( 國家那部分 ) 
# 4. 如果訓練的資料不夠多，則也盡量少用 hidden layer，避免 over fitting
# 5. EarlyStopping 是避免 over fitting 的好方式