# -*- coding: UTF-8 -*-
# 練習要怎麼把訓練好的 model 存起來，並且再利用
# pip install -q h5py pyyaml 

from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
from tensorflow import keras

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

# Define a model
def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation=tf.keras.activations.relu, input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation=tf.keras.activations.softmax)
  ])
  model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])
  return model

model = create_model()
model.summary()


# Save checkpoints during training
## Checkpoint callback usage
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=1)
model = create_model()
model.fit(train_images, train_labels,  epochs = 10, 
          validation_data = (test_images,test_labels),
          callbacks = [cp_callback])  # pass callback to training
# 會在 training_1 下面看到 checkpoint  cp.ckpt.data-00000-of-00001  cp.ckpt.index



# 如果只要 restore 權重, 那新的 model 就必須與舊的有相同的架構；
# 儘管是不同 instance 的 model, 只要有一樣的架構, 我們就可以共享權重
# 接下來建立一個新的 model, 直接評估, 發現很爛
model = create_model()
loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))
# 會得到 Untrained model, accuracy: 12.70%

# 載入以前的資料啊！
model.load_weights(checkpoint_path)
loss,acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
# Restored model, accuracy: 86.00%

## Checkpoint callback options
# checkpoint 有很多 option 可以調整
# include the epoch in the file name. (uses `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every 5-epochs.
    period=5)

model = create_model()
model.save_weights(checkpoint_path.format(epoch=0))
model.fit(train_images, train_labels,
          epochs = 50, callbacks = [cp_callback],
          validation_data = (test_images,test_labels),
          verbose=0)
# 會在 training_2 下面看到很多檔案
latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest) 
# 會得到 'training_2/cp-0050.ckpt'

# 用最後一個還原
model = create_model()
model.load_weights(latest)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
# 會得到 Restored model, accuracy: 87.20%

## Manually save weights
# Save the weights
model.save_weights('./checkpoints/my_checkpoint')

# Restore the weights
model = create_model()
model.load_weights('./checkpoints/my_checkpoint')

loss,acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
# 會得到 Restored model, accuracy: 87.20%


## Save the entire model
# 我們也可以儲存整個 model, 包含權重、設定值、最佳化的設定
# 在 TF 裡，我們會存成 HDF5 檔

model = create_model()
model.fit(train_images, train_labels, epochs=5)
# Save entire model to a HDF5 file
model.save('my_model.h5')

# Recreate the exact same model, including weights and optimizer.
new_model = keras.models.load_model('my_model.h5')
new_model.summary()

loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
# 會得到 Restored model, accuracy: 85.50%

## As a saved model
model = create_model()
model.fit(train_images, train_labels, epochs=5)
saved_model_path = tf.contrib.saved_model.save_keras_model(model, "./saved_models")

new_model = tf.contrib.saved_model.load_keras_model(saved_model_path)
new_model.summary()

new_model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
# 會得到 Restored model, accuracy: 86.20%



