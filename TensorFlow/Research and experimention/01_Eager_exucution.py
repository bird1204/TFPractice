# -*- coding: UTF-8 -*-
# 匯入必要的 TensorFlow lib
import tensorflow as tf

tf.enable_eager_execution()

## Tensor 
# tensor 是一個 multi-dimensional array，跟 NumPy 的 ndarray 很像，tensor object 包含了 datatype, shape
print(tf.add(1, 2))
print(tf.add([1, 2], [3, 4]))
print(tf.square(5))
print(tf.reduce_sum([1, 2, 3]))
print(tf.encode_base64("hello world"))
print(tf.square(2) + tf.square(3))
x = tf.matmul([[1]], [[2, 3]])
print(x.shape)
print(x.dtype)

# Tensor 跟 NumPy arry 最大差別是 :
# 1. Tensor 可以被加速器支持 ( GPU, TPU )
# 2. Tensor 是不可改變的

## NumPy Compatibility
# tensor 跟  NumPy ndarrays 之間是可以相互轉換的～
import numpy as np

ndarray = np.ones([3, 3])

print("TensorFlow operations convert numpy arrays to Tensors automatically")
tensor = tf.multiply(ndarray, 42)
print(tensor)


print("And NumPy operations convert Tensors to numpy arrays automatically")
print(np.add(tensor, 1))

print("The .numpy() method explicitly converts a Tensor to a numpy array")
print(tensor.numpy())

## GPU acceleration
# 使用 GPU 計算可以加快 TF 的操作，在沒有指定的情況下，TF 會自己決定要用 CPU 還是 GPU
x = tf.random_uniform([3, 3])

print("Is there a GPU available: "),
print(tf.test.is_gpu_available())

print("Is the Tensor on GPU #0:  "),
print(x.device.endswith('GPU:0'))

## Device Names
## Explicit Device Placement
# TensorFlow 中的  placement 指的是如何為執行中的程序分配設備，我們可以用 tf.device 管理
import time

def time_matmul(x):
  start = time.time()
  for loop in range(10):
    tf.matmul(x, x)
  result = time.time()-start
  print("10 loops: {:0.2f}ms".format(1000*result))


# Force execution on CPU
print("On CPU:")
with tf.device("CPU:0"):
  x = tf.random_uniform([1000, 1000])
  assert x.device.endswith("CPU:0")
  time_matmul(x)

# Force execution on GPU #0 if available
if tf.test.is_gpu_available():
  with tf.device("GPU:0"): # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
    x = tf.random_uniform([1000, 1000])
    assert x.device.endswith("GPU:0")
    time_matmul(x)

## Datasets
# Create a source Dataset
ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])

# Create a CSV file
import tempfile
_, filename = tempfile.mkstemp()

with open(filename, 'w') as f:
  f.write("""Line 1
Line 2
Line 3
  """)

ds_file = tf.data.TextLineDataset(filename)

## Apply transformations
ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)

ds_file = ds_file.batch(2)

## Iterate
print('Elements of ds_tensors:')
for x in ds_tensors:
  print(x)

print('\nElements in ds_file:')
for x in ds_file:
  print(x)