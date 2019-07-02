# -*- coding: UTF-8 -*-
# 匯入必要的 TensorFlow lib
import tensorflow as tf

# 宣告 constant
hello = tf.constant('Hello, TensorFlow!')

# 宣告 Session
sess = tf.Session()
output = sess.run(hello)
print(output)
sess.close()
print('session close.')

# 或是使用 with, 就可以自動關session
with tf.Session() as sess:
  print(sess.run(hello))
