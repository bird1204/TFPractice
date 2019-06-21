# First Steps with TF

## toolkit

![Google example](https://developers.google.com/machine-learning/crash-course/images/TFHierarchy.svg?authuser=1&refresh=1)

## tf.estimator API

```python
import tensorflow as tf

# Set up a linear classifier.
classifier = tf.estimator.LinearClassifier(feature_columns)

# Train the model on some example data.
classifier.train(input_fn=train_input_fn, steps=2000)

# Use it to predict.
predictions = classifier.predict(input_fn=predict_input_fn)
```

剩下的看練習