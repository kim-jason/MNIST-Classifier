import numpy as np
import mnist
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

model = Sequential([
  Dense(64, activation='relu', input_shape=(784,)),
  Dense(64, activation='relu'),
  Dense(10, activation='softmax'),
])

model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

model.fit(
  train_images,
  to_categorical(train_labels), 
  epochs=5,
  batch_size=32,
)

model.evaluate(
  test_images,
  to_categorical(test_labels)
)

#model.save_weights('model.h5')

# model = Sequential([
#   Dense(64, activation='relu', input_shape=(784,)),
#   Dense(64, activation='relu'),
#   Dense(10, activation='softmax'),
# ])

# model.load_weights('model.h5')

predictions = model.predict(test_images[:5])
print(np.argmax(predictions, axis=1)) # [7, 2, 1, 0, 4]
print(test_labels[:5]) # [7, 2, 1, 0, 4]