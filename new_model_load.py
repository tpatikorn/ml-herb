import os

import keras
import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs
from sklearn.metrics import confusion_matrix
from tensorflow.math import argmax

PATH = 'D:/Projects/ml-herb/training_images'
test = 'D:/Projects/ml-herb/test_images'

file_names = os.listdir(test)
print(file_names)

train_dir = os.path.join(PATH)
validation_dir = os.path.join(test)

BATCH_SIZE = 32
IMG_SIZE = (180, 180)

train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE,
                                                            validation_split=0.2,
                                                            subset="training",
                                                            seed=1234567,
                                                            label_mode="categorical")

validation_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                                 shuffle=True,
                                                                 batch_size=BATCH_SIZE,
                                                                 image_size=IMG_SIZE,
                                                                 validation_split=0.2,
                                                                 subset="validation",
                                                                 seed=1234567,
                                                                 label_mode="categorical")

test_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                           shuffle=False,
                                                           batch_size=BATCH_SIZE,
                                                           image_size=IMG_SIZE,
                                                           label_mode="categorical")

model = keras.models.load_model("models/new_cleaner_test.keras")

predicted_classes = np.array([])
true_classes = np.array([])

for x, y in validation_dataset:
    predicted_classes = np.concatenate([predicted_classes, np.argmax(model.predict(x), axis=-1)])
    true_classes = np.concatenate([true_classes, np.argmax(y.numpy(), axis=-1)])

cf = confusion_matrix(true_classes, predicted_classes)
print("val", cf)

actual_y = [k for x, y in test_dataset for k in y]  # need some magic because it's batched
argmax_actual_y = argmax(actual_y, axis=1)
y_pred = model.predict(test_dataset)
argmax_prediction = argmax(y_pred, axis=1)
cf = confusion_matrix(argmax_actual_y, argmax_prediction)
print("test", cf)

tfjs.converters.save_keras_model(model, tfjs_target_dir)
