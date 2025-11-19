import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.math import argmax

PATH = 'D:/Projects/ml-herb/training_images'
test = 'D:/Projects/ml-herb/test_images'

file_names = os.listdir(test)
print(file_names)

train_dir = os.path.join(PATH)
validation_dir = os.path.join(test)

BATCH_SIZE = 32
IMG_SIZE = (224, 224)

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

print("All classnames", validation_dataset.class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    # tf.keras.layers.RandomSaturation(3),
    # tf.keras.layers.RandomCrop(central_fraction=0.5),
])

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# Create the base model from the pre-trained model MobileNet V2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV3Large(input_shape=IMG_SHAPE,
                                                    include_top=False, weights='imagenet')

image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)

base_model.trainable = False

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

prediction_layer = tf.keras.layers.Dense(20, activation="softmax")
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

inputs = tf.keras.Input(shape=(180, 180, 3))
x1 = data_augmentation(inputs)
x2 = preprocess_input(x1)
x3 = base_model(x2, training=False)
x4 = global_average_layer(x3)
x5 = tf.keras.layers.Dropout(0.2)(x4)
outputs = prediction_layer(x5)
model = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.0001
# model.compile(optimizer='your_optimizer', loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

model.summary()

model.save("models/new_cleaner_test.keras")

initial_epochs = 1
loss0, accuracy0 = model.evaluate(validation_dataset)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset)

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
