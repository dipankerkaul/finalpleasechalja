import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.preprocessing.image import ImageDataGenerator
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

img_height = 170
img_width = 170
batch_size = 2
model = keras.Sequential(name="my_sequential")
model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(170, 170, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=5,
    zoom_range=(0.95, 0.95),
    horizontal_flip=False,
    vertical_flip=False,
    data_format="channels_last",
    validation_split=0.0,
    dtype=tf.float32,
)
datagen2 = ImageDataGenerator(rescale=1.0 / 255)
train_generator = datagen.flow_from_directory(
    "Train2/Train2",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="sparse",
    shuffle=True,
    subset="training",
    seed=123,
)
val_generator = datagen2.flow_from_directory(
    "Val2/Val2",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="sparse",
    shuffle=True,
    subset="training",
    seed=123,
)







print("[INFO] Model Compilation starts...")
start = time.time()
model.compile(loss='binary_crossentropy',
 optimizer=keras.optimizers.RMSprop(lr=1e-4),
 metrics=['acc'])
end = time.time()

print("[INFO] Model Compilation took {:.6f} seconds".format(end - start))

print("[INFO] Model Fitting starts...")
start = time.time()
history = model.fit_generator(
 train_generator,
 steps_per_epoch=len(train_generator),
 epochs=10,
 validation_data=val_generator,
 validation_steps=len(val_generator))
end = time.time()

print("[INFO] Model Fitting took {:.6f} seconds".format(end - start))

model.save('cake_mobile_with_data_augmenation2.h5')

print(history.history.keys())

import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


