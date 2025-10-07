# train_emotion.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, callbacks, optimizers

num_classes = 7
img_size = (48, 48)   # common size for facial emotion datasets

def build_model(input_shape=(48,48,1), num_classes=7):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(256, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D()(x)
    x = layers.Flatten()(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    out = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inp, out)
    return model

# Data generators
train_dir = "data/train"
val_dir = "data/val"

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

val_gen = ImageDataGenerator(rescale=1./255)

train_flow = train_gen.flow_from_directory(
    train_dir, target_size=img_size, color_mode='grayscale',
    batch_size=64, class_mode='categorical'
)

val_flow = val_gen.flow_from_directory(
    val_dir, target_size=img_size, color_mode='grayscale',
    batch_size=64, class_mode='categorical'
)

model = build_model(input_shape=(48,48,1), num_classes=num_classes)
model.compile(optimizer=optimizers.Adam(1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

cb = [
    callbacks.ModelCheckpoint("best_emotion_model.h5", save_best_only=True, monitor='data_loss'),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6),
    callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
]

model.fit(train_flow,
          epochs=30,
          validation_data=val_flow,
          callbacks=cb)
# saved model: best_emotion_model.h5
