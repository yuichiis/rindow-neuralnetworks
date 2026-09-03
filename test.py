#!/usr/bin/env python
import tensorflow as tf

input = tf.keras.layers.Input(shape=(6, 6, 4))
conv = tf.keras.layers.Conv2D(
    filters=2,
    kernel_size=3,
    strides=1,
    #padding='same',
)(input)
model = tf.keras.models.Model(inputs=input, outputs=conv)
model.summary()

input = tf.keras.layers.Input(shape=(6, 6, 4))
conv = tf.keras.layers.MaxPooling2D(
    pool_size=2,
    strides=1,
    #padding='same',
)(input)
model = tf.keras.models.Model(inputs=input, outputs=conv)
model.summary()

