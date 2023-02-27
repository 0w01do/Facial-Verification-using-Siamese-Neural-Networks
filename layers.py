# * custom L1 distance layer module
# * WHY DO WE NEED THIS:its needed to load the custom model


# * Import dependecies

import tensorflow as tf
from tensorflow.python.keras.layers import Layer

# * custom L1 distance
# * get from jyupiter note-book part-4


class L1Dist(Layer):
    # * init method - inheritance
    def __init__(self):
        super().__init__()

    # * Magic Happens Here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
