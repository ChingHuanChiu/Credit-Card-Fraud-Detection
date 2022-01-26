import tensorflow as tf

class Encoder(tf.keras.layers.Layer):
    def __init__(self, first_layer_dim):
        super().__init__()
        self.dense_first_layer = tf.keras.layers.Dense(first_layer_dim)

        self.dense_128 = tf.keras.layers.Dense(128)
        self.dense_64 = tf.keras.layers.Dense(64)
        self.dense_32 = tf.keras.layers.Dense(32)
        self.dense_8 = tf.keras.layers.Dense(8)

    @tf.function
    def call(self, inputs):
        x = self.dense_first_layer(inputs)
        x = self.dense_128(x)
        x = self.dense_64(x)
        x = self.dense_32(x)
        x = self.dense_8(x)
        return x
    
class Decoder(tf.keras.layers.Layer):
    def __init__(self, last_layer_dim):
        super().__init__()
        self.dense_32 = tf.keras.layers.Dense(32)
        self.dense_64 = tf.keras.layers.Dense(64)
        self.dense_128 = tf.keras.layers.Dense(128)
        self.dense_last_layer = tf.keras.layers.Dense(last_layer_dim)

    @tf.function 
    def call(self, inputs):
        x = self.dense_32(inputs)
        x = self.dense_64(x)
        x = self.dense_128(x)
        x = self.dense_last_layer(x)
        return x



class AutoEncoderDecoder(tf.keras.layers.Layer):
    def __init__(self, inp_dim):
        super().__init__()

        self.encoder = Encoder(inp_dim)
        self.decoder = Decoder(inp_dim)
    @tf.function    
    def call(self, inp):
        x = self.encoder(inp)
        x = self.decoder(x)
        return x