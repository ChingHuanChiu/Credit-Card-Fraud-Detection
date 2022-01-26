import tensorflow as tf

class Encoder(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.dense_128 = tf.keras.layers.Dense(128)
        self.dense_64 = tf.keras.layers.Dense(64)
        self.dense_32 = tf.keras.layers.Dense(32)
        self.dense_8 = tf.keras.layers.Dense(8)
        self.dense_4 = tf.keras.layers.Dense(4)


    def call(self, inputs):
        x = self.dense_128(inputs)
        x = self.dense_64(x)
        x = self.dense_32(x)
        x = self.dense_8(x)
        x = self.dense_4(x)
        return x
    
class Decoder(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.dense_8 = tf.keras.layers.Dense(8)
        self.dense_32 = tf.keras.layers.Dense(32)
        self.dense_64 = tf.keras.layers.Dense(64)
        self.dense_128 = tf.keras.layers.Dense(128)

        
    def call(self, inputs):
        x = self.dense_8(inputs)
        x = self.dense_32(x)
        x = self.dense_64(x)
        x = self.dense_128(x)
        return x


class CredictFraudDetect(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        self.encoder = Encoder()
        self.decoder = Decoder()
        
        self.dense = tf.keras.layers.Dense(2, activation='softmax')
    def call(self, inp):
        x = self.encoder(inp)
        x = self.decoder(x)
        x = self.dense(x)
        
        return x


# class OneHotEmbedding(tf.keras.layers.Layer):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.input_dim = int(input_dim)
#         self.embedding = tf.keras.layers.Embedding(self.input_dim, int(self.input_dim / 2), name='embedding')
#         self.flatten = tf.keras.layers.Flatten(name='flatten')
#         self.dense_64 = tf.keras.layers.Dense(64, name='onehot_dense64')
#         self.dense_32 = tf.keras.layers.Dense(32, name='onehot_dense32')

#     @tf.function
#     def call(self, inputs):
#         x = self.embedding(inputs)
#         x = self.flatten(x)
#         x = self.dense_64(x)
#         x = self.dense_32(x)

#         return x


# class NN(tf.keras.layers.Layer):
#     def __init__(self):
#         super().__init__()
#         self.dense_64 = tf.keras.layers.Dense(64, name='nn_dense64')
#         self.dense_32 = tf.keras.layers.Dense(32, name='nn_dense32')

#     @tf.function
#     def call(self, inputs):
#         x = self.dense_64(inputs)
#         x = self.dense_32(x)

#         return x




# class MultiInputCredictFraudDetect(tf.keras.Model):
#     def __init__(self, embedding_input_dim):
#         super().__init__()
        
#         self.onehotembedding = OneHotEmbedding(embedding_input_dim)
#         self.nn = NN()
        
#         self.dense_8 = tf.keras.layers.Dense(8)
#         self.dense = tf.keras.layers.Dense(2, activation='softmax')
    
#     @tf.function(input_signature=[tf.TensorSpec([None,196], tf.float32,name='onehot'), 
                                #   tf.TensorSpec([None,7], tf.float32,name='numeric')])
#     def call(self,one_hot_inp, inp):
#         x1 = self.onehotembedding(one_hot_inp)
#         x2 = self.nn(inp)
#         concat_x = tf.keras.layers.Concatenate()([x1, x2])
#         out = self.dense_8(concat_x)

#         out = self.dense(out)
        
#         return out


    # def get_config(self):

    #     config = super().get_config().copy()
    #     config.update({
    #         'embedding_input_dim': self.embedding_input_dim,
        
    #     })
    #     return config
    

