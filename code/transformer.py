import tensorflow as tf
from MultiHeadAttention import MultiHeadAttention
class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff=512, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=128)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)

    self.d1=tf.keras.layers.Dense(dff, activation='relu')
    self.d2=tf.keras.layers.Dense(d_model)

  def call(self, x, training):

    attn_output= self.mha(x, x, x) 
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)

    ffn_output = self.d1(out1)
    ffn_output = self.d2(ffn_output)  
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output) 

    return out2