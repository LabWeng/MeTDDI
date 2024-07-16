import tensorflow as tf
from tensorflow import keras
import time
import numpy as np
import matplotlib.pyplot as plt

def rescale_distance_matrix(w): ### For global
    constant_value = tf.constant(1.0,dtype=tf.float32) ## Vi default set as 1
    return (constant_value+tf.math.exp(constant_value))/(constant_value+tf.math.exp(constant_value-w))

def gelu(x):
    return 0.5 * x * (1.0 + tf.math.erf(x / tf.sqrt(2.)))

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


# def create_padding_mask(seq):
#     seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
#
#     # add extra dimensions to add the padding
#     # to the attention logits.
#     return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
#
# def create_look_ahead_mask(size):
#   mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
#   return mask  # (seq_len, seq_len)
def create_padding_mask_atom(batch_data):
    padding_mask = tf.cast(tf.math.equal(tf.reduce_sum(batch_data,axis=-1), 0), tf.float32)
    # [batch_size, 1, 1, seq_len]
    return padding_mask[:, tf.newaxis, tf.newaxis, :]

def scaled_dot_product_attention(q, k, v, mask,adjoin_matrix,dist_matrix):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    # matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # # scale matmul_qk
    # dk = tf.cast(tf.shape(k)[-1], tf.float32)
    # scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if dist_matrix is not None:
        matmul_qk = tf.nn.relu(tf.matmul(q, k, transpose_b = True))
        dist_matrix = rescale_distance_matrix(dist_matrix)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = (tf.multiply(matmul_qk,dist_matrix)) / tf.math.sqrt(dk)
    else:
        matmul_qk = tf.matmul(q, k, transpose_b = True)
        
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    if adjoin_matrix is not None:
        scaled_attention_logits += adjoin_matrix 
        # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, d_model, num_heads,**kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask,adjoin_matrix,dist_matrix):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask,adjoin_matrix,dist_matrix)
        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

def feed_forward_network(d_model, dff):
    # dff: dim of feed forward network.
    return keras.Sequential([
        keras.layers.Dense(dff, activation=gelu),
        keras.layers.Dense(d_model)
    ])
class EncoderLayer(keras.layers.Layer):
    """
    x -> self attention -> add & normalize & dropout
      -> feed_forward -> add & normalize & dropout
    """
    def __init__(self, d_model, num_heads, dff,rate,**kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.mha1 = MultiHeadAttention(int(d_model/2), num_heads)
        self.mha2 = MultiHeadAttention(int(d_model/2), num_heads)
        self.ffn = feed_forward_network(d_model, dff)
        self.layer_norm1 = keras.layers.LayerNormalization(
            epsilon = 1e-6)
        self.layer_norm2 = keras.layers.LayerNormalization(
            epsilon = 1e-6)
        
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)
    def call(self, x, training, encoder_padding_mask,adjoin_matrix,dist_matrix):
        # x.shape          : (batch_size, seq_len, dim=d_model)
        # attn_output.shape: (batch_size, seq_len, d_model)
        # out1.shape       : (batch_size, seq_len, d_model)
        x1,x2 = tf.split(x,2,-1)
        x_l,attention_weights_local = self.mha1(x1, x1, x1, encoder_padding_mask,adjoin_matrix,dist_matrix = None)
        x_g,attention_weights_global = self.mha2(x2, x2, x2, encoder_padding_mask,adjoin_matrix = None,dist_matrix = dist_matrix)
        attn_output = tf.concat([x_l,x_g],axis=-1)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layer_norm1(x + attn_output)
        # ffn_output.shape: (batch_size, seq_len, d_model)
        # out2.shape      : (batch_size, seq_len, d_model)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layer_norm2(out1 + ffn_output)
        # x.shape: (batch_size, input_seq_len, d_model)
        x_l_g = out2
        return x_l_g,attention_weights_local,attention_weights_global

class EncoderModel_atom(keras.layers.Layer):
    def __init__(self, num_layers, 
                 d_model, num_heads, dff, rate=0.1,**kwargs):
        super(EncoderModel_atom, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_layers = num_layers
        # self.max_length = max_length
        
        self.embedding = keras.layers.Dense(self.d_model,activation='relu')
        # position_embedding.shape: (1, max_length, d_model)
        # self.position_embedding = get_position_embedding(max_length,
        #                                                  self.d_model) 
        self.global_embedding = keras.layers.Dense(256, activation='relu')

        self.dropout = keras.layers.Dropout(rate)
        self.encoder_layers = [
            EncoderLayer(int(d_model), num_heads, dff, rate)
            for _ in range(self.num_layers)]

    def call(self, x, training,adjoin_matrix = None,
             dist_matrix = None,atom_match_matrix = None,sum_atoms = None):
        # x.shape: (batch_size, input_seq_len)
        input_seq_len = tf.shape(x)[1] 
        encoder_padding_mask = create_padding_mask_atom(x) 
        # tf.debugging.assert_less_equal(
        #     input_seq_len, self.max_length,
        #     "input_seq_len should be less or equal to self.max_length")
        if adjoin_matrix is not None:
            adjoin_matrix = adjoin_matrix[:,tf.newaxis,:,:]
        if dist_matrix is not None:
            # dist_matrix_temp = tf.pad(dist_matrix,[[0,0],[1,0],[1,0]],constant_values=0)
            dist_matrix = dist_matrix[:,tf.newaxis,:,:]
        # x.shape: (batch_size, input_seq_len, d_model)
        x = self.embedding(x)
        x = self.dropout(x, training = training)
        attention_weights_list_local = []
        # xs_local = []
        attention_weights_list_global = []
        for i in range(self.num_layers):
            x,attention_weights_local,attention_weights_global = self.encoder_layers[i](x, training, 
                                       encoder_padding_mask,adjoin_matrix,dist_matrix = dist_matrix) 
            attention_weights_list_local.append(attention_weights_local) 
            attention_weights_list_global.append(attention_weights_global)
        x = tf.matmul(atom_match_matrix,x)/sum_atoms 
        x = self.global_embedding(x)
        # x_temp = tf.reduce_sum(x_temp,axis = 1)[:,tf.newaxis,:]
        # x = tf.concat([x_temp,x],axis=1)
        return x,attention_weights_list_local,attention_weights_list_global,encoder_padding_mask



