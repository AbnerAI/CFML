#from tensorflow import keras
import keras
from keras import regularizers
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Activation, Dropout,Input,Reshape,Lambda,Add,Permute
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)#(B,num_heads,n_tokens,n_tokens)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)#(B,num_heads,n_tokens,projection_dim)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(
            x, (batch_size, -1, self.num_heads, self.projection_dim)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)#(B,n_tokens,embed_dim)
        query = self.separate_heads(query, batch_size)#(B,num_heads,n_tokens,projection_dim)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)#(B,num_heads,n_tokens,projection_dim)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])#(B,n_tokens,num_heads,projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )#(B,n_tokens,num_heads*projection_dim)
        output = self.combine_heads(concat_attention)
        return output

# def CrossHeadsAttModel(nq , d, num_heads,dout = None,nk=None):
#     if dout is None:
#         dout = d
#     dv = int(d / num_heads)
#     if nk is None:
#      nk = nq
#
#     q1 = Input(shape=(nq, d))
#     v1 = Input(shape=(nk, d))
#     k1 = Input(shape=(nk, d))
#
#     q2 = Dense(d,kernel_initializer=keras.initializers.he_normal(seed=None),activation="relu")(q1)
#     v2 = Dense(d,kernel_initializer=keras.initializers.he_normal(seed=None), activation="relu")(v1)
#     k2 = Dense(d,kernel_initializer=keras.initializers.he_normal(seed=None), activation="relu")(k1)
#
#     q = Reshape([nq,num_heads,  dv])(q2)#(Batch,nq,num_heads,dv)
#     q = Permute((2, 1, 3))(q)#(Batch,num_heads,nq,dv)
#
#     v = Reshape([nk,num_heads,  dv])(v2)
#     v = Permute((2, 1, 3))(v)#(Batch,num_heads,nk,dv)
#
#     k = Reshape([ nk, num_heads,  dv])(k2)
#     k = Permute((2, 1, 3))(k)#(Batch,num_heads,nk,dv)
#
#     #att: QK^T  #(Batch,num_heads,nq,nk)
#     att = Lambda(lambda x: K.batch_dot(x[0], K.permute_dimensions(x[1], (0, 1, 3, 2))) / np.sqrt(dv))([q, k])  # l, num_heads, num_heads
#     att = Lambda(lambda x: K.softmax(x,axis=-1))(att)
#
#     out = Lambda(lambda x: K.batch_dot(x[0], x[1]))([att, v])#out: att V #(Batch,num_heads,nq,dv)
#     out = Permute((2, 1, 3))(out)  # (Batch,nq,num_heads,dv)
#     out = Reshape([nq, d])(out)#(Batch,nq,d)
#     out = Dropout(0.5)(out)
#     out = Dense(dout, kernel_regularizer=regularizers.l2(0.001),kernel_initializer=keras.initializers.he_normal(seed=None),activation="relu")(out)
#     return Model(inputs=[q1, k1, v1], outputs=out)

def MultiHeadsAttModel(nq , d, num_heads,dout = None,nk=None):
    if dout is None:
        dout = d
    dv = int(d / num_heads)
    if nk is None:
     nk = nq
    q1 = Input(shape=(nq, d))
    v1 = Input(shape=(nk, d))
    k1 = Input(shape=(nk, d))

    q2 = Dense(d, activation="relu")(q1)
    v2 = Dense(d, activation="relu")(v1)
    k2 = Dense(d, activation="relu")(k1)

    q = Reshape([nq, num_heads, dv])(q2)
    v = Reshape([nk, num_heads, dv])(v2)
    k = Reshape([nk, num_heads, dv])(k2)


    #q:(B,nq,num_heads,dv)
    #k:(B,nk,num_heads,dv)
    att = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[-1, -1]) / np.sqrt(dv))([q, k])  # l, num_heads, num_heads
    #att:(B,nq,dv,dv)
    att = Lambda(lambda x: K.softmax(x))(att)
    out = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[4, 3]))([att, v])
    out = Reshape([nq, d])(out)
    out = Dense(dout, activation="relu")(out)

    return Model(inputs=[q1, k1, v1], outputs=out)

def CrossHeadsAttModel(nq , d, num_heads,dout = None,nk=None):
    if dout is None:
        dout = d
    dv = int(d / num_heads)
    if nk is None:
     nk = nq
    q1 = Input(shape=(nq, d))
    v1 = Input(shape=(nk, d))
    k1 = Input(shape=(nk, d))

    q2 = Dense(d, activation="relu")(q1)
    v2 = Dense(d, activation="relu")(v1)
    k2 = Dense(d, activation="relu")(k1)

    # q = Reshape([nq, num_heads, dv])(q2)
    # v = Reshape([nk, num_heads, dv])(v2)
    # k = Reshape([nk, num_heads, dv])(k2)

    q = Reshape([nq, num_heads, dv])(q2)  # (Batch,nq,num_heads,dv)
    q = Permute((2, 1, 3))(q)#(Batch,num_heads,nq,dv)

    v = Reshape([nk,num_heads,  dv])(v2)
    v = Permute((2, 1, 3))(v)#(Batch,num_heads,nk,dv)

    k = Reshape([ nk, num_heads,  dv])(k2)
    k = Permute((2, 1, 3))(k)#(Batch,num_heads,nk,dv)

    #q:(B,nq,num_heads,dv)
    #k:(B,nk,num_heads,dv)
    att = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[3, 3]) / np.sqrt(dv))([q, k])  # l, num_heads, num_heads
    #att:(B,nq,dv,dv)
    att = Lambda(lambda x: K.softmax(x))(att)
    out = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[3, 2]))([att, v])
    out = Reshape([nq, d])(out)
    out = Dense(dout, activation="relu")(out)
    return Model(inputs=[q1, k1, v1], outputs=out)



class NormL(Layer):
    def __init__(self, **kwargs):
        super(NormL, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.a = self.add_weight(name='kernel',
                                      shape=(1,input_shape[-1]),
                                      initializer='ones',
                                      trainable=True)
        self.b = self.add_weight(name='kernel',
                                      shape=(1,input_shape[-1]),
                                      initializer='zeros',
                                      trainable=True)
        super(NormL, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        eps = 0.000001
        mu = K.mean(x, keepdims=True, axis=-1)
        sigma = K.std(x, keepdims=True, axis=-1)
        ln_out = (x - mu) / (sigma + eps)
        return ln_out*self.a + self.b

    def compute_output_shape(self, input_shape):
        return input_shape

class ScaledDotProductAttention(keras.layers.Layer):
    r"""The attention layer that takes three inputs representing queries, keys and values.
    \text{Attention}(Q, K, V) = \text{softmax}(\frac{Q K^T}{\sqrt{d_k}}) V
    See: https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self,
                 return_attention=False,
                 history_only=False,
                 **kwargs):
        """Initialize the layer.
        :param return_attention: Whether to return attention weights.
        :param history_only: Whether to only use history data.
        :param kwargs: Arguments for parent class.
        """
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.return_attention = return_attention
        self.history_only = history_only
        self.intensity = self.attention = None

    def get_config(self):
        config = {
            'return_attention': self.return_attention,
            'history_only': self.history_only,
        }
        base_config = super(ScaledDotProductAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            query_shape, key_shape, value_shape = input_shape
        else:
            query_shape = key_shape = value_shape = input_shape
        output_shape = query_shape[:-1] + value_shape[-1:]
        if self.return_attention:
            attention_shape = query_shape[:2] + (key_shape[1],)
            return [output_shape, attention_shape]
        return output_shape

    def compute_mask(self, inputs, mask=None):
        if isinstance(mask, list):
            mask = mask[0]
        if self.return_attention:
            return [mask, None]
        return mask

    def call(self, inputs, mask=None, **kwargs):
        if isinstance(inputs, list):
            query, key, value = inputs
        else:
            query = key = value = inputs
        # if isinstance(mask, list):
        #     mask = mask[1]
        feature_dim = K.shape(query)[-1]
        e = K.batch_dot(query, key, axes=2) / K.sqrt(K.cast(feature_dim, dtype=K.floatx()))
        if self.history_only:
            query_len, key_len = K.shape(query)[1], K.shape(key)[1]
            indices = K.expand_dims(K.arange(0, key_len), axis=0)
            upper = K.expand_dims(K.arange(0, query_len), axis=-1)
            e -= 10000.0 * K.expand_dims(K.cast(indices > upper, K.floatx()), axis=0)
        # if mask is not None:
        #     e -= 10000.0 * (1.0 - K.cast(K.expand_dims(mask, axis=-2), K.floatx()))
        self.intensity = e
        e = K.exp(e - K.max(e, axis=-1, keepdims=True))
        self.attention = e / K.sum(e, axis=-1, keepdims=True)
        v = K.batch_dot(self.attention, value)
        if self.return_attention:
            return [v, self.attention]
        return v

class MultiHeadAttention(keras.layers.Layer):
    """Multi-head attention layer.
    See: https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self,
                 head_num,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 history_only=False,
                 **kwargs):
        """Initialize the layer.
        :param head_num: Number of heads.
        :param activation: Activations for linear mappings.
        :param use_bias: Whether to use bias term.
        :param kernel_initializer: Initializer for linear mappings.
        :param bias_initializer: Initializer for linear mappings.
        :param kernel_regularizer: Regularizer for linear mappings.
        :param bias_regularizer: Regularizer for linear mappings.
        :param kernel_constraint: Constraints for linear mappings.
        :param bias_constraint: Constraints for linear mappings.
        :param history_only: Whether to only use history in attention layer.
        """
        self.supports_masking = True
        self.head_num = head_num
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.history_only = history_only

        self.Wq = self.Wk = self.Wv = self.Wo = None
        self.bq = self.bk = self.bv = self.bo = None

        self.intensity = self.attention = None
        super(MultiHeadAttention, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'head_num': self.head_num,
            'activation': keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint),
            'history_only': self.history_only,
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            q, k, v = input_shape
            return q[:-1] + (v[-1],)
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        if isinstance(input_mask, list):
            return input_mask[0]
        return input_mask

    def build(self, input_shape):
        if isinstance(input_shape, list):
            q, k, v = input_shape
        else:
            q = k = v = input_shape
        feature_dim = int(v[-1])
        if feature_dim % self.head_num != 0:
            raise IndexError('Invalid head number %d with the given input dim %d' % (self.head_num, feature_dim))
        self.Wq = self.add_weight(
            shape=(int(q[-1]), feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wq' % self.name,
        )
        if self.use_bias:
            self.bq = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bq' % self.name,
            )
        self.Wk = self.add_weight(
            shape=(int(k[-1]), feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wk' % self.name,
        )
        if self.use_bias:
            self.bk = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bk' % self.name,
            )
        self.Wv = self.add_weight(
            shape=(int(v[-1]), feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wv' % self.name,
        )
        if self.use_bias:
            self.bv = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bv' % self.name,
            )
        self.Wo = self.add_weight(
            shape=(feature_dim, feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wo' % self.name,
        )
        if self.use_bias:
            self.bo = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bo' % self.name,
            )
        super(MultiHeadAttention, self).build(input_shape)

    @staticmethod
    def _reshape_to_batches(x, head_num):
        input_shape = K.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        head_dim = feature_dim // head_num
        x = K.reshape(x, (batch_size, seq_len, head_num, head_dim))
        x = K.permute_dimensions(x, [0, 2, 1, 3])
        return K.reshape(x, (batch_size * head_num, seq_len, head_dim))

    @staticmethod
    def _reshape_attention_from_batches(x, head_num):
        input_shape = K.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        x = K.reshape(x, (batch_size // head_num, head_num, seq_len, feature_dim))
        return K.permute_dimensions(x, [0, 2, 1, 3])

    @staticmethod
    def _reshape_from_batches(x, head_num):
        input_shape = K.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        x = K.reshape(x, (batch_size // head_num, head_num, seq_len, feature_dim))
        x = K.permute_dimensions(x, [0, 2, 1, 3])
        return K.reshape(x, (batch_size // head_num, seq_len, feature_dim * head_num))

    @staticmethod
    def _reshape_mask(mask, head_num):
        if mask is None:
            return mask
        seq_len = K.shape(mask)[1]
        mask = K.expand_dims(mask, axis=1)
        mask = K.tile(mask, [1, head_num, 1])
        return K.reshape(mask, (-1, seq_len))

    def call(self, inputs, mask=None):
        if isinstance(inputs, list):
            q, k, v = inputs
        else:
            q = k = v = inputs
        # if isinstance(mask, list):
        #     q_mask, k_mask, v_mask = mask
        # else:
        #     q_mask = k_mask = v_mask = mask
        q = K.dot(q, self.Wq)
        k = K.dot(k, self.Wk)
        v = K.dot(v, self.Wv)
        if self.use_bias:
            q += self.bq
            k += self.bk
            v += self.bv
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)
        scaled_dot_product_attention = ScaledDotProductAttention(
            history_only=self.history_only,
            name='%s-Attention' % self.name,
        )
        y = scaled_dot_product_attention(
            inputs=[
                self._reshape_to_batches(q, self.head_num),
                self._reshape_to_batches(k, self.head_num),
                self._reshape_to_batches(v, self.head_num),
            ],
            # mask=[
            #     self._reshape_mask(q_mask, self.head_num),
            #     self._reshape_mask(k_mask, self.head_num),
            #     self._reshape_mask(v_mask, self.head_num),
            # ],
        )
        self.intensity = self._reshape_attention_from_batches(scaled_dot_product_attention.intensity, self.head_num)
        self.attention = self._reshape_attention_from_batches(scaled_dot_product_attention.attention, self.head_num)
        y = self._reshape_from_batches(y, self.head_num)
        y = K.dot(y, self.Wo)
        if self.use_bias:
            y += self.bo
        if self.activation is not None:
            y = self.activation(y)

        # Add shape information to tensor
        input_shape = [K.int_shape(q), K.int_shape(k), K.int_shape(v)]
        output_shape = self.compute_output_shape(input_shape)
        if output_shape[1] is not None:
            output_shape = (-1,) + output_shape[1:]
            y = K.reshape(y, output_shape)
        return y
