import numpy as np
from models.layers import NormL,MultiHeadSelfAttention,MultiHeadsAttModel
from keras.layers import Dense, Activation, Dropout,Lambda, Concatenate,BatchNormalization,Input,Add,Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.utils import np_utils
from keras.models import load_model
import os
from keras.layers.recurrent import GRU
from keras import backend as K
from keras.regularizers import l1_l2
from keras.models import Model
from keras.layers.convolutional import Conv1D,MaxPooling1D,Conv2D
from keras.layers import  Reshape,multiply,Permute,Multiply
from keras.layers import Dense, Dropout
import logging
from keras.callbacks import CSVLogger
from tensorflow import keras

def attach_attention_module(Config,net):
    input_dim = Config.input_dim
    seq_length = Config.seq_length
    net_transpose = Permute((2,3,1))(net)
    print(net_transpose.shape)
    avg_pool = Lambda(lambda x:K.mean(x,axis=3,keepdims=True))(net_transpose)
    assert avg_pool.shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(net_transpose)
    assert max_pool.shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat.shape[-1] == 2
    feature = Conv2D(filters=input_dim, kernel_size=(input_dim, 1),  activation='sigmoid',
                                       kernel_initializer='he_normal', use_bias=False)(concat)
    feature = Permute((1, 3, 2))(feature)
    feature = Lambda(lambda x: K.tile(x, [1,seq_length, 1, 1]))(feature)
    return multiply([net,feature])

def multilayer_perceptron(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = Dense(units, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
    return x


from keras.engine.topology import Layer
import tensorflow as tf
class MyLayer1(Layer):
    def __init__(self, **kwargs):
        super(MyLayer1, self).__init__(**kwargs)
        self.constant = tf.Variable(initial_value=1.0, dtype=tf.float32, trainable=True)

    def call(self, input):
        x = self.constant * input
        return x

class MyLayer2(Layer):
    def __init__(self, **kwargs):
        super(MyLayer2, self).__init__(**kwargs)
        self.constant = tf.Variable(initial_value=1.0, dtype=tf.float32, trainable=True)

    def call(self, input):
        x = self.constant * input
        return x

class MyLayer3(Layer):
    def __init__(self, **kwargs):
        super(MyLayer3, self).__init__(**kwargs)
        self.constant = tf.Variable(initial_value=1.0, dtype=tf.float32, trainable=True)

    def call(self, input):
        x = self.constant * input
        return x

class MyLayer4(Layer):
    def __init__(self, **kwargs):
        super(MyLayer4, self).__init__(**kwargs)
        self.constant = tf.Variable(initial_value=1.0, dtype=tf.float32, trainable=True)

    def call(self, input):
        x = self.constant * input
        return x

class MyLayer5(Layer):
    def __init__(self, **kwargs):
        super(MyLayer5, self).__init__(**kwargs)
        self.constant = tf.Variable(initial_value=1.0, dtype=tf.float32, trainable=True)

    def call(self, input):
        x = self.constant * input
        return x


def create_model(Config):
    seq_length = Config.seq_length
    features_length = Config.input_dim
    projection_dim = 50
    num_heads = 2
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]
    transformer_layers = 1
    mlp_head_units = [512, 256]

    # RNN-based TC-specific Encoder
    input_shape = (seq_length, features_length)
    input_layer = Input(shape=input_shape)
    reshape_1 = Reshape((seq_length, features_length, 1))(input_layer)
    attention_layer = attach_attention_module(Config, reshape_1)
    reshape_2 = Reshape((seq_length, features_length))(attention_layer)
    conv_1_1 = Conv1D(32, 2, activation='relu', padding='same')(reshape_2)
    conv_1_2 = Conv1D(16, 4, activation='relu', padding='same')(reshape_2)
    conv_1_3 = Conv1D(16, 8, activation='relu', padding='same')(reshape_2)

    concat_1 = Concatenate()([conv_1_1, conv_1_2, conv_1_3])
    max_pool_1 = MaxPooling1D(3)(concat_1)
    gru_layer_1 = GRU(32, return_sequences=True, dropout=0.3,
                      kernel_regularizer=l1_l2(0.0001, 0.0001),
                      recurrent_regularizer=l1_l2(0.0001, 0.0001)
                      )(max_pool_1)
    gru_concat = Concatenate()([max_pool_1, gru_layer_1])
    gru_layer = GRU(32, return_sequences=True, dropout=0.3,
                    kernel_regularizer=l1_l2(0.0001, 0.0001),
                    recurrent_regularizer=l1_l2(0.0001, 0.0001)
                    )(gru_concat)
    tc_features = Lambda(lambda x: K.mean(x[:, 10:, :], axis=1))(
        gru_layer)  # Average from the 10th time step for further prediction
    print('====== Averaged state =========')

    # Transformer-based FNC-specific Encoder
    input1 = Input(shape=(50, 50))
    encoded_patches = input1
    decay = 0.01
    for _ in range(transformer_layers):
        x1 = NormL()(encoded_patches)

        attention_output = MultiHeadsAttModel(
            nq = 50, d = 50, num_heads = num_heads,
        )([x1, x1, x1])

        x2 = Add()([attention_output, encoded_patches])

        x3 = NormL()(x2)

        x3 = multilayer_perceptron(x3, hidden_units=transformer_units, dropout_rate=0.1)

        encoded_patches1 = Add()([x3, x2])

    # Cross-feature Information Fusion Module
    representation = NormL()(encoded_patches1)
    representation = Flatten()(representation)
    representation = Dropout(0.5)(representation)
    fc_features = multilayer_perceptron(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    fc_features = Dense(32)(fc_features)
    weight1_fc = MyLayer1()(fc_features)
    weight1_tc = MyLayer2()(tc_features)
    fuse1 = Add()([weight1_fc, weight1_tc])
    fc_features = Dense(32)(fc_features)
    tc_features = Dense(32)(tc_features)
    weight2_fc = MyLayer3()(fc_features)
    weight2_tc = MyLayer4()(tc_features)
    fuse2 = Add()([weight2_fc, weight2_tc])
    fuse11 = MyLayer5()(fuse1)
    fuse = Add()([fuse11, fuse2])
    dense_last_step = Dense(32)(fuse)
    drop_last_step = Dropout(0.5)(dense_last_step)
    output_layer = Dense(2, activation='softmax', name='combine')(drop_last_step)

    fc_output = Dense(32)(fc_features)
    fc_output = Dropout(0.5)(fc_output)
    output_fc = Dense(2, activation='softmax',name='fc')(fc_output)
    tc_output = Dense(32)(tc_features)
    tc_output = Dropout(0.5)(tc_output)
    output_tc = Dense(2, activation='softmax',name='tc')(tc_output)

    model = Model(inputs=[input1, input_layer], outputs=[output_layer,output_tc,output_fc])
    return model

def CFML(Config, X_train, y_train, X_test, X_train_cv_tc,X_test_cv_tc, i, k):
    Config.seq_length = 170
    Config.input_dim = 50
    Config.random_state = 10
    Config.batch_size = 64
    Config.lr = 0.001
    y_train = np_utils.to_categorical(y_train, 2)
    batch_size = Config.batch_size
    nb_epoch = 1000

    model = create_model(Config)


    def contrastive_loss(projections_1, projections_2):
        # InfoNCE loss (information noise-contrastive estimation)
        # NT-Xent loss (normalized temperature-scaled cross entropy)

        temperature = 1

        # Cosine similarity: the dot product of the l2-normalized feature vectors
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)
        similarities = (
                tf.matmul(projections_1, projections_2, transpose_b=True) / temperature
        )

        # The similarity between the representations of two augmented views of the
        # same image should be higher than their similarity with other views
        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)

        # The temperature-scaled similarities are used as logits for cross-entropy
        # a symmetrized version of the loss is used here
        loss_1_2 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, similarities, from_logits=True
        )
        loss_2_1 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, tf.transpose(similarities), from_logits=True
        )
        return (loss_1_2 + loss_2_1) / 2

    model.compile(
        optimizer=Adam(lr=Config.lr, decay=1e-2),
        loss={'combine':'categorical_crossentropy', # final loss
              'tc':'categorical_crossentropy', # TC-specific loss
              'fc': 'categorical_crossentropy',# FNC-specific loss
              'mutual': contrastive_loss, # Feature-exchange loss
              },
        loss_weights={'combine':1.0,
                      'tc': 0.5,
                      'fc': 0.5,
                      'mutual': 0.1
                      },
        metrics=['accuracy'], )
    model.summary()

    model_save_path = os.path.join(Config.model_path, 'rnn_repeat_' + str(k + 1) + '_fold_' + str(i + 1) + '_' + "model.hdf5")
    checkpoint = ModelCheckpoint(
        model_save_path,
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
    )
    early_stopping = EarlyStopping(patience=50)


    csv_logger = CSVLogger(os.path.join(Config.workspace,'logs.csv'), append=True, separator=';')
    model.fit(x = [X_train,X_train_cv_tc],y = [y_train,y_train,y_train],batch_size=batch_size,validation_split=0.1,
        verbose=1,callbacks=[checkpoint, early_stopping,csv_logger],epochs=nb_epoch)

    #evaluate model
    model_save = load_model(model_save_path, custom_objects={'NormL': NormL,'MyLayer1':MyLayer1,'MyLayer2':MyLayer2,
                                                             'MyLayer3':MyLayer3,'MyLayer4':MyLayer4,'MyLayer5':MyLayer5})
    print(X_test.shape)
    print(X_test_cv_tc.shape)
    y_submission,_,_ = model_save.predict(x = [X_test,X_test_cv_tc])
    pro = y_submission[:, 1]
    prediction = np.argmax(y_submission, axis=1)
    return pro, prediction, model