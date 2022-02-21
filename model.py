from tqdm import tqdm

import tensorflow as tf
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Layer, Dropout, LayerNormalization, Conv1D, Concatenate, GlobalAveragePooling1D, Input, GlobalMaxPooling1D

from layers import Time2Vector, TransformerEncoder, CrossAttention


def create_model(batch_size, seq_len, d_k, d_v, n_heads, ff_dim):
    conv1 = Conv1D(32, 2, padding='causal', dilation_rate=1, activation='relu')
    conv2 = Conv1D(32, 2, padding='causal', dilation_rate=2, activation='relu')
    conv3 = Conv1D(32, 2, padding='causal', dilation_rate=4, activation='relu')
    conv4 = Conv1D(32, 2, padding='causal', dilation_rate=8, activation='relu')
    conv5 = Conv1D(32, 2, padding='causal', dilation_rate=16, activation='relu')
    conv6 = Conv1D(32, 2, padding='causal', dilation_rate=32, activation='relu')
    conv_lnorm = LayerNormalization(input_shape=(batch_size, seq_len, 32), epsilon=1e-6)
    ob_time_embedding = Time2Vector(seq_len)
    tr_time_embedding = Time2Vector(seq_len)
    ob_attn_layer1 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)

    tr_attn_layer1 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)

    cross_attn_layer = CrossAttention(d_k, d_v, n_heads, ff_dim)

    '''Construct model'''
    ob_in_seq = Input(shape=(seq_len, 100))
    tr_in_seq = Input(shape=(seq_len, 3))

    x_conv = conv1(ob_in_seq)
    x_conv = conv2(x_conv)
    x_conv = conv3(x_conv)
    x_conv = conv4(x_conv)
    x_conv = conv5(x_conv)
    x_conv = conv6(x_conv)

    x_conv = conv_lnorm(x_conv)
    ob_pos = ob_time_embedding(tf.math.reduce_mean(ob_in_seq[:, :, ::2], axis=-1))
    ob_x = Concatenate(axis=-1)([x_conv, ob_pos])
    ob_x = ob_attn_layer1((ob_x, ob_x, ob_x))

    tr_pos = tr_time_embedding(tr_in_seq[:, :, 1])
    tr_x = Concatenate(axis=-1)([tr_in_seq, tr_pos])
    tr_x = tr_attn_layer1((tr_x, tr_x, tr_x))

    cross = cross_attn_layer((ob_x, tr_x))

    x = GlobalAveragePooling1D(data_format='channels_last')(cross)
    x = Dense(128, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[ob_in_seq, tr_in_seq], outputs=out)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model
