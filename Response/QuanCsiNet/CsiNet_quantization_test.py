# ---------------------------------------------------------------------------------------------------------------------+
# This is the Test script for CsiNet with Quantization
# Supporting reporting result in Response of "A Novel Lightwieht Complex-valued Backbone Network for CSI Feedback"
# Hzl 2022.1
# ---------------------------------------------------------------------------------------------------------------------+
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU, Activation, \
    subtract, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, Callback
from tensorflow.keras.callbacks import ModelCheckpoint
import scipy.io as sio
import numpy as np
import random
import math
import time
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

import os

os.environ['CUDA_VISIBLE_DEVICES'] = "1"  # Your GPU device

# 40%
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config=config)

encoded_dim = 512  # 64 for compression rate 1/32, 128 for compression rate 1/16, 256 for compression rate 1/8, 512 for compression rate 1/4.
envir = 'outdoor'  # 'indoor' for pico-cellular indoor-hall, 'Outdoor' for semi-urban outdoor


# quantization setting
bits = 3
B = bits

img_height = 32
img_width = 32
img_channels = 2
img_total = img_height * img_width * img_channels
residual_num = 2


@tf.custom_gradient
def QuantizationOp(x):
    step = tf.cast((2 ** B), dtype=tf.float32)
    result = tf.cast((tf.round(x * step - 0.5)), dtype=tf.float32)

    def custom_grad(dy):
        grad = dy * 1
        return grad

    return result, custom_grad


class QuantizationLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(QuantizationLayer, self).__init__()

    def call(self, x):
        quantized = QuantizationOp(x)
        return quantized

    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        base_config = super(QuantizationLayer, self).get_config()
        return base_config


@tf.custom_gradient
def DequantizationOp(x):
    step = tf.cast((2 ** B), dtype=tf.float32)
    result = tf.cast((x + 0.5) / step, dtype=tf.float32)

    def custom_grad(dy):
        grad = dy * 1
        return grad

    return result, custom_grad


class DequantizationLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DequantizationLayer, self).__init__()

    def call(self, x):
        dequantized = DequantizationOp(x)
        return dequantized

    def get_config(self):
        base_config = super(DequantizationLayer, self).get_config()
        return base_config


# Bulid the autoencoder model of CsiNet
def add_common_layers(y):
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)
    return y


def residual_block_decoded(y):
    shortcut = y
    y = Conv2D(8, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
    y = add_common_layers(y)

    y = Conv2D(16, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
    y = add_common_layers(y)

    y = Conv2D(2, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
    y = BatchNormalization()(y)

    y = add([shortcut, y])
    y = LeakyReLU()(y)

    return y


def encoder_network(x):
    x = Conv2D(2, (3, 3), padding='same', data_format="channels_first")(x)
    x = add_common_layers(x)

    x = Reshape((img_total,))(x)
    encoded = Dense(encoded_dim, activation='sigmoid', name='encoded_layer')(x)  # encoded result "sigmoid"
    encoded = QuantizationLayer()(encoded)
    return encoded


def decoder_network(encoded):
    # decoder
    encoded = DequantizationLayer()(encoded)
    encoded = Reshape((encoded_dim,))(encoded)

    x = Dense(img_total, activation='sigmoid')(encoded)
    x = Reshape((img_channels, img_height, img_width,), name='reconstructed_image')(x)

    for i in range(residual_num):
        x = residual_block_decoded(x)

    x = Conv2D(2, (3, 3), activation='sigmoid', padding='same', data_format="channels_first")(x)

    return x


image_tensor = keras.Input(shape=(img_channels, img_height, img_width))
decoder_input = keras.Input(shape=(encoded_dim,))

encoder = keras.Model(inputs=[image_tensor], outputs=[encoder_network(image_tensor)])
decoder = keras.Model(inputs=[decoder_input], outputs=[decoder_network(decoder_input)])
autoencoder = keras.Model(inputs=[image_tensor], outputs=[decoder(encoder(image_tensor))])

print(encoder.summary())
print(decoder.summary())
print(autoencoder.summary())

if envir == 'indoor':
    mat = sio.loadmat('--/DATA_Htrainin.mat')
    x_train = mat['HT']
    mat = sio.loadmat('--/DATA_Hvalin.mat')
    x_val = mat['HT']
    mat = sio.loadmat('--/DATA_Htestin.mat')
    x_test = mat['HT']


elif envir == 'outdoor':
    mat = sio.loadmat('--/DATA_Htrainout.mat')
    x_train = mat['HT']
    mat = sio.loadmat('--/DATA_Hvalout.mat')
    x_val = mat['HT']
    mat = sio.loadmat('--/DATA_Htestout.mat')
    x_test = mat['HT']

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')

x_train = np.reshape(x_train, [len(x_train), img_channels, img_height, img_width])
x_val = np.reshape(x_val, [len(x_val), img_channels, img_height, img_width])
x_test = np.reshape(x_test, [len(x_test), img_channels, img_height, img_width])

autoencoder.load_weights('--/path_model.h5')  #load the trained model here

tStart = time.time()
x_hat = autoencoder.predict(x_test)
tEnd = time.time()
print("It cost %f sec" % ((tEnd - tStart) / x_test.shape[0]))

# Calcaulating the NMSE and rho
if envir == 'indoor':
    mat = sio.loadmat('--/DATA_HtestFin_all.mat')
    X_test = mat['HF_all']  # array

elif envir == 'outdoor':
    mat = sio.loadmat('--/DATA_HtestFout_all.mat')
    X_test = mat['HF_all']  # array

X_test = np.reshape(X_test, (len(X_test), img_height, 125))
x_test_real = np.reshape(x_test[:, 0, :, :], (len(x_test), -1))
x_test_imag = np.reshape(x_test[:, 1, :, :], (len(x_test), -1))
x_test_C = x_test_real - 0.5 + 1j * (x_test_imag - 0.5)
x_hat_real = np.reshape(x_hat[:, 0, :, :], (len(x_hat), -1))
x_hat_imag = np.reshape(x_hat[:, 1, :, :], (len(x_hat), -1))
x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)
x_hat_F = np.reshape(x_hat_C, (len(x_hat_C), img_height, img_width))
X_hat = np.fft.fft(np.concatenate((x_hat_F, np.zeros((len(x_hat_C), img_height, 257 - img_width))), axis=2), axis=2)
X_hat = X_hat[:, :, 0:125]

n1 = np.sqrt(np.sum(np.conj(X_test) * X_test, axis=1))
n1 = n1.astype('float64')
n2 = np.sqrt(np.sum(np.conj(X_hat) * X_hat, axis=1))
n2 = n2.astype('float64')
aa = abs(np.sum(np.conj(X_test) * X_hat, axis=1))
rho = np.mean(aa / (n1 * n2), axis=1)
X_hat = np.reshape(X_hat, (len(X_hat), -1))
X_test = np.reshape(X_test, (len(X_test), -1))
power = np.sum(abs(x_test_C) ** 2, axis=1)
power_d = np.sum(abs(X_hat) ** 2, axis=1)
mse = np.sum(abs(x_test_C - x_hat_C) ** 2, axis=1)

print("In " + envir + " environment")
print("When dimension is", encoded_dim)
print("quantization bits is", bits)
print("NMSE is ", 10 * math.log10(np.mean(mse / power)))
print("Correlation is ", np.mean(rho))