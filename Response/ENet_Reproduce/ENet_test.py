# ---------------------------------------------------------------------------------------------------------------------+
# This is the Test script for ENet Test
# Supporting reproduce result in papar "A Lightwieht deep network for efficient CSI feedback in massive MIMO Systems"
# ---------------------------------------------------------------------------------------------------------------------+
import tensorflow as tf
from keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2D, Conv2DTranspose, Add, LeakyReLU, Activation, UpSampling2D,\
    Conv3D, concatenate, Lambda, MaxPooling2D, Dropout
from keras.models import Model
from keras.callbacks import TensorBoard, Callback, ModelCheckpoint
import scipy.io as sio
import numpy as np
import math
import time
import hdf5storage  # load Matlab data bigger than 2GB
import keras
import os
# install lib if needed


# Hardware Config
# Delete if you don't need
os.environ["CUDA_VISIBLE_DEVICES"] = "X"  # cuda device
tf.reset_default_graph()
# auto storage allocation
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))


# Network Parameter
# The dense params need change
encoded_dim = 256    # 32 for compression rate 1/32, 64 for compression rate 1/16, 128 for compression rate 1/8, 256 for compression rate 1/4.
data = 'Outdoor'    # 'indoor' for pico-cellular indoor-hall, 'Outdoor' for semi-urban outdoor
test_dim = encoded_dim*2
f = 32  # 16 or 32



# image params
img_height = 32
img_width = 32
img_channels = 1
# Real and Imaginary total params
img_total = img_height * img_width * img_channels
path_img = img_total//2



# ENetwork Component
def add_common_layers(y):
    y = LeakyReLU()(y)
    y = BatchNormalization()(y)
    return y
# BN layer and Activation layer for Real Convolution


def ENet_encoder(x):
    x = Conv2D(filters=f, kernel_size=(3, 5), strides=(1, 2), padding='same', data_format='channels_first')(x)
    x = add_common_layers(x)
    x = Conv2D(filters=f, kernel_size=(3, 3), padding='same', data_format='channels_first')(x)
    x = add_common_layers(x)
    x = Conv2D(filters=f, kernel_size=(3, 3), padding='same', data_format='channels_first')(x)
    x = add_common_layers(x)
    x = Conv2D(filters=1, kernel_size=(3, 3), padding='same', data_format='channels_first')(x)
    x = add_common_layers(x)
    x = Reshape((path_img,))(x)
    encoded = Dense(encoded_dim, activation='linear', name='encoded_layer')(x)
    return encoded


def ENet_decoder(encoded):
    x = Dense(path_img, activation='linear', name='decoded_layer')(encoded)
    x = Reshape((1, img_height, img_width//2), name='reconstructed_image')(x)
    x = Conv2DTranspose(filters=f, kernel_size=(3, 3), padding='same', data_format='channels_first')(x)
    x = add_common_layers(x)
    x = Conv2DTranspose(filters=f, kernel_size=(3, 3), padding='same', data_format='channels_first')(x)
    x = add_common_layers(x)
    x = Conv2DTranspose(filters=f, kernel_size=(3, 3), padding='same', data_format='channels_first')(x)
    x = add_common_layers(x)
    x = Conv2DTranspose(filters=1, kernel_size=(3, 5), strides=(1, 2), padding='same', data_format='channels_first')(x)
    return x


image_tensor = Input(shape=(img_channels, img_height, img_width))
codewords_vector = Input(shape=(encoded_dim,))

encoder = Model(inputs=[image_tensor], outputs=[ENet_encoder(image_tensor)])
decoder = Model(inputs=[codewords_vector], outputs=[ENet_decoder(codewords_vector)])
autoencoder = Model(inputs=[image_tensor], outputs=[decoder(encoder(image_tensor))])
autoencoder.compile(optimizer='adam', loss='mse')
print(encoder.summary())
print(decoder.summary())
print(autoencoder.summary())

# Data loading
if data == 'indoor':
    mat_test = hdf5storage.loadmat("/DATA_Htestin.mat")
    x_test = mat_test['HT']

if data == 'Outdoor':
    mat_test = hdf5storage.loadmat("/DATA_Htestout.mat")
    x_test = mat_test['HT']

x_test = x_test.astype('float32')
x_test = np.reshape(x_test, (len(x_test), 2, img_height, img_width))
# adapt this if using `channels_first` image data format
# shape in batch_2_32_32

x_test_r = x_test[:, 0:1, :, :]
x_test_i = x_test[:, 1:2, :, :]

# Trained model path
autoencoder.load_weights("/F32/ENet_dim512_withOutdoor_filter32_model.h5")  # 512_outdoor_F32 here

# Testing data
tStart = time.time()
x_hat_r = autoencoder.predict([x_test_r])
x_hat_i = autoencoder.predict([x_test_i])
tEnd = time.time()
print("It cost %f sec" % ((tEnd - tStart)/x_test.shape[0]))

# Calcaulating the NMSE and rho
if data == 'indoor':
    mat = sio.loadmat('/DATA_HtestFin_all.mat')
    X_test = mat['HF_all']  # array

elif data == 'Outdoor':
    mat = sio.loadmat('/DATA_HtestFout_all.mat')
    X_test = mat['HF_all']  # array
#
print(X_test.shape)
X_test = np.reshape(X_test, (len(X_test), img_height, 125))
x_test_r = np.reshape(x_test_r, (len(x_test_r), -1))
x_test_i = np.reshape(x_test_i, (len(x_test_i), -1))
x_test_C = x_test_r-0.5 + 1j*(x_test_i-0.5)
x_hat_r = np.reshape(x_hat_r, (len(x_hat_r), -1))
x_hat_i = np.reshape(x_hat_i, (len(x_hat_i), -1))
x_hat_C = x_hat_r-0.5 + 1j*(x_hat_i-0.5)
x_hat_F = np.reshape(x_hat_C, (len(x_hat_C), img_height, img_width))
X_hat = np.fft.fft(np.concatenate((x_hat_F, np.zeros((len(x_hat_C), img_height, 257-img_width))), axis=2), axis=2)
X_hat = X_hat[:, :, 0:125]

print("Test real dataset shape is ", x_test_r.shape)
print("Test imaginary dataset shape is", x_test_i.shape)

power_r = np.sum(abs(x_test_r)**2, axis=1)
power_i = np.sum(abs(x_test_i)**2, axis=1)

mse_r = np.sum(abs(x_test_r - x_hat_r) ** 2, axis=1)
mse_i = np.sum(abs(x_test_i - x_hat_i) ** 2, axis=1)

n1 = np.sqrt(np.sum(np.conj(X_test)*X_test, axis=1))
n1 = n1.astype('float64')
n2 = np.sqrt(np.sum(np.conj(X_hat)*X_hat, axis=1))
n2 = n2.astype('float64')
aa = abs(np.sum(np.conj(X_test)*X_hat, axis=1))
rho = np.mean(aa/(n1*n2), axis=1)
X_hat = np.reshape(X_hat, (len(X_hat), -1))
X_test = np.reshape(X_test, (len(X_test), -1))
power = np.sum(abs(x_test_C)**2, axis=1)
power_d = np.sum(abs(X_hat)**2, axis=1)
mse = np.sum(abs(x_test_C-x_hat_C)**2, axis=1)

print("In "+data+" environment")
print("When dimension is", encoded_dim)
print("MSE_r is ", 10 * math.log10(np.mean(mse_r)))
print("MSE_i is ", 10 * math.log10(np.mean(mse_i)))
print("MSE is ", 10 * math.log10(np.mean(mse)))
print("NMSE_r is ", 10 * math.log10(np.mean(mse_r/power_r)))
print("NMSE_i is ", 10 * math.log10(np.mean(mse_i/power_i)))
print("Correlation is ", np.mean(rho))
print("NMSE is ", 10 * math.log10(np.mean(mse/power)))


# import matplotlib.pyplot as plt
# '''abs'''
# n = 10
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # display origoutal
#     ax = plt.subplot(2, n, i + 1 )
#     x_testplo = abs(x_test[i, 0, :, :]-0.5 + 1j*(x_test[i, 1, :, :]-0.5))
#     plt.imshow(np.max(np.max(x_testplo))-x_testplo.T)
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     ax.invert_yaxis()
#     # display reconstruction
#     ax = plt.subplot(2, n, i + 1 + n)
#     decoded_imgsplo = abs(x_hat[i, 0, :, :]-0.5
#                           + 1j*(x_hat[i, 1, :, :]-0.5))
#     plt.imshow(np.max(np.max(decoded_imgsplo))-decoded_imgsplo.T)
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     ax.invert_yaxis()
# plt.show()