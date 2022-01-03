# ---------------------------------------------------------------------------------------------------------------------+
# This is the Test script for LCVNet Test
# Supporting reproduce result in Papar "A Novel Lightwieht Complex-valued Backbone Network for CSI Feedback"
# ---------------------------------------------------------------------------------------------------------------------+
import tensorflow as tf
from keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2D, Add, LeakyReLU, Activation, UpSampling2D,\
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
import complexnn
# install lib if needed

# Hardware Config
# Delete if you don't need
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # “0 or 1”
tf.reset_default_graph()
# auto storage allocation
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))


# Network Parameter
encoded_dim = 64    # 64 for compression rate 1/32, 128 for compression rate 1/16
data = 'indoor'    # 'indoor' for pico-cellular indoor-hall, 'Outdoor' for semi-urban outdoor
# image params
img_height = 32
img_width = 32
img_channels = 2
img_depth = 1
# 3D_conv_total = img_height * img_width
img_total = img_height * img_width * img_depth * img_channels


# LCVNetwork Component
def add_common_layers(y):
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)
    return y
# BN layer and Activation layer for Real Convolution


def add_complex_common_layers(y):
    y = complexnn.bn.ComplexBatchNormalization()(y)
    y = LeakyReLU()(y)
    return y
# BN layer and Activation layer for Complex Convolution


def FPB(y):
    y1 = Reshape((1, img_height, img_width, 2), name='Reshape_3D_RI_correlation')(y)
    y1 = Conv3D(filters=2, kernel_size=(1, 1, 2), padding='same', data_format="channels_first")(y1)
    y1 = add_common_layers(y1)
    y1 = Reshape((4, img_height, img_width), name='Reshape_2D_RI')(y1)
    # Pixel-level Feature Extraction Unit

    y2 = complexnn.conv.ComplexConv2D(filters=2, kernel_size=(1, 5), padding='same', data_format='channels_first')(y)
    y2 = add_complex_common_layers(y2)
    y2 = complexnn.conv.ComplexConv2D(filters=2, kernel_size=(5, 1), padding='same', data_format='channels_first')(y2)
    y2 = add_complex_common_layers(y2)
    # Region-level Feature Extraction Unit

    p1 = MaxPooling2D(pool_size=(2, 1), strides=(1, 1), padding='same', data_format='channels_first')(y)
    p2 = MaxPooling2D(pool_size=(1, 2), strides=(1, 1), padding='same', data_format='channels_first')(y)
    y3 = Add()([p1, p2])
    # Image-level Feature Argumentation Unit

    y = Add()([y1, y2])
    # Extraction Fusion

    return y, y3
# Feature Processing Block in Encoder side


# Multi-scale Feature Augmentation Encoder
def encoder(x):
    x, m_p = FPB(x)
    x = complexnn.conv.ComplexConv2D(filters=1, kernel_size=(1, 1), padding='same', data_format='channels_first')(x)
    x = Add()([x, m_p])
    x = add_complex_common_layers(x)
    # Dense-layer Compression Block
    x = Reshape((img_total,))(x)
    x = Dense(encoded_dim, activation='linear', name='encoded_layer')(x)
    encoded = Dropout(0.03, noise_shape=None, seed=None, input_shape=(encoded_dim,))(x)
    return encoded


# Multi-resolution X-shaped Reconstruction and Refinement Decoder
def decoder(encoded):
    x = Dense(img_total, activation='linear')(encoded)
    x = Reshape((2, img_height, img_width), name='reconstructed_image')(x)

    # Down-sampling & Reconstruction Block I(DRI)
    # 2*32*32 Image
    # Output Feature Channel Number: 72(total)
    x1 = complexnn.conv.ComplexConv2D(filters=12, kernel_size=(1, 7), padding='same', data_format="channels_first")(x)
    x1 = add_complex_common_layers(x1)
    x1_real, x1_image = Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': 2})(x1)
    x2 = complexnn.conv.ComplexConv2D(filters=12, kernel_size=(7, 1), padding='same', data_format="channels_first")(x)
    x2 = add_complex_common_layers(x2)
    x2_real, x2_image = Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': 2})(x2)
    x3 = complexnn.conv.ComplexConv2D(filters=12, kernel_size=(3, 3), padding='same', data_format="channels_first")(x)
    x3 = add_complex_common_layers(x3)
    x3_real, x3_image = Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': 2})(x3)
    x = concatenate([x1_real, x2_real, x3_real, x1_image, x2_image, x3_image], axis=1)
    short_cut1 = x

    # Down-sampling & Reconstruction Block II(DRII)
    # 2*16*16 Image
    # Output Feature Channel Number: 144(total)
    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_first')(x)
    x1 = complexnn.conv.ComplexConv2D(filters=24, kernel_size=(1, 5), padding='same', data_format="channels_first")(x)
    x1 = add_complex_common_layers(x1)
    x1_real, x1_image = Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': 2})(x1)
    x2 = complexnn.conv.ComplexConv2D(filters=24, kernel_size=(5, 1), padding='same', data_format="channels_first")(x)
    x2 = add_complex_common_layers(x2)
    x2_real, x2_image = Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': 2})(x2)
    x3 = complexnn.conv.ComplexConv2D(filters=24, kernel_size=(3, 3), padding='same', data_format="channels_first")(x)
    x3 = add_complex_common_layers(x3)
    x3_real, x3_image = Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': 2})(x3)
    x = concatenate([x1_real, x2_real, x3_real, x1_image, x2_image, x3_image], axis=1)
    short_cut2 = x

    # Decoder Bottleneck
    # 2*8*8 Image
    # Output Feature Channel Number: 288(total)
    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_first')(x)
    x = complexnn.conv.ComplexConv2D(filters=144, kernel_size=(3, 3), padding='same', data_format="channels_first")(x)
    x = add_complex_common_layers(x)

    # Up-sampling & Reconstruction Block I(URI)
    # 2*16*16 Image
    # channel number: 144(total)
    x = UpSampling2D(size=(2, 2), data_format='channels_first', interpolation='nearest')(x)
    x = complexnn.conv.ComplexConv2D(filters=72, kernel_size=(1, 1), padding='same', data_format="channels_first")(x)
    x = Add()([x, short_cut2])

    x1 = complexnn.conv.ComplexConv2D(filters=24, kernel_size=(1, 5), padding='same', data_format="channels_first")(x)
    x1 = add_complex_common_layers(x1)
    x1_real, x1_image = Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': 2})(x1)
    x2 = complexnn.conv.ComplexConv2D(filters=24, kernel_size=(5, 1), padding='same', data_format="channels_first")(x)
    x2 = add_complex_common_layers(x2)
    x2_real, x2_image = Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': 2})(x2)
    x3 = complexnn.conv.ComplexConv2D(filters=24, kernel_size=(3, 3), padding='same', data_format="channels_first")(x)
    x3 = add_complex_common_layers(x3)
    x3_real, x3_image = Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': 2})(x3)
    x = concatenate([x1_real, x2_real, x3_real, x1_image, x2_image, x3_image], axis=1)

    # Up-sampling & Reconstruction Block II(URII)
    # 2*32*32 Image
    # channel number: 72(total)
    x = UpSampling2D(size=(2, 2), data_format='channels_first', interpolation='nearest')(x)
    x = complexnn.conv.ComplexConv2D(filters=36, kernel_size=(1, 1), padding='same', data_format="channels_first")(x)
    x = Add()([x, short_cut1])

    x1 = complexnn.conv.ComplexConv2D(filters=12, kernel_size=(1, 3), padding='same', data_format="channels_first")(x)
    x1 = add_complex_common_layers(x1)
    x1_real, x1_image = Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': 2})(x1)
    x2 = complexnn.conv.ComplexConv2D(filters=12, kernel_size=(3, 1), padding='same', data_format="channels_first")(x)
    x2 = add_complex_common_layers(x2)
    x2_real, x2_image = Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': 2})(x2)
    x3 = complexnn.conv.ComplexConv2D(filters=12, kernel_size=(3, 3), padding='same', data_format="channels_first")(x)
    x3 = add_complex_common_layers(x3)
    x3_real, x3_image = Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': 2})(x3)
    x = concatenate([x1_real, x2_real, x3_real, x1_image, x2_image, x3_image], axis=1)

    # Regression
    x = complexnn.conv.ComplexConv2D(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same',
                                     data_format="channels_first")(x)
    return x


image_tensor = Input(shape=(img_channels, img_height, img_width))
codewords_vector = Input(shape=(encoded_dim,))

encoder = Model(inputs=[image_tensor], outputs=[encoder(image_tensor)])
decoder = Model(inputs=[codewords_vector], outputs=[decoder(codewords_vector)])
autoencoder = Model(inputs=[image_tensor], outputs=[decoder(encoder(image_tensor))])
autoencoder.compile(optimizer='adam', loss='mse')
print(encoder.summary())
print(decoder.summary())
print(autoencoder.summary())

# Trained model path
autoencoder.load_weights("/loading the path to model")

# Data loading
if data == 'indoor':
    mat_test = hdf5storage.loadmat("/DATA_Htestin.mat")
    x_test = mat_test['HT']

if data == 'Outdoor':
    mat_test = hdf5storage.loadmat("/DATA_Htestout.mat")
    x_test = mat_test['HT']

x_test = x_test.astype('float32')
x_test = np.reshape(x_test, (len(x_test), img_channels, img_height, img_width))
# adapt this if using `channels_first` image data format
# shape in batch_2_32_32

# Testing data
tStart = time.time()
x_hat = autoencoder.predict(x_test)
tEnd = time.time()
print("It cost %f sec" % ((tEnd - tStart)/x_test.shape[0]))

# Calcaulating the NMSE and rho
if data == 'indoor':
    mat = sio.loadmat('/DATA_HtestFin_all.mat')
    X_test = mat['HF_all']# array

elif data == 'Outdoor':
    mat = sio.loadmat('/DATA_HtestFout_all.mat')
    X_test = mat['HF_all']# array

X_test = np.reshape(X_test, (len(X_test), img_height, 125))
x_test_real = np.reshape(x_test[:, 0, :, :], (len(x_test), -1))
x_test_imag = np.reshape(x_test[:, 1, :, :], (len(x_test), -1))
x_test_C = x_test_real-0.5 + 1j*(x_test_imag-0.5)
x_hat_real = np.reshape(x_hat[:, 0, :, :], (len(x_hat), -1))
x_hat_imag = np.reshape(x_hat[:, 1, :, :], (len(x_hat), -1))
x_hat_C = x_hat_real-0.5 + 1j*(x_hat_imag-0.5)
x_hat_F = np.reshape(x_hat_C, (len(x_hat_C), img_height, img_width))
X_hat = np.fft.fft(np.concatenate((x_hat_F, np.zeros((len(x_hat_C), img_height, 257-img_width))), axis=2), axis=2)
X_hat = X_hat[:, :, 0:125]

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
print("MSE is ", 10 * math.log10(np.mean(mse)))
print("NMSE is ", 10*math.log10(np.mean(mse/power)))
print("Correlation is ", np.mean(rho))


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