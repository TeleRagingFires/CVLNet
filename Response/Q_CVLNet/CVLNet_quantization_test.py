# ---------------------------------------------------------------------------------------------------------------------+
# This is the Training script for CVLNet with Quantization
# Supporting reporting result in Response of "A Novel Lightwieht Complex-valued Backbone Network for CSI Feedback"
# ---------------------------------------------------------------------------------------------------------------------+
import tensorflow as tf
from keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2D, Add, LeakyReLU, Activation, UpSampling2D,\
    Conv3D, concatenate, Lambda, MaxPooling2D, Dropout, Layer
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

os.environ['CUDA_VISIBLE_DEVICES'] = "X"  # Your GPU device


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.25
session = tf.Session(config=config)

encoded_dim = 512   # # 64 for compression rate 1/32, 128 for compression rate 1/16, 256 for compression rate 1/8, 512 for compression rate 1/4.
envir = 'outdoor'  # 'indoor' for pico-cellular indoor-hall, 'Outdoor' for semi-urban outdoor


# quantization setting
bits = 3
B = bits

img_height = 32
img_width = 32
img_channels = 2
img_total = img_height * img_width * img_channels



@tf.custom_gradient
def QuantizationOp(x):
    step = tf.cast((2 ** B), dtype=tf.float32)
    result = tf.cast((tf.round(x * step - 0.5)), dtype=tf.float32)

    def custom_grad(dy):
        grad = dy * 1
        return grad

    return result, custom_grad


class QuantizationLayer(Layer):
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


class DequantizationLayer(Layer):
    def __init__(self, **kwargs):
        super(DequantizationLayer, self).__init__()

    def call(self, x):
        dequantized = DequantizationOp(x)
        return dequantized

    def get_config(self):
        base_config = super(DequantizationLayer, self).get_config()
        return base_config


# Bulid the autoencoder model of CVLNet
def add_common_layers(y):
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)
    return y

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



def encoder_network(x):
    # encoder of CVLNet
    x, m_p = FPB(x)
    x = complexnn.conv.ComplexConv2D(filters=1, kernel_size=(1, 1), padding='same', data_format='channels_first')(x)
    x = Add()([x, m_p])
    x = add_complex_common_layers(x)
    # Dense-layer Compression Block
    x = Reshape((img_total,))(x)
    x = Dense(encoded_dim, activation='sigmoid', name='encoded_layer')(x)
    x = Dropout(0.03, noise_shape=None, seed=None, input_shape=(encoded_dim,))(x)
    encoded = QuantizationLayer()(x)
    return encoded



def decoder_network(encoded):
    # decoder of CVLNet
    encoded = DequantizationLayer()(encoded)
    encoded = Reshape((encoded_dim,))(encoded)
    x = Dense(img_total, activation='sigmoid')(encoded)
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
decoder_input = Input(shape=(encoded_dim,))

encoder = Model(inputs=[image_tensor], outputs=[encoder_network(image_tensor)])
decoder = Model(inputs=[decoder_input], outputs=[decoder_network(decoder_input)])
autoencoder = Model(inputs=[image_tensor], outputs=[decoder(encoder(image_tensor))])
autoencoder.compile(optimizer='adam', loss='mse')


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

autoencoder.load_weights('--/path_model.h5')  # load the trained model here

tStart = time.time()
x_hat = autoencoder.predict(x_test)
tEnd = time.time()
print("It cost %f sec" % ((tEnd - tStart) / x_test.shape[0]))

# Calcaulating the NMSE and rho
if envir == 'indoor':
    mat = sio.loadmat('--/DATA_HtestFin_all.mat')
    X_test = mat['HF_all']# array

elif envir == 'outdoor':
    mat = sio.loadmat('--/DATA_HtestFout_all.mat')
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

print("In "+envir+" environment")
print("When dimension is", encoded_dim)
print("quantization bits is", bits)
print("NMSE is ", 10*math.log10(np.mean(mse/power)))
print("Correlation is ", np.mean(rho))