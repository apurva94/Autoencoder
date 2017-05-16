import os
#os.environ(KERAS_BACKEND='tensorflow')
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
#from keras.callbacks import TensorBoard
from keras import metrics


#Parameters
n_epochs=50
noise_factor = 0.2
conv1_filterSize=(3,3)
conv1_numfilters=30
maxpool1_size=(2,2)
conv2_filterSize=(3,3)
conv2_numfilters=30
maxpool2_size=(2,2)

print('n_epochs=',n_epochs)
print('noise_factor =',noise_factor)
print('conv1_filterSize=',conv1_filterSize)
print('conv1_numfilters=',conv1_numfilters)
print('maxpool1_size=',maxpool1_size)
print('conv2_filterSize=',conv2_filterSize)
print('conv2_numfilters=',conv2_numfilters)
print('maxpool2_size=',maxpool2_size)


K.set_image_data_format('channels_last')
# img_row=32
# img_col=32
img_row=28
img_col=28
(x_train, _), (x_test, _) = mnist.load_data()
# (x_train, _), (x_test, _) = cifar10.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), img_row, img_col, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), img_row, img_col, 1)) 
# x_train = np.reshape(x_train, (len(x_train), img_row, img_col, 3))  # adapt this if using `channels_first` image data format
# x_test = np.reshape(x_test, (len(x_test), img_row, img_col, 3))  # adapt this if using `channels_first` image data format

#noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

#Visualizing CIFAR 10
n = 10
plt.figure(figsize=(20, 2))
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(x_test_noisy[i].reshape(img_row, img_col))
    #plt.imshow(x_test_noisy[i].reshape(img_row, img_col,3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

input_img = Input(shape=(img_row, img_col,1))
# input_img = Input(shape=(img_row, img_col, 3))  # adapt this if using `channels_first` image data format

x = Conv2D(conv1_numfilters, conv1_filterSize, activation='relu', padding='same')(input_img)
x = MaxPooling2D(maxpool1_size, padding='same')(x)
x = Conv2D(conv2_numfilters, conv2_filterSize, activation='relu', padding='same')(x)
encoded = MaxPooling2D(maxpool2_size, padding='same')(x)


x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
# decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.summary()
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy',metrics=[metrics.binary_accuracy])

autoencoder.fit(x_train_noisy, x_train,
                epochs=n_epochs,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))
                

early_stopping = EarlyStopping(monitor='val_loss', patience=100)
model_checkpoint = ModelCheckpoint(os.path.join('/home/microway/Desktop/Apurva/autoencoder/logs',"weights_spm2",'weights_inouts.{val_loss:.2f}.hdf5'), monitor='val_loss', verbose=0, save_best_only=False)

decoded_imgs = autoencoder.predict(x_test_noisy)
n = 10
plt.figure(figsize=(20, 2))
for i in range(n):
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(img_row, img_col))
    #plt.imshow(decoded_imgs[i].reshape(img_row, img_col,3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()            