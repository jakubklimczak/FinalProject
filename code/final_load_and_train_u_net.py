import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Dropout, Conv3DTranspose, concatenate
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os

mri_img_path = '../resources/dataset/BraTS2021_Training_Data/'
dim = 128

def load_nifti_file(file_path):
    nifti = nib.load(file_path)
    image = nifti.get_fdata()
    return image


def unet_3d(input_shape=(None, None, None, 4)):
    inputs = Input(input_shape)
    
    # Encoder
    conv1 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    
    conv2 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    
    conv3 = Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    
    conv4 = Conv3D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv3D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(drop4)
    
    # Bottom
    conv5 = Conv3D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv3D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    # Decoder
    up6 = Conv3DTranspose(512, 2, strides=(2, 2, 2), padding='same')(drop5)
    merge6 = concatenate([drop4, up6], axis=4)
    conv6 = Conv3D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv3D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    
    up7 = Conv3DTranspose(256, 2, strides=(2, 2, 2), padding='same')(conv6)
    merge7 = concatenate([conv3, up7], axis=4)
    conv7 = Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    
    up8 = Conv3DTranspose(128, 2, strides=(2, 2, 2), padding='same')(conv7)
    merge8 = concatenate([conv2, up8], axis=4)
    conv8 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    
    up9 = Conv3DTranspose(64, 2, strides=(2, 2, 2), padding='same')(conv8)
    merge9 = concatenate([conv1, up9], axis=4)
    conv9 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    
    outputs = Conv3D(1, 1, activation='sigmoid')(conv9)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model

# Testing the model
model_3d = unet_3d(input_shape=(None, None, None, 4))
model_3d.summary()

def load_data_for_patient(path):
    flair_file = [f for f in os.listdir(path) if f.endswith('flair.nii.gz')][0]
    t1_file = [f for f in os.listdir(path) if f.endswith('t1.nii.gz')][0]
    t1ce_file = [f for f in os.listdir(path) if f.endswith('t1ce.nii.gz')][0]
    t2_file = [f for f in os.listdir(path) if f.endswith('t2.nii.gz')][0]
    flair_img = load_nifti_file(os.path.join(path, flair_file))
    t1_img = load_nifti_file(os.path.join(path, t1_file))
    t1ce_img = load_nifti_file(os.path.join(path, t1ce_file))
    t2_img = load_nifti_file(os.path.join(path, t2_file))

    # single NumPy tensor as output
    data_tensor = np.stack([flair_img, t1_img, t1ce_img, t2_img], axis=-1)

    return data_tensor

data_tensor = load_data_for_patient(mri_img_path + 'BraTS2021_00000')
print(data_tensor.shape)
#model_3d.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val))