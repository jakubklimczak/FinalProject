import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # to prevent walls of annoying errors about lack of NUMA support of my GPU

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import pydicom


train_dir = '../resources/dataset/train'
test_dir = '../resources/dataset/test'
label_csv_dir = "../resources/dataset/train_labels.csv"

def load_and_preprocess_dicom_images(file_path, target_size):
    dicom_data = pydicom.dcmread(file_path)
    image_array = dicom_data.pixel_array
    # Resize the image if needed
    resized_image = tf.image.resize(image_array, target_size)
    # Normalize pixel values
    normalized_image = resized_image / 255.0
    return normalized_image


image_size = (128, 128)  #image size
batch_size = 32
labels_df = pd.read_csv(label_csv_dir, dtype={'BraTS21ID': str})
labels_df['BraTS21ID'] = labels_df['BraTS21ID'].apply(lambda x: os.path.join(train_dir, x))

print(labels_df)

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, preprocessing_function=lambda x: load_and_preprocess_dicom_images(x, image_size))
train_generator = train_datagen.flow_from_dataframe(dataframe=labels_df, x_col='BraTS21ID', y_col='MGMT_value', target_size=image_size, batch_size=batch_size, class_mode='raw')

test_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=lambda x: load_and_preprocess_dicom_images(x, image_size))
test_generator = test_datagen.flow_from_directory(test_dir, target_size=image_size, batch_size=batch_size, class_mode='binary')


model = tf.keras.models.Sequential([
#setup for later - here will be the model
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train your model using the TensorFlow Dataset
history = model.fit(train_generator, epochs=10, validation_data=test_generator)

# Save your model if needed
model.save('initial_model.h5')
