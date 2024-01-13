import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # to prevent walls of annoying errors about lack of NUMA support of my GPU

import warnings

# Suppress Keras UserWarning about invalid image filenames - happens with DICOM images and gets annoying
warnings.simplefilter("ignore", UserWarning)

import gc
import time
import pandas as pd
import pydicom
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage import exposure
from tqdm import tqdm
import psutil

# Global variables
images_processed = 0

#DEBUG
def print_memory_usage():
    memory_info = psutil.virtual_memory()
    print(f"Total: {memory_info.total} bytes")
    print(f"Available: {memory_info.available} bytes")
    print(f"Used: {memory_info.used} bytes")
    print(f"Percentage Used: {memory_info.percent}%")
    print("")
#END DEBUG

# ImageDataGenerator works really slowly and I want to see if it is working or not - this shows a progress bar while preparing the generator.
class CustomDataFrameGenerator(ImageDataGenerator):
    def flow_from_dataframe_with_progress(self, dataframe, **kwargs):
        generator = super().flow_from_dataframe(dataframe, **kwargs)
        num_batches = len(generator)
        custom_bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        for batch_index in tqdm(range(num_batches), position=0, leave=True, bar_format=custom_bar_format):
            yield next(generator)

# Function to load and preprocess DICOM images
def load_dicom_image_and_preprocess_for_tf(file_path):
    global images_processed
    images_processed += 1
    try: 
        with open(file_path, 'rb') as file:
            dicom_data = pydicom.dcmread(file)
            image_array = dicom_data.pixel_array
            uin8_image_array = exposure.rescale_intensity(image_array, in_range='uint16', out_range='uint8')
            #print("Converted the image: " + file_path)
            return uin8_image_array
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None
    finally:
        file.close()
        if images_processed%1500==0:
            print_memory_usage()
            

# Directory paths
train_dir = '../resources/dataset/train'
test_dir = '../resources/dataset/test'
label_csv_dir = "../resources/dataset/train_labels.csv"

# Constants
image_size = (512, 512) # all images should be this size already - that's why I abandoned most image checks in my code. If you encounter an issues, keep in mind to rescale images to 512x512
batch_size = 32 #32

# Read the CSV file and preprocess 'BraTS21ID' column
labels_df = pd.read_csv(label_csv_dir, dtype={'BraTS21ID': str})
labels_df['BraTS21ID'] = labels_df['BraTS21ID'].apply(lambda x: os.path.join(train_dir, x))

def create_rows(row):
    base_path = row['BraTS21ID']
    mgmt_value = row['MGMT_value']
    new_rows = []
    for modality in ['/FLAIR', '/T1w', '/T1wCE', '/T2w']:
        modality_path = base_path + modality
        for filename in os.listdir(modality_path):
            if filename.endswith('.dcm'):
                file_path = os.path.join(modality_path, filename)
                new_row = {'BraTS21ID': file_path, 'MGMT_value': mgmt_value}
                new_rows.append(new_row)
    return new_rows


# Apply the function to each row and explode the DataFrame
labels_df = labels_df.apply(create_rows, axis=1).explode().reset_index(drop=True)

# Debug

#print(labels_df.head())
labels_df.to_csv('debug_output.txt', sep='\n', index=False)

labels_df = labels_df.apply(pd.Series)
labels_df.columns = ['BraTS21ID', 'MGMT_value']
labels_df['BraTS21ID'] = labels_df['BraTS21ID'].astype('string')
labels_df['MGMT_value'] = labels_df['MGMT_value'].astype('string')

#print(labels_df.head())


# Debug
'''
print(labels_df.columns)
print(labels_df['BraTS21ID'].dtype)
print(labels_df['MGMT_value'].dtype)

print(labels_df['BraTS21ID'][0])

for x_col in labels_df['BraTS21ID']:
    if os.path.exists(x_col):
        print(f"Image file '{x_col}' found in directory.")
    else:
        print(f"Image file '{x_col}' not found in directory.")
'''


# Custom version
gc.collect()
start_time = time.time()
print("Clock started")
custom_generator = CustomDataFrameGenerator()
timestamp1 = time.time()
print(f"Got here! Time elapsed: {timestamp1 - start_time}")
processed_data = labels_df['BraTS21ID'].apply(load_dicom_image_and_preprocess_for_tf)
timestamp2 = time.time()
print(f"Got here! Time elapsed: {timestamp2 - timestamp1}")
processed_df = pd.DataFrame({'BraTS21ID': processed_data, 'MGMT_value': labels_df['MGMT_value']})
timestamp3 = time.time()
print(f"Got here! Time elapsed: {timestamp3 - timestamp2}")
train_generator = custom_generator.flow_from_dataframe_with_progress(processed_df, x_col='BraTS21ID', y_col='MGMT_value', target_size=image_size, batch_size=batch_size)
end_time = time.time()
print(f"Finished! Total time elapsed: {end_time - start_time}")
gc.collect()

# ImageDataGenerator for testing
test_datagen = ImageDataGenerator(preprocessing_function=lambda x: load_dicom_image_and_preprocess_for_tf(x))
test_generator = test_datagen.flow_from_directory(test_dir, target_size=image_size, batch_size=batch_size, class_mode='binary')


model = tf.keras.models.Sequential([
#setup for later - here will be the model
])

#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train your model using the TensorFlow Dataset
#history = model.fit(train_generator, epochs=10, validation_data=test_generator)

# Save your model if needed
#model.save('initial_model.h5')
