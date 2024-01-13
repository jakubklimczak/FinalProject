import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skimage import exposure
import tensorflow as tf
from tensorflow.keras import layers, models
import pydicom
import gc
import psutil

# Global variables
images_processed = 0

# Load your CSV file
csv_path = '../resources/dataset/train_labels.csv'
df = pd.read_csv(csv_path)
training_directory = '../resources/dataset/train'

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

#DEBUG
def print_memory_usage():
    # Get the current memory usage
    memory_info = psutil.virtual_memory()

    # Print the memory information
    print(f"Total: {memory_info.total} bytes")
    print(f"Available: {memory_info.available} bytes")
    print(f"Used: {memory_info.used} bytes")
    print(f"Percentage Used: {memory_info.percent}%")
    print("")
#END DEBUG

def close_all_files():
    # Get the list of process IDs associated with the current Python process
    process = psutil.Process()
    current_pid = process.pid

    # Iterate over open files and close them
    for fd in process.open_files():
        try:
            # Check if the file is associated with the current Python process
            if fd.pid == current_pid:
                # Close the file
                fd.fileobj.close()
        except Exception as e:
            print(f"Error closing file: {e}")

def load_dicom_images(folder_name):
    folder_path = os.path.join(training_directory, str(folder_name).zfill(5))
    global images_processed
    images_processed += 1
    
    try: 
        # Check if the folder exists
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist.")
            return None

        images = []
        for subdir, _, files in os.walk(folder_path):
            for filename in files:
                if filename.endswith('.dcm'):
                    dicom_path = os.path.join(subdir, filename)
                    dicom_data = pydicom.dcmread(dicom_path)
                    pixel_array = dicom_data.pixel_array

                    # Pad or truncate to target size
                    uin8_image_array = exposure.rescale_intensity(pixel_array, in_range='uint16', out_range='uint8')

                    images.append(uin8_image_array)

        return np.array(images)
    except Exception as e:
        print(f"Error processing file {filename}: {e}")
        return None
    finally:
        close_all_files()
        if images_processed%30==0:
            print_memory_usage()
            print("Garbage collect")
            gc.collect()
            images_processed=0

# Load training data
train_data = []
for index, row in train_df.iterrows():
    folder_name = row['BraTS21ID']
    label = row['MGMT_value']
    images = load_dicom_images(folder_name)
    if images is not None:
        train_data.append((images, label))

# Load testing data
test_data = []
for index, row in test_df.iterrows():
    folder_name = row['BraTS21ID']
    label = row['MGMT_value']
    images = load_dicom_images(folder_name)
    if images is not None:
        test_data.append((images, label))
        

# Convert data to NumPy arrays
X_train, y_train = zip(*train_data)
X_test, y_test = zip(*test_data)

X_train, X_test = np.array(X_train), np.array(X_test)
y_train, y_test = np.array(y_train), np.array(y_test)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')
