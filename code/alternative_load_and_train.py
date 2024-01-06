import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import pydicom
import gc

# Global variables
images_processed = 0

# Load your CSV file
csv_path = '/home/jaklimczak/envs/FinalProject/resources/dataset/train_labels.csv'
df = pd.read_csv(csv_path)
training_directory = '/home/jaklimczak/envs/FinalProject/resources/dataset/train'

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

def load_dicom_images(folder_name, target_size=(512, 512)):
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
                    if pixel_array.shape[0] < target_size[0]:
                        pad_size = target_size[0] - pixel_array.shape[0]
                        pixel_array = np.pad(pixel_array, ((0, pad_size), (0, 0)), mode='constant')
                    elif pixel_array.shape[0] > target_size[0]:
                        pixel_array = pixel_array[:target_size[0], :]

                    if pixel_array.shape[1] < target_size[1]:
                        pad_size = target_size[1] - pixel_array.shape[1]
                        pixel_array = np.pad(pixel_array, ((0, 0), (0, pad_size)), mode='constant')
                    elif pixel_array.shape[1] > target_size[1]:
                        pixel_array = pixel_array[:, :target_size[1]]

                    images.append(pixel_array)

        return np.array(images)
    except Exception as e:
        print(f"Error processing file {filename}: {e}")
        return None
    finally:
        dicom_data.close()
        if images_processed%30==0:
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
