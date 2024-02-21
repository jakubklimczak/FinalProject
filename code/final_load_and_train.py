import pandas as pd
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # to prevent walls of annoying errors at launch which do not impact the program


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, concatenate, Input, UpSampling2D
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import CSVLogger
from datetime import datetime
import matplotlib.pyplot as plt

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


csv_path = '../resources/dataset/expanded_rows.csv'
dataset_path = '../resources/dataset/png_train/'
root_path = '../resources/dataset/'
resources_path = '../resources/'

df = pd.read_csv(csv_path, dtype={'BraTS21ID': str, 'MGMT_value': str}) # converting the values to strings to avoid problems such as losing 0s etc

batch_size = 16
size1dim = 256
image_size = (size1dim, size1dim)
num_classes = 2 # either 1 or 0


def get_model(input_size=(size1dim, size1dim, 1)):
    inputs = Input(input_size)
    # Contracting Path
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # Bottom of U-Net
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    # Expansive Path
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)
    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)
    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)
    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
    # Output layer
    output = Conv2D(1, 1, activation='sigmoid')(conv9)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.summary()
    return model



datagen = ImageDataGenerator(rescale=1./255) # rescale normalises the pixel values to be in range [0-1]

'''
# I use cross validation to reduce the random factors of validation set when training 
num_folds = 5 # the number of folds for cross-validation - so dataset is divided into n patrs, n different times the model will be trained and evaluated, each time 1 part will be used as validation set
kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=2137) # shuffle set to true helps distribute the data evenly to try to avoid any bias. I use random_state number for reproducibility - if I don't change the seed, I should always get the same division
'''


'''
# Iterating through all of the folds
for fold, (train_indices, val_indices) in enumerate(kfold.split(df['BraTS21ID'], df['MGMT_value'])):
    train_data = df.iloc[train_indices]
    val_data = df.iloc[val_indices]

    train_generator = datagen.flow_from_dataframe(
        train_data,
        directory=None,
        x_col='full_path',
        y_col='MGMT_value',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',  # binary classification, as MGMT_values are either 1 or 0
        validate_filenames=False
    )

    val_generator = datagen.flow_from_dataframe(
        val_data,
        directory=None,
        x_col='full_path',
        y_col='MGMT_value',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',  # binary classification, as MGMT_values are either 1 or 0
        validate_filenames=False
    )

    # building a model
    model = get_model()

    model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])

    log_file_path = root_path + f'training_log_fold_{fold + 1}.csv'
    csv_logger = CSVLogger(log_file_path)

    num_epochs = 10
    history = model.fit(train_generator, epochs=num_epochs, validation_data=val_generator, callbacks=[csv_logger])

    validation_loss, validation_accuracy = model.evaluate(val_generator)
    print(f'Fold {fold + 1} - Validation Accuracy: {validation_accuracy}')
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    with open(root_path+'fold_output.txt', 'a') as file:
        file.write(f'{formatted_datetime} - Fold {fold + 1} - Validation Accuracy: {validation_accuracy}\n')

    model_save_path = resources_path + f"model_fold_{fold + 1}.h5"
    model.save(model_save_path)
    print(f"Model for fold {fold + 1} saved to {model_save_path}")

    history_save_path = resources_path + f"history_fold_{fold + 1}.txt"
    with open(history_save_path, 'w') as file:
        file.write(str(history.history))
    print(f"Training history for fold {fold + 1} saved to {history_save_path}")

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

'''