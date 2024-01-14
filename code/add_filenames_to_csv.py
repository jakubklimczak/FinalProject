import pandas as pd
import os

# run this script only once, it will save the csv file and it won't be necessary

csv_path = '../resources/dataset/train_labels.csv'
dataset_path = '../resources/dataset/png_train/'
save_path = '../resources/dataset/'

# preparing the dataframe
df = pd.read_csv(csv_path, dtype={'BraTS21ID': str, 'MGMT_value': str}) # converting the values to strings to avoid problems such as losing 0s etc

expanded_rows = pd.DataFrame(columns=df.columns)

def find_png_files(directory_path):
    png_files = []
    for root, _, files in os.walk(directory_path):
       for file in files:
           if file.endswith('.png'):
               png_files.append(os.path.join(root, file))
    return png_files


for index, row in df.iterrows():
    print(row['BraTS21ID'])
    directory_path = dataset_path + row['BraTS21ID']
    row['BraTS21ID'] = dataset_path + row['BraTS21ID']
    image_files = find_png_files(directory_path)

    # create a new row for each image in the directory
    for image_file in image_files:
        new_row = row.copy()
        new_row['full_path'] = image_file
        expanded_rows = pd.concat([expanded_rows, new_row.to_frame().transpose()], ignore_index=True)




print(expanded_rows)
expanded_rows.to_csv(save_path + 'expanded_rows.csv', index=False)
