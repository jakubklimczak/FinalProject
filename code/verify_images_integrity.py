import pydicom
from pydicom import FileDataset
import os

def find_files(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

def check_dicom_file_integrity(file_path):
    try:
        dicom_data = pydicom.dcmread(file_path)
        dicom_data.is_valid()  # Checks for general file integrity
        dicom_data.file_meta.is_valid()  # Checks for file meta information integrity
        dicom_data.dataset.is_consistent()  # Checks for dataset consistency
        
        pixel_array = dicom_data.pixel_array # Checks if the image isn't blank (frequent occurance)
        if not any(pixel_array.flatten() > 0):
            raise ValueError("Image is entirely black.")
        #print(f"File '{file_path}' appears to be valid.")
    except Exception as e:
        print(f"File '{file_path}' failed integrity check: {e}")

file_path = '../resources/dataset/train/00059/T1wCE/Image-51.dcm'

all_files = find_files(file_path)

for path in all_files:
    check_dicom_file_integrity(path)
