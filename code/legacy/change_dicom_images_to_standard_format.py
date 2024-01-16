import os
import shutil
import pydicom
from PIL import Image

def copy_and_convert_dicom_files(source_dir, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    #walk through the source directory and its subdirectories
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if is_dicom_file(file_path):
                relative_path = os.path.relpath(file_path, source_dir)
                destination_path = os.path.join(destination_dir, relative_path)
                destination_folder = os.path.dirname(destination_path)
                if not os.path.exists(destination_folder):
                    os.makedirs(destination_folder)
                dicom_data = pydicom.dcmread(file_path)
                #convert DICOM pixel data -> PIL Image
                image = Image.fromarray(dicom_data.pixel_array)
                #save as png
                image.save(destination_path.replace(".dcm", ".png"))
                print(f"Converted and saved: {file_path} to {destination_path.replace('.dcm', '.png')}")

def is_dicom_file(file_path):
    try:
        pydicom.dcmread(file_path)
        return True
    except pydicom.errors.DicomError:
        return False

source_directory = "../resources/dataset/train"
destination_directory = "../resources/dataset/png_train"
copy_and_convert_dicom_files(source_directory, destination_directory)
