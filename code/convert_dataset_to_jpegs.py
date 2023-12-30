from PIL import Image
import pydicom
import os

resources_path = "../resources/dataset/train/"
output_path =  "../resources/dataset/jpg/train/"

def convert_dicom_to_jpeg(dicom_path, output_path):
    dicom_data = pydicom.dcmread(dicom_path)
    image_array = dicom_data.pixel_array
    image = Image.fromarray(image_array)
    image.save(output_path, format='JPEG')


def convert_all():
    for dir in os.listdir(resources_path):
        for modality in ['/FLAIR', '/T1w', '/T1wCE', '/T2w']:
            modality_path = resources_path + dir + modality
            for filename in os.listdir(modality_path):
                if filename.endswith('.dcm'):
                    file_path = os.path.join(modality_path, filename)
                    save_path = output_path + dir + modality
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    convert_dicom_to_jpeg(file_path, save_path + "/" + filename)

convert_all()