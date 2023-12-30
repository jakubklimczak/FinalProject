# this script shows a photo and helps me verify if it is valid - non-black
import matplotlib.pyplot as plt 
import pydicom 
import pydicom.data 

base = '../resources/dataset/train/00014/FLAIR'
pass_dicom = 'Image-146.dcm'

filename = pydicom.data.data_manager.get_files(base, pass_dicom)[0] 

ds = pydicom.dcmread(filename) 

# if numbers are printed - the image is valid, even if not visible
for x in range(512):
    for y in range(512):
        if ds.pixel_array[x, y] != 0:
            print(ds.pixel_array[x, y])

print(type(ds.pixel_array))

plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
plt.show() 
