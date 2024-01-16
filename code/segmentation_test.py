import numpy as np 
import os
import copy
from math import *
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from functools import reduce
from glob import glob

# reading in dicom files
import pydicom

# skimage image processing packages
from skimage import measure, morphology
from skimage.morphology import ball, binary_closing
from skimage.measure import label, regionprops

# scipy linear algebra functions 
from scipy.linalg import norm
from scipy import ndimage

# ipywidgets for some interactive plots
from ipywidgets.widgets import * 
import ipywidgets as widgets

def load_scan(paths):
    slices = [pydicom.read_file(path) for path in paths]
    #slices.sort(key = lambda x: int(x.InstanceNumber), reverse = True)
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

def get_pixels(scans):
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

   
def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    # Remove background label if present
    if bg in vals:
        counts = counts[vals != bg]
        vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_brain_mask(image, fill_brain_structures=True):
    # Adjust intensity threshold for brain
    threshold = np.mean(image) * 0.1
    print(threshold)

    binary_image = np.array(image >= threshold, dtype=np.int8) + 1
    print(binary_image[binary_image>2])

    labels, num_features = ndimage.label(binary_image)
    print(binary_image[binary_image>2])

    background_label = labels[0, 0, 0]
    binary_image[background_label == labels] = 2

    if fill_brain_structures:
        for i in range(1, num_features + 1):
            component = (labels == i)
            binary_image[component] = 1

    binary_image -= 1
    binary_image = 1 - binary_image

    # Perform morphological operations with adjusted structuring element sizes
    binary_image = morphology.binary_erosion(binary_image, np.ones((5, 5, 5)))
    binary_image = morphology.binary_dilation(binary_image, np.ones((10, 10, 10)))

    # Remove other structures inside the brain
    labels, num_features = ndimage.label(binary_image)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:
        binary_image[labels != l_max] = 0

    return binary_image



patient_id = '00000'
patient_folder = f'../resources/dataset/train/{patient_id}/'
data_paths = glob(patient_folder + '/*/*.dcm')

# Print out the first 5 file names to verify we're in the right folder.
#print (f'Total of {len(data_paths)} DICOM images.\nFirst 5 filenames:' )
#print(data_paths[:5])



# set path and load files 
patient_dicom = load_scan(data_paths)
patient_pixels = get_pixels(patient_dicom)
#sanity check
plt.imshow(patient_pixels[80], cmap=plt.cm.bone)
plt.savefig('output_plot.png')  


segmented_brain = segment_brain_mask(patient_pixels, fill_brain_structures=False)
segmented_brain_fill = segment_brain_mask(patient_pixels, fill_brain_structures=True)
internal_structures = np.logical_xor(segmented_brain_fill, segmented_brain)


copied_pixels = copy.deepcopy(patient_pixels)
for i, mask in enumerate(segmented_brain_fill): 
    get_high_vals = mask == 0
    copied_pixels[i][get_high_vals] = 0
seg_lung_pixels = copied_pixels
# sanity check
f, ax = plt.subplots(1,2, figsize=(10,6))
ax[0].imshow(patient_pixels[80], cmap=plt.cm.bone)
ax[0].axis(False)
ax[0].set_title('Original')
ax[1].imshow(seg_lung_pixels[80], cmap=plt.cm.bone)
ax[1].axis(False)
ax[1].set_title('Segmented')
plt.savefig('output2.png')