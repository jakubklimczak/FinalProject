# Final Project - Jakub Klimczak
**Silesian University of Technology**

**Analysis of brain MRI using deep learning**

- The final project for engineering degree - **Informatics**

- Prepared to work on Ubuntu **22.04.3** on WSL2, not guaranteed to work in a different environment and I discourage you to try.

- Used dataset:
U.Baid, et al., “The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification”, arXiv:2107.02314, 2021.

- Setup (if you want to train the model yourself): 
1. Download the dataset using ./download_kaggle_dataset.sh in resources folder (run it as sudo). WARNING: it's big
2. Install python and libraries from requirements.txt in the main project folder: pip install -r requirements.txt
3. Install CUDA, Cuda Toolkit, cuDNN, TensorRT. They are necessary for Tensorflow to run on GPU. If you wish to run it on CPU, you can omit this step, however some further setup might be required on your part, the requirements.txt libraries are all with GPU in mind. For help refer to Nvidia's manual pages, it's all well documented. To select correct versions, refer to https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html