# Final Project - Jakub Klimczak
**Silesian University of Technology**

**Analysis of brain MRI using deep learning**

- The final project for engineering degree - **Informatics**

- Prepared to work on Ubuntu **22.04.3** on WSL2, not guaranteed to work in a different environment and I discourage you to try.

- Used dataset:
U.Baid, et al., “The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification”, arXiv:2107.02314, 2021.

- Citations:
1. U.Baid, et al., "The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification", arXiv:2107.02314, 2021(opens in a new window).

2. B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby, et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE Transactions on Medical Imaging 34(10), 1993-2024 (2015) DOI: 10.1109/TMI.2014.2377694 (opens in a new window)

3. S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J.S. Kirby, et al., "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features", Nature Scientific Data, 4:170117 (2017) DOI: 10.1038/sdata.2017.117

- Setup (if you want to train the model yourself): 
1. Download the dataset using ./download_kaggle_dataset.sh in resources folder (run it as sudo). WARNING: it's big

2. Install python and libraries from requirements.txt in the main project folder: pip install -r requirements.txt

3. Install CUDA, Cuda Toolkit, cuDNN, TensorRT. They are necessary for Tensorflow to run on GPU. If you wish to run it on CPU, you can omit this step, however some further setup might be required on your part, the requirements.txt libraries are all with GPU in mind. For help refer to Nvidia's manual pages, it's all well documented. To select correct versions, refer to https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html