According to the Kaggle page of the dataset:
-   train/ - folder containing the training files, with each top-level folder representing a subject.
    NOTE: There are some unexpected issues with the following three cases in the training dataset, participants can exclude the cases during training: [00109, 00123, 00709]. 
    We have checked and confirmed that the testing dataset is free from such issues.
-   train_labels.csv - file containing the target MGMT_value for each subject in the training data (e.g. the presence of MGMT promoter methylation)
-   test/ - the test files, which use the same structure as train 
    NOTE: the total size of the rerun test set (Public and Private) is ~5x the size of the Public test set