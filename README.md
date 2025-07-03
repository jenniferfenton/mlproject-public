# CS4641
## COPY OF ORIGINAL
Group Project for ML
[Website Link](https://jenniferfenton.github.io/mlproject-public/)

./Proposal.md: Our original proposal for this project

./index.md: The main page of our project, currently contains the midpoint report.

./Model_Implementation_Files/: Contains our training data and jupyter notebooks used to train models on that data.

./Model_Implementation_Files/ExtractedFeatures/: Folder containing preprocessed features for subjects

./Model_Implementation_Files/ExtractedFeatures/EverySubjectWholeSignal_features.tsv: The compiled feature set we used to train models in the midpoint report.

./Model_Implementation_Files/eeg/: A placeholder folder. This exists in the google drive where we actually processed the data. In this folder is a link to the raw data

./Model_Implementation_Files/participants/: Contains a tsv file describing participant age, gender, and cognitive health. Used to label data

./Model_Implementation_Files/kNN.ipynb: Contains code for extracting features from all subjects, standardizing those features, and using them to train kNN predictors.

./Model_Implementation_Files/Feature Selection.ipynb: Used to select features from the standardized feature set using forward feature selection

./Model_Implementation_Files/SVM.ipynb: Contains code for training support vector machine on standardized feature set. 

./Model_Implementation_Files/multilayer perceptron.ipynb: Contains code for training multi-layer perceptrons on standardized feature set. 
