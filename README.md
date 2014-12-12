This project is an implementation of the following paper:

http://cs229.stanford.edu/proj2013/ShiraniMehr-SMSSpamDetectionUsingMachineLearningApproach.pdf

This paper is present in references/ folder.

It is written in Python.

Prerequisites for running this project are:

Python-2.x(used 2.7.6)
Numpy-1.8.x(used 1.8.2)
scikit-learn-0.15.x(used 0.15.2)


Dataset:

Dataset used is the SMS-Spam dataset from UCI Machine Learning repository.
This dataset contains 5574 sms, with label of ham/spam as prefix for each message. 100% of this data is present in
dataset/training-data-full.

70% of the dataset is taken as training data and is present in dataset/train/train-data-1
30% of the dataset is taken as test data to measure accuracy. It is present in dataset/test/test-data-1

The actual test data has been collected into dataset/test/sms-data

bin_data/ folder is empty. After each stage of processing, BOWs, Feature Vectors, classifiers etc., are stored in binary
format with numpy in this folder, to be used later.

references/ folder contains some reference papers which are source/relevant to this project.

Project details / How to run:

Major code is in 4 classes: Preprocessor.py, FeatureModel.py, Trainer.py, Tester.py

Preprocessor processes training/test data and builds BOWs
FeatureModel reads BOWs, extracts features for Training phase, and computes feature vectors from training/test data.
Trainer reads features, features vectors and trains the classifier.
Tester reads the classifier, predicts labels and computes accuracy scores from test data.

XMLMessageExtractor.py is a utility program to extract the TextMessages from smsCorpus_en.xml data file.
It stores the data into the file at dataset/test/sms-data
For size concerns, this project doesn't contain the smsCorpus_en xml file, but the processed messages file is present in the sms-data file.

Remaining python scripts are named according to their tasks:

process_training_data.py --> Processes Training data and builds BOWs
process_testing_data.py --> Processes Testing data and builds BOWs
extract_features.py --> Extracts Features from Training data BOWs
vectorize_training_data.py --> Computes Feature Vectors for Training data
vectorize_testing_data.py --> Computes Feature Vectors for Testing data
train_classifier.py --> Trains a MultinomialNB classifier
test_classifier.py --> Computes the accuracy of the trained classifier
label_custom_data.py --> Processes actual sms-data, computes BOWs and FVs, predicts the labels

There are 2 shell script files, they run the above python scripts in some sequences:

normal-flow.sh
custom-flow.sh

They can be run as sh normal-flow.sh, sh custom-flow.sh
The sh commands must be run from this project folder's base/root level

normal-flow.sh processes, trains and computes accuracy from the 70-30 split data.
custom-flow.sh processes and trains from 100% data, and then predicts the labels for the sms-data
file built from smsCorpus_en.xml. The results(label-suffixed-messages) are written to output/results.txt


Project implementation details are present in report/report.pdf








