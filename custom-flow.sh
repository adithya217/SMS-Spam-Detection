
echo 'Processing training data...'
python process_training_data.py dataset/train/training-data-full

echo 'Now extracting features from training data...'
python extract_features.py

echo 'Now building feature vectors from training data...'
python vectorize_training_data.py

echo 'Now training classifier...'
python train_classifier.py

echo 'Now Processing test data...'
python label_custom_data.py
