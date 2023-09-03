import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report

# Step 1: Load and preprocess the data
train_data = pd.read_csv('/Users/swastikanupam/Desktop/vscode/Amity Project/test.csv')  # Replace with the actual path to your training dataset

# Preprocess the training data
X_train = train_data['tweet']  # Input text (tweets)
y_train = train_data['labels']  # Labels

# Convert labels to list of lists format
train_labels = [label.split(', ') for label in y_train]

# Transform labels to binary representation
label_encoder = MultiLabelBinarizer()
y_train_encoded = label_encoder.fit_transform(train_labels)

# Step 2: Feature Engineering
vectorizer = TfidfVectorizer(max_features=5000)  # Adjust parameters as needed
X_train_transformed = vectorizer.fit_transform(X_train).toarray()

# Step 3: Create the TensorFlow model
input_dim = X_train_transformed.shape[1]  # Input dimension (number of features)
output_dim = y_train_encoded.shape[1]  # Output dimension (number of labels)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_dim=input_dim),
    tf.keras.layers.Dropout(0.7),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.7),
    tf.keras.layers.Dense(output_dim, activation='sigmoid')
])

# Step 4: Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 5: Train the model and save the training history within the model
history = model.fit(X_train_transformed, y_train_encoded, batch_size=64, epochs=100, validation_split=0.2)

# Save the training history within the model
model.history = history.history

# Step 6: Predict on new test data
test_data = pd.read_csv('test_dataset.csv')  # Replace with the actual path to your test dataset

# Preprocess the test data
X_test = test_data['tweet']  # Input text (tweets)
y_test = test_data['labels']  # Labels

# Convert labels to list of lists format
test_labels = [label.split(', ') for label in y_test]

# Transform labels to binary representation
y_test_encoded = label_encoder.transform(test_labels)

# Feature Engineering on the test data
X_test_transformed = vectorizer.transform(X_test).toarray()

# Perform prediction on the test data
y_pred = model.predict(X_test_transformed)
y_pred_binary = (y_pred >= 0.5).astype(int)

# Step 7: Evaluate the model on the test dataset
macro_f1 = classification_report(y_test_encoded, y_pred_binary, target_names=label_encoder.classes_, output_dict=False)

# Display the macro F1 score and classification report
print("Macro F1 Score:", macro_f1)
print("\nClassification Report:")
print(classification_report(y_test_encoded, y_pred_binary, target_names=label_encoder.classes_))
