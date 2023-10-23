import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

# Step 1: Load and preprocess the data
data = pd.read_csv('/Users/swastikanupam/Desktop/vscode/Amity Project/test.csv')  
# Preprocess the data
X = data['tweet']  # Input text (tweets)
y = data['labels']  # Labels

# Convert labels to list of lists format
labels = [label.split(', ') for label in y]

# Transform labels to binary representation
label_encoder = MultiLabelBinarizer()
y_encoded = label_encoder.fit_transform(labels)

# Step 2: Feature Engineering using tdf vectorizer
vectorizer = TfidfVectorizer(max_features=5000)  # Adjust parameters as needed
X_transformed = vectorizer.fit_transform(X).toarray()

# Step 3: Create the TensorFlow model
input_dim = X_transformed.shape[1]  # Input dimension (number of features)
output_dim = y_encoded.shape[1]  # Output dimension (number of labels)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_dim=input_dim),
    tf.keras.layers.Dropout(0.7),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.7),
    tf.keras.layers.Dense(output_dim, activation='sigmoid')
])

# Step 4: Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 5: Train the model
batch_size = 64  
epochs = 100  
max_iterations = 1000  

iteration = 0
while iteration < max_iterations:
    model.fit(X_transformed, y_encoded, batch_size=batch_size, epochs=epochs, validation_split=0.2)
    iteration += 1
