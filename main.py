import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Tokenize the URL
def tokenizer(url):
    tokens = re.split('[/-]', url)
    for i in tokens:
        if i.find(".") >= 0:
            dot_split = i.split('.')
            if "com" in dot_split:
                dot_split.remove("com")
            if "www" in dot_split:
                dot_split.remove("www")
            tokens += dot_split
    return tokens

# Load the CSV data into a DataFrame
url_df = pd.read_csv("mal1.csv")

# Split the data into features (X) and labels (y)
X = url_df['URLs']
y = url_df['Class']

# Split the data into training and testing sets
test_percentage = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percentage, random_state=42)

# Tokenize and vectorize the textual features
vectorizer = TfidfVectorizer(tokenizer=tokenizer)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Define Random Forest classifier
rf_classifier = RandomForestClassifier(n_jobs=-1)

# Train the Random Forest classifier
rf_classifier.fit(X_train_vec, y_train)

# Evaluate the trained model on the testing data
y_pred = rf_classifier.predict(X_test_vec)

# Report the accuracy
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)

# Generate confusion matrix and classification report
cmatrix = confusion_matrix(y_test, y_pred)
creport = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(creport)

print("\nConfusion Matrix:")
print(cmatrix)

# Generate confusion matrix heatmap
plt.figure(figsize=(5,5))
sns.heatmap(cmatrix, annot=True, fmt="d", linewidths=.5, square=True, cmap='Blues', annot_kws={"size": 16},
            xticklabels=['bad', 'good'], yticklabels=['bad', 'good'])
plt.xlabel('Actual Label', size=20)
plt.ylabel('Predicted Label', size=20)
plt.title('Confusion Matrix', size=20)
plt.xticks(rotation='horizontal', fontsize=16)
plt.yticks(rotation='horizontal', fontsize=16)
plt.show()

# Generate adversarial examples using FGSM
def fgsm(model, x, y, eps):
    with tf.GradientTape() as tape:
        tape.watch(x)
        prediction = model(x)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, prediction)
    gradient = tape.gradient(loss, x)
    signed_grad = tf.sign(gradient)
    x_adversarial = x + eps * signed_grad
    return x_adversarial

eps = 0.01  # perturbation magnitude
X_test_sparse = tf.convert_to_tensor(X_test_vec.toarray(), dtype=tf.float32)
y_test_sparse = y_test.to_numpy()

# Convert labels to one-hot encoding
num_classes = len(np.unique(y_test_sparse))
y_test_sparse_one_hot = tf.one_hot(y_test_sparse, num_classes, dtype=tf.float32)

# Generate adversarial examples
X_test_adversarial = fgsm(rf_classifier, X_test_sparse, y_test_sparse_one_hot, eps)

# Save the adversarial examples
np.savez('URL_adversarial.npz', X_test_adversarial)
print("Adversarial examples saved.")
