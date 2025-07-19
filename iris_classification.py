# iris_classifier_v2.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
file_path = "C:/Users/shash/OneDrive/Desktop/Internship 2/task 1/Iris.csv"
iris_data = pd.read_csv(file_path)

# Prepare features (X) and labels (y)
features = iris_data.drop(["Id", "Species"], axis=1)
labels = iris_data["Species"]

# Encode species labels
encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features, labels_encoded, test_size=0.3, random_state=42
)

# Initialize and train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)

# Evaluate model
model_accuracy = accuracy_score(y_test, predictions)
print(f"\n✅ Model Accuracy: {model_accuracy * 100:.2f}%")
print("\n📄 Classification Report:")
print(classification_report(y_test, predictions, target_names=encoder.classes_))

# Visualize data with seaborn pairplot
sns.pairplot(iris_data, hue="Species")
plt.suptitle("🌸 Iris Dataset Visualization", y=1.02)
plt.show()
