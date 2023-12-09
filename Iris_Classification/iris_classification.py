import pandas as p
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as pl
import seaborn as sb

# Load the Iris dataset from a CSV file
csv_file_path = 'C:\\Users\\samar\\OneDrive\\Documents\\GitHub\\OIBSIP\\Iris_Classification\\Iris.csv'
Iris_data = p.read_csv(csv_file_path)

# Shuffling of sample data
df = p.DataFrame(Iris_data)
df = df.sample(frac = 1)

# Display the first few rows of the dataset
print("Dataset preview:")
print(Iris_data.head())

# Split the dataset into features (X) and target variable (y)
X = Iris_data.drop('Species', axis=1)
y = Iris_data['Species']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Support Vector Machine (SVM) model
model = SVC(kernel='linear', C=1)
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
classification_rep = classification_report(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)

# Print the evaluation results
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_rep)
print("Confusion Matrix:\n", conf_matrix)

# Visualize the confusion matrix
pl.figure(figsize=(8, 6))
sb.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
pl.title('Confusion Matrix')
pl.xlabel('Predicted')
pl.ylabel('Actual')
pl.show()

# Example: Make predictions for a new data point
new_data_point = [[151, 4.8, 3.2, 1.1, 0.2]]
new_prediction = model.predict(new_data_point)
print(f"Prediction for new data point {new_data_point}: {new_prediction[0]}")