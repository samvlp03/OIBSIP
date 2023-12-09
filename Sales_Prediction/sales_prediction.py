import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

file_path="C:\\Users\\samar\\OneDrive\\Documents\\GitHub\\OIBSIP\\Sales_Prediction\\Sales_Prediction.csv"

# Load your advertising sales data 
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
print(data.head())

# Identify features (X) and target variable (y)
X = data[['TV', 'Radio', 'Newspaper']]  
y = data['Sales']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Visualize actual vs predicted sales
plt.scatter(X_test['TV'], y_test, label='Actual Sales', alpha=0.5)
plt.scatter(X_test['TV'], y_pred, label='Predicted Sales', alpha=0.5)
plt.xlabel('Ad Budget')
plt.ylabel('Sales')
plt.title('Actual vs Predicted Sales')
plt.legend()
plt.show()
