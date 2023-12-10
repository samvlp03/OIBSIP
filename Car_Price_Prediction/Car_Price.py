# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

fp='C:\\Users\\samar\\OneDrive\\Documents\\GitHub\\OIBSIP\\Car_Price_Prediction\\Car_data.csv'
# Load your dataset 
data = pd.read_csv('Car_data.csv')

# Display the first few rows of the dataset to understand its structure
print(data.head())

# Identify features (X) and target variable (y)
X = data.drop('Selling_Price', axis=1)  
y = data['Present_Price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a preprocessing pipeline (you can customize this based on your dataset)
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
numeric_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

# Create a pipeline with the preprocessing and the model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor(random_state=42))])

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')


# Scatter plot for actual vs predicted prices
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual Prices vs Predicted Prices')
plt.show()

# Line plot for actual prices and predicted prices
plt.plot(y_test.values, label='Actual Prices', linestyle='-', marker='o')
plt.plot(y_pred, label='Predicted Prices', linestyle='-', marker='o')
plt.xlabel('Index')
plt.ylabel('Price')
plt.title('Actual Prices vs Predicted Prices')
plt.legend()
plt.show()

