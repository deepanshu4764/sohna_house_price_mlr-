import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv("sohna_house_prices.csv")

# Define Features (X) and Target Variable (Y)
X = df[['Area', 'Bedrooms', 'Bathrooms', 'Location Score', 'Age of House']]
y = df['Price']

# Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Multiple Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model as a .pkl file
joblib.dump(model, "sohna_price_model.pkl")

print("✅ Model trained and saved as 'sohna_price_model.pkl'")
