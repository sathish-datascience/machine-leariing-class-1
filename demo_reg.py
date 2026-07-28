from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Feature / Input data (Area in sq ft)
X = [[1000], [1200], [1400], [1600], [1800], [2000]]

# Target (House price in lakhs)
y = [20, 25, 30, 35, 40, 45]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)


# Predict new house price
new_house = [[2200]]

result = model.predict(new_house)

print("Predicted price for 2200 sq ft:", result)