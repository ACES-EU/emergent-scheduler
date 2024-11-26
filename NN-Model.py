import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Step 1: Load the data from the CSV file
df = pd.read_csv('normalized_peer_selection_training_data.csv')

# Step 2: Split the data into features (X) and labels (y)
X = df[['new_pod_demand_cpu', 'new_pod_demand_mem', 'peer_pod_slack_cpu', 'peer_pod_slack_mem', 
        'new_pod_demand_steps', 'peer_pod_remain_steps']]  # Features
y = df[['f1', 'f2']]  # Labels

# Step 3: Split the data into 75% training and 25% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 4: Normalize the data (if necessary)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform training data
X_test_scaled = scaler.transform(X_test)  # Transform test data

# Step 5: Create and train the neural network model
# Using MLPRegressor (Multi-layer Perceptron for regression)
model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
model.fit(X_train_scaled, y_train)

# Step 6: Make predictions on the test data
y_pred = model.predict(X_test_scaled)

# To save trained model and dcalar, to be used for prediction later
joblib.dump(model, 'trained_neural_net_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)  # Mean squared error as a performance metric
print(f'Mean Squared Error: {mse}')
