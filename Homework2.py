import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# Load the dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Set up parameter grid for grid search
param_grid = [
    {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]},
    {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100], 'gamma': [0.1, 0.01, 0.001]}
]

# Create SVR model
svr = SVR()

# Initialize GridSearchCV
grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)

# Fit the model with grid search
grid_search.fit(X_train, y_train)

# Get the best SVR predictor
best_svr = grid_search.best_estimator_
y_pred = best_svr.predict(X_test)

# Calculate the performance of the best SVR predictor
mse = mean_squared_error(y_test, y_pred)
print(f"Best SVR predictor performance (MSE): {mse}")
print(f"Best hyperparameters: {grid_search.best_params_}")
from sklearn.model_selection import RandomizedSearchCV

# Set up parameter distributions for randomized search
param_dist = {
    'kernel': ['linear', 'rbf'],
    'C': np.logspace(-3, 2, 6),
    'gamma': np.logspace(-3, 2, 6)
}
svr = SVR()
random_search = RandomizedSearchCV(svr, param_distributions=param_dist, n_iter=10, cv=5,
                                   scoring='neg_mean_squared_error', verbose=1)
random_search.fit(X_train, y_train)
best_svr_random = random_search.best_estimator_
y_pred_random = best_svr_random.predict(X_test)
# Calculate the performance of the best SVR predictor from randomized search
mse_random = mean_squared_error(y_test, y_pred_random)
print(f"Best SVR predictor performance with RandomizedSearchCV (MSE): {mse_random}")
print(f"Best hyperparameters from RandomizedSearchCV: {random_search.best_params_}")
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
# Create the preparation pipeline
preparation_pipeline = Pipeline([
    ('feature_selection', SelectKBest(k=5)),
    # Add other necessary preprocessing steps here (e.g., scaling, encoding)
    ('svr', SVR(kernel='rbf', C=10, gamma=0.01)) 
])


preparation_pipeline.fit(X_train, y_train)

# Make predictions
y_pred_pipeline = preparation_pipeline.predict(X_test)


mse_pipeline = mean_squared_error(y_test, y_pred_pipeline)
print(f"Pipeline performance with feature selection (MSE): {mse_pipeline}")
# Create the full pipeline including data preparation and SVR
full_pipeline = Pipeline([
    ('feature_selection', SelectKBest(k=5)),  # to set the desired k value here
    ('svr', SVR(kernel='rbf', C=10, gamma=0.01)) 
])
# to Fit the full pipeline
full_pipeline.fit(X_train, y_train)

# Make predictions
y_pred_full_pipeline = full_pipeline.predict(X_test)

# Calculate the performance of the full pipeline
mse_full_pipeline = mean_squared_error(y_test, y_pred_full_pipeline)
print(f"Full pipeline performance (MSE): {mse_full_pipeline}")
from sklearn.preprocessing import StandardScaler

# Create a combined pipeline with data preparation and SVR
combined_pipeline = Pipeline([
    ('feature_selection', SelectKBest()),  # We'll let GridSearchCV explore k
    ('scaler', StandardScaler()),  # Let's include scaling as well
    ('svr', SVR())
])
# Set up parameter grid for grid search
param_grid_combined = {
    'feature_selection__k': [3, 5, 7],  # Setr desired range for k 
    'scaler': [None, StandardScaler()],
    'svr__kernel': ['linear', 'rbf'],
    'svr__C': np.logspace(-3, 2, 6),
    'svr__gamma': np.logspace(-3, 2, 6)
}

# Initialize GridSearchCV
grid_search_combined = GridSearchCV(combined_pipeline, param_grid_combined, cv=5,
                                    scoring='neg_mean_squared_error', verbose=1)

# Fit the model with grid search
grid_search_combined.fit(X_train, y_train)

# Get the best pipeline
best_pipeline = grid_search_combined.best_estimator_

# Make predictions
y_pred_combined = best_pipeline.predict(X_test)

# Calculate the performance of the best pipeline
mse_combined = mean_squared_error(y_test, y_pred_combined)
print(f"Best pipeline performance (MSE): {mse_combined}")
print(f"Best hyperparameters from combined pipeline: {grid_search_combined.best_params_}")
