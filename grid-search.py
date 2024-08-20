import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

!pip install tensorflow==2.12.0
!pip install scikit-learn

"""#**Preparing** dataset

y = 3 * x^2 + 5 * x + 1
"""

np.random.seed(0)
x_train = np.random.rand(1000, 1)  # Example feature data
y_train = 3 * x_train**2 + 5 * x_train + 1  # Target data based on y = 3*x^2 + 5*x + 1

# Normalize data (optional but recommended)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

param_grid = {
    'hidden_units': [16, 32, 64],
    'optimizer': ['adam', 'sgd']
}

"""#Model definition"""

def create_model(hidden_units=64, optimizer='adam'):
  model = keras.Sequential()
  model.add(keras.layers.Dense(units=hidden_units, activation='relu', input_shape= (1,)))
  model.add(keras.layers.Dense(units=hidden_units, activation='relu'))
  model.add(keras.layers.Dense(units=hidden_units, activation='relu'))
  model.add(keras.layers.Dense(units=1))
  model.compile(optimizer=optimizer, loss='mean_squared_error')
  #model.summary()
  return model;

def create_keras_regressor(hidden_units=64, optimizer='adam'):
  return KerasRegressor(build_fn=create_model, hidden_units=hidden_units, optimizer=optimizer, epochs=100, batch_size=32, verbose=0)

grid = GridSearchCV(estimator=create_keras_regressor(),
                    param_grid=param_grid,
                    scoring='neg_mean_squared_error',  # Use negative mean squared error for regression
                    cv=3)  # Number of cross-validation folds

# Fit grid search
grid_result = grid.fit(x_train, y_train)

grid_result.best_score_

# Generate synthetic test data for demonstration
x_test = np.random.rand(200, 1)
y_test = 3 * x_test**2 + 5 * x_test + 1
x_test = scaler.transform(x_test)

# Get the best model
best_model = grid_result.best_estimator_

# Evaluate on test data
test_pred = best_model.predict(x_test[50])
test_pred, y_test[50]
