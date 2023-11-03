import os
import requests

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as mape
from itertools import combinations

# checking ../Data directory presence
if not os.path.exists('./data'):
    os.mkdir('./data')

# download data if it is unavailable
if 'data.csv' not in os.listdir('./data'):
    url = "https://www.dropbox.com/s/3cml50uv7zm46ly/data.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('./data/data.csv', 'wb').write(r.content)

# read data
data = pd.read_csv('./data/data.csv')

# Make X a dataframe with a predictor 'rating' and y a series with a target 'salary'
X = data.drop(columns=['salary'])
y = data[['salary']]

# Create a correlationmatrix to see which parameters are correlated with each other
corr_matrix = X.corr()

# There are 3 parameters that are correlated: age, experience, rating
high_corr = ['age', 'experience', 'rating']

# Create a list of the possible combinations of high correlation
combination = list(combinations(high_corr, 1)) + list(combinations(high_corr, 2))

mape_score = []

# Loop through the combinations
for item in combination:

    # Drop the high correlation combinations iteratively
    if len(item) == 1:
        X_ = X.drop(columns=item[0])
    else:
        X_ = X.drop(columns=list(item))

    # Split predictor and target into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_, y, test_size=0.3, random_state=100)

    # Fit the linear regression model with the training data
    model = LinearRegression().fit(X_train, y_train)

    # Predict the salary for the test rating
    y_predict = model.predict(X_test)

    # Calculate the MAPE
    mape_score.append(mape(y_test, y_predict))


# The best MAPE was removing age and experience from the dataset
X_ = X.drop(columns=['age', 'experience'])

# Split predictor and target into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_, y, test_size=0.3, random_state=100)

# Fit the linear regression model with the training data
model = LinearRegression().fit(X_train, y_train)

# Predict the salary for the test rating
y_predict = model.predict(X_test)

# Replace the negative values with 0
y_predict[y_predict < 0] = 0

# Calculate the MAPE when negatives are replaced with zeros
mape_zero = mape(y_test, y_predict)

# Predict the salary for the test rating
y_predict = model.predict(X_test)

# Replace the negative values with the media of the training set
y_predict_median = y_predict
median_y = np.median(y_train)
y_predict[y_predict < 0] = median_y

# Calculate the MAPE when negatives are replaced with the median
mape_median = mape(y_test, y_predict)

# Print the smallest MAPE of the two sceanrios
print(min(mape_zero, mape_median).round(5))