import numpy as np
import q3 as q3

#-------------------------------------------------------------------------------
# Auto Data
#-------------------------------------------------------------------------------

# load auto-mpg-regression.tsv, including  Keys are the column names, including mpg.
auto_data_all = []

# The choice of feature processing for each feature, mpg is always raw and
# does not need to be specified.  Other choices are q3.standard and q3.one_hot.

features1 = [('cylinders', q3.standard),
            ('displacement', q3.standard),
            ('horsepower', q3.standard),
            ('weight', q3.standard),
            ('acceleration', q3.standard),
            ('origin', q3.one_hot)]

features2 = [('cylinders', q3.one_hot),
            ('displacement', q3.standard),
            ('horsepower', q3.standard),
            ('weight', q3.standard),
            ('acceleration', q3.standard),
            ('origin', q3.one_hot)]

# Construct the standard data and label arrays
#auto_data[0] has the features for choice features1
#auto_data[1] has the features for choice features2
#The labels for both are the same, and are in auto_values
auto_data = [0, 0]
auto_values = 0
auto_data[0], auto_values = q3.auto_data_and_values(auto_data_all, features1)
auto_data[1], _ = q3.auto_data_and_values(auto_data_all, features2)

#standardize the y-values
auto_values, mu, sigma = q3.std_y(auto_values)

#-------------------------------------------------------------------------------
# Analyze auto data
#-------------------------------------------------------------------------------     
        
#Your code for cross-validation goes here

def cross_validation(X, y, lam_values, k=10):
    """Performs cross-validation on the dataset for different lambda values."""
    rmse_results = {}
    
    for lam in lam_values:
        rmse = q3.xval_learning_alg(X, y, lam, k)
        rmse_results[lam] = rmse
        
    # Return the lambda with the smallest RMSE
    best_lam = min(rmse_results, key=rmse_results.get)
    return best_lam, rmse_results[best_lam]

# Define lambda ranges for the grid search
lambda_values = np.linspace(0, 0.1, 11)

# Perform cross-validation on both feature sets
best_lambda1, rmse1 = cross_validation(auto_data[0], auto_values, lambda_values)
best_lambda2, rmse2 = cross_validation(auto_data[1], auto_values, lambda_values)

# Print results
print(f"Feature set 1: Best lambda = {best_lambda1}, RMSE = {rmse1}")
print(f"Feature set 2: Best lambda = {best_lambda2}, RMSE = {rmse2}")