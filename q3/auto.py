import numpy as np
import q3 as q3
import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

#-------------------------------------------------------------------------------
# Auto Data
#-------------------------------------------------------------------------------

# Load the auto-mpg-regression.tsv data
def load_auto_data(path_data):
    """Load the auto-mpg-regression.tsv file."""
    auto_data_all = []
    
    with open(path_data, 'r') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        
        for row in reader:
            # Convert appropriate fields to numeric types
            for key in row:
                try:
                    row[key] = float(row[key])
                except ValueError:
                    pass
            
            auto_data_all.append(row)
    
    return auto_data_all

# Load the data
auto_data_all = load_auto_data('C:/Users/Danik/Documents/GitHub/AIproject1/q3/auto-mpg-regression.tsv')

# Feature selection and processing (use one-hot encoding for categorical variables)
features = [('cylinders', q3.one_hot), 
            ('displacement', q3.standard),
            ('horsepower', q3.standard),
            ('weight', q3.standard),
            ('acceleration', q3.standard),
            ('origin', q3.one_hot)]

# Get data and labels
auto_data = [0]
auto_values = 0
auto_data[0], auto_values = q3.auto_data_and_values(auto_data_all, features)

# Convert continuous 'mpg' into binary classification (above/below median mpg)
y = np.array(auto_values).ravel()  # Ensure y is a 1D array
X = auto_data[0].T  # The feature matrix

# Binarize the target variable (above/below median mpg)
median_value = np.median(y)
y_binned = np.where(y > median_value, 1, 0)  # 1 if above median, 0 otherwise

print("Shape of X:", X.shape)  # Should be (392, 12)
print("Shape of y_binned:", y_binned.shape)  # Should be (392,)


#-------------------------------------------------------------------------------
# KNN Classification with Cross-Validation and Evaluation Metrics
#-------------------------------------------------------------------------------

def knn_classification(X, y, k_values, n_splits=10):
    """Perform KNN classification with k-fold cross-validation using multiple metrics."""
    
    # Initialize results dictionary to store metrics for each K value
    results = {k: {'precision': [], 'recall': [], 'f1': [], 'accuracy': []} for k in k_values}

    # Scale the features (important for KNN)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Fold Cross Validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Iterate over different K values
    for k in k_values:
        print(f"Evaluating for K = {k}")
        knn = KNeighborsClassifier(n_neighbors=k)

        # Iterate over each fold
        for train_index, test_index in kf.split(X_scaled):
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Train KNN classifier
            knn.fit(X_train, y_train)
            
            # Make predictions
            y_pred = knn.predict(X_test)
            
            # Calculate metrics
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Store results
            results[k]['precision'].append(precision)
            results[k]['recall'].append(recall)
            results[k]['f1'].append(f1)
            results[k]['accuracy'].append(accuracy)
    
    return results

# Define K values to test
k_values = [3, 5, 10]

# Perform KNN classification and evaluate results
results = knn_classification(X, y_binned, k_values)

# Display results in a table-like format
for k in k_values:
    print(f"K = {k}")
    print(f"  Average Accuracy: {np.mean(results[k]['accuracy']):.4f}")
    print(f"  Average Precision: {np.mean(results[k]['precision']):.4f}")
    print(f"  Average Recall: {np.mean(results[k]['recall']):.4f}")
    print(f"  Average F1-Score: {np.mean(results[k]['f1']):.4f}")
