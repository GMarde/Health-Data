import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler


data_train = pd.read_csv('Train.csv')
data_test = pd.read_csv('Test.csv')

data_train
data_test

data_train.head()
data_test.head()

#%%

missing_values = data_train.isnull().sum()

# Get descriptive statistics of the numeric variables
stats_description = data_train.describe()

missing_values, stats_description

#%%

# Data Preprocessing: Blood Pressure

# Split the column 'Blood Pressure (s/d)' into two columns: 'Systolic Pressure' and 'Diastolic Pressure'
data_train[['Systolic Pressure', 'Diastolic Pressure']] = data_train['Blood Pressure (s/d)'].str.split('/', expand=True)
data_test[['Systolic Pressure', 'Diastolic Pressure']] = data_test['Blood Pressure (s/d)'].str.split('/', expand=True)

# Convert the new columns to numeric type if necessary
data_train['Systolic Pressure'] = pd.to_numeric(data_train['Systolic Pressure'], errors='coerce')
data_train['Diastolic Pressure'] = pd.to_numeric(data_train['Diastolic Pressure'], errors='coerce')

data_test['Systolic Pressure'] = pd.to_numeric(data_test['Systolic Pressure'], errors='coerce')
data_test['Diastolic Pressure'] = pd.to_numeric(data_test['Diastolic Pressure'], errors='coerce')

# Optional: remove the original blood pressure column
data_train = data_train.drop(columns=['Blood Pressure (s/d)'])
data_test = data_test.drop(columns=['Blood Pressure (s/d)'])

# Display the first rows to check the result
print(data_train[['Systolic Pressure', 'Diastolic Pressure']].head())
print(data_test[['Systolic Pressure', 'Diastolic Pressure']].head())


# Get the current list of columns
cols_train = list(data_train.columns)
cols_test = list(data_test.columns)

# Remove 'Systolic Pressure' and 'Diastolic Pressure' from their current positions
cols_train.remove('Systolic Pressure')
cols_train.remove('Diastolic Pressure')

cols_test.remove('Systolic Pressure')
cols_test.remove('Diastolic Pressure')

# Insert 'Systolic Pressure' and 'Diastolic Pressure' at the desired positions (3 and 4)
cols_train.insert(3, 'Systolic Pressure')
cols_train.insert(4, 'Diastolic Pressure')

cols_test.insert(3, 'Systolic Pressure')
cols_test.insert(4, 'Diastolic Pressure')

# Reorder the data_trainFrame with the new column order
data_train = data_train[cols_train]
data_test = data_test[cols_test]

# Display the first few rows to verify the new order
print(data_train.head())
print(data_test.head())


#%%

# Convert categorical columns into numeric variables

categorical_columns = ['Gender', 'Smoking Status', 'Alcohol Consumption', 
                       'Diet', 'Chronic Diseases', 'Medication Use', 'Family History', 
                       'Mental Health Status', 'Sleep Patterns', 'Education Level', 'Income Level', 'Physical Activity Level']

# Convert categorical columns into numeric variables using the get_dummies method (one-hot encoding)
data_train_encoded = pd.get_dummies(data_train, columns=categorical_columns, drop_first=True, dtype=int)
data_test_encoded = pd.get_dummies(data_test, columns=categorical_columns, drop_first=True, dtype=int)

# Display the first rows after encoding
data_train_encoded.head()
data_test_encoded.head()


#%%

# Splitting the data and training the model

# Separate the features (X) and target (Y) in the training data
X_train = data_train_encoded.drop(columns=['Age (years)'])
y_train = data_train_encoded['Age (years)']

# Test data only has features
X_test = data_test_encoded

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data (X_train)
X_train_scaled = scaler.fit_transform(X_train)

# Use the same scaler to transform the test data (X_test)
X_test_scaled = scaler.transform(X_test)

# Convert the scaled arrays back to DataFrames for easier handling (optional)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Display the scaled training data
print(X_train_scaled_df.head())

#%%


# Add a constant to the training data (for the intercept)
X_train_scaled_const = sm.add_constant(X_train_scaled)

# Fit the OLS (Ordinary Least Squares) regression model
model = sm.OLS(y_train, X_train_scaled_const)
results = model.fit()

# Print out the statistical summary of the model
print(results.summary())

# Add a constant to the test data as well
X_test_scaled_const = sm.add_constant(X_test_scaled)

# Make predictions on the test data
y_pred_statsmodels = results.predict(X_test_scaled_const)

# Display the first few predictions
print("Predictions on test data: ", y_pred_statsmodels[:5])


#%%


# Stepwise function for backward elimination
def backward_elimination(data, target, significance_level=0.05):
    features = data.columns.tolist()
    while len(features) > 0:
        # Add a constant to the model for the intercept
        X = sm.add_constant(data[features])
        
        # Fit the model
        model = sm.OLS(target, X).fit()
        
        # Get the p-values for the model
        p_values = model.pvalues.iloc[1:]  # Exclude the intercept
        
        # Find the predictor with the highest p-value
        max_p_value = p_values.max()
        if max_p_value > significance_level:
            # Remove the predictor with the highest p-value
            excluded_feature = p_values.idxmax()
            features.remove(excluded_feature)
            print(f"Removing {excluded_feature} with p-value {max_p_value}")
        else:
            break  # Stop if all p-values are below the significance level
            
    # Return the final model with the remaining features
    return model

# Applying backward elimination on your data
X_train_scaled_const = sm.add_constant(X_train_scaled_df)
final_model = backward_elimination(X_train_scaled_df, y_train)

# Display the final model summary
print(final_model.summary())

#%%

# Preprocessing the test data (add a constant for the intercept if using statsmodels)
X_test_scaled_const = sm.add_constant(X_test_scaled)

# Use the trained model (e.g., 'results' from statsmodels) to make predictions
y_test_pred = results.predict(X_test_scaled_const)

# Display the first few predictions
print("Predicted ages in the test set: ", y_test_pred[:5])

# If needed, you can save the predictions to a CSV file for further analysis
import pandas as pd
test_predictions = pd.DataFrame(y_test_pred, columns=["Predicted_Age"])
test_predictions.to_csv('predicted_test_ages.csv', index=False)

