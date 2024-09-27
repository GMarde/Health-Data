import pandas as pd

from sklearn.preprocessing import StandardScaler


data_train = pd.read_csv('Train.csv')
data_test = pd.read_csv('Test.csv')

data_train
data_test

data_train.head()
data_test.head()

#%%

missing_values = data_train.isnull().sum()

# Obter estatísticas descritivas das variáveis numéricas
stats_description = data_train.describe()

missing_values, stats_description

#%%

# Separar a coluna 'Blood Pressure (s/d)' em duas colunas: 'Systolic Pressure' e 'Diastolic Pressure'
data_train[['Systolic Pressure', 'Diastolic Pressure']] = data_train['Blood Pressure (s/d)'].str.split('/', expand=True)
data_test[['Systolic Pressure', 'Diastolic Pressure']] = data_test['Blood Pressure (s/d)'].str.split('/', expand=True)

# Converter as novas colunas para o tipo numérico, caso necessário
data_train['Systolic Pressure'] = pd.to_numeric(data_train['Systolic Pressure'], errors='coerce')
data_train['Diastolic Pressure'] = pd.to_numeric(data_train['Diastolic Pressure'], errors='coerce')

data_test['Systolic Pressure'] = pd.to_numeric(data_test['Systolic Pressure'], errors='coerce')
data_test['Diastolic Pressure'] = pd.to_numeric(data_test['Diastolic Pressure'], errors='coerce')

# Opcional: eliminar a coluna original de pressão arterial
data_train = data_train.drop(columns=['Blood Pressure (s/d)'])
data_test = data_test.drop(columns=['Blood Pressure (s/d)'])

# Exibir as primeiras linhas para verificar o resultado
print(data_train[['Systolic Pressure', 'Diastolic Pressure']].head())
print(data_test[['Systolic Pressure', 'Diastolic Pressure']].head())


#%%

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

categorical_columns = ['Gender', 'Smoking Status', 'Alcohol Consumption', 
                       'Diet', 'Chronic Diseases', 'Medication Use', 'Family History', 
                       'Mental Health Status', 'Sleep Patterns', 'Education Level', 'Income Level', 'Physical Activity Level']

# Converter colunas categóricas em variáveis numéricas usando o método get_dummies (one-hot encoding)
data_train_encoded = pd.get_dummies(data_train, columns=categorical_columns, drop_first=True, dtype=int)
data_test_encoded = pd.get_dummies(data_test, columns=categorical_columns, drop_first=True, dtype=int)

# Exibir as primeiras linhas após a codificação
data_train_encoded.head()
data_test_encoded.head()


#%%


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

import statsmodels.api as sm

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

correlation_matrix = data_train_encoded.corr()
correlation_matrix