# import all necessary libraries here
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Impute Missing Values
def impute_missing_values(data, strategy='mean'): #Define a function for imputing missing values, parameters are the data source and the strategy for imputing
    
    data_copy = data.copy() #make a copy of the data to avoid messing up the original

    for column in data_copy.columns:
        if column == 'target': #skip the target column
            continue

        if data_copy[column].isnull().sum() > 0: #if the column is missing more than 0 entries, replace it with either the mean, median or mode of that column (for numerical columns only)
            if data_copy[column].dtype in ['float64', 'int64']:
                if strategy == 'mean':
                    data_copy[column].fillna(data_copy[column].mean(), inplace=True)
                elif strategy == 'median':
                    data_copy[column].fillna(data_copy[column].median(), inplace=True)
                elif strategy == 'mode':
                    data_copy[column].fillna(data_copy[column].mode()[0], inplace=True)
            else:
                data_copy[column].fillna(data_copy[column].mode()[0], inplace=True) #for non numerical columns, always just use the mode to fill in missing data.

    return data_copy

    # TODO: Fill missing values based on the specified strategy
    


# 2. Remove Duplicates
def remove_duplicates(data): #define a function called remove duplicates, with parameter data.
    """
    Remove duplicate rows from the dataset.
    :param data: pandas DataFrame
    :return: pandas DataFrame
    """
    # TODO: Remove duplicate rows
    return data.drop_duplicates() #Within the data variable, remove any duplicates.

# 3. Normalize Numerical Data
def normalize_data(data,method='minmax'): #Define a function called normalize_data, with parameters of data and method (default method is minmax)
    """Apply normalization to numerical features.
    :param data: pandas DataFrame
    :param method: str, normalization method ('minmax' (default) or 'standard')
    """
    # TODO: Normalize numerical data using Min-Max or Standard scaling
    data_copy = data.copy() #Make a copy of the data first before messing with it

    numeric_cols = data_copy.select_dtypes(include=['float64', 'int64']).columns #If the column is numeric, add it to the variable numeric_cols
    numeric_cols = [col for col in numeric_cols if col != 'target'] #Target column is also numeric, this line ensures we exclude it though

    if method == 'minmax': #Decide if the method of normalization is minmax or standard, and assign it to the variable scaler
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Invalid method. Use 'minmax' or 'standard'.")

    data_copy[numeric_cols] = scaler.fit_transform(data_copy[numeric_cols]) #Actually transform the data using the scaler variable which has the assigned method.

    return data_copy

# 4. Remove Redundant Features   
def remove_redundant_features(data, threshold=0.9): #Define a function called remove_redundant_features with the parameters data and threshold, which is 0.9 by default. Threshold refers to correlation coefficient that will define what are redundant features
    """Remove redundant or duplicate columns.
    :param data: pandas DataFrame
    :param threshold: float, correlation threshold
    :return: pandas DataFrame
    """
    # TODO: Remove redundant features based on the correlation threshold (HINT: you can use the corr() method)
    data_copy = data.copy() #Make a copy of the data
    numeric_data = data_copy.select_dtypes(include=['float64', 'int64']) #Assign numeric columns to a variable called numeric_data
    corr_matrix = numeric_data.corr().abs() #Using numeric_data, create a correlation matrix and check the correlation of each feature to each other. Abs ensures we look at absolute values, whether negative or positive corelations are revealed
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)) #This line ensures we do not repeat comparisons. For example, we don't have to compare A --> B and also B --> A. Only one of those is fine since both will have the same correlation
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)] #If the correlation between features is higher than the threshold (0.9 by default), assign them to the variable to_drop
    data_copy.drop(columns=to_drop, inplace=True) #Drop the columns in the to_drop variable. We always drop the variable that occurs later in the data frame. So if A --> B correlation is >0.9, we would drop B.
    return data_copy

# ---------------------------------------------------

def simple_model(input_data, split_data=True, scale_data=False, print_report=False):
    """
    A simple logistic regression model for target classification.
    Parameters:
    input_data (pd.DataFrame): The input data containing features and the target variable 'target' (assume 'target' is the first column).
    split_data (bool): Whether to split the data into training and testing sets. Default is True.
    scale_data (bool): Whether to scale the features using StandardScaler. Default is False.
    print_report (bool): Whether to print the classification report. Default is False.
    Returns:
    None
    The function performs the following steps:
    1. Removes columns with missing data.
    2. Splits the input data into features and target.
    3. Encodes categorical features using one-hot encoding.
    4. Splits the data into training and testing sets (if split_data is True).
    5. Scales the features using StandardScaler (if scale_data is True).
    6. Instantiates and fits a logistic regression model.
    7. Makes predictions on the test set.
    8. Evaluates the model using accuracy score and classification report.
    9. Prints the accuracy and classification report (if print_report is True).
    """

    # if there's any missing data, remove the columns
    input_data.dropna(inplace=True)

    # split the data into features and target
    target = input_data.copy()[input_data.columns[0]]
    features = input_data.copy()[input_data.columns[1:]]

    # if the column is not numeric, encode it (one-hot)
    for col in features.columns:
        if features[col].dtype == 'object':
            features = pd.concat([features, pd.get_dummies(features[col], prefix=col)], axis=1)
            features.drop(col, axis=1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=target, random_state=42)

    if scale_data:
        # scale the data
        X_train = normalize_data(X_train)
        X_test = normalize_data(X_test)
        
    # instantiate and fit the model
    log_reg = LogisticRegression(random_state=42, max_iter=100, solver='liblinear', penalty='l2', C=1.0)
    log_reg.fit(X_train, y_train)

    # make predictions and evaluate the model
    y_pred = log_reg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy}')
    
    # if specified, print the classification report
    if print_report:
        print('Classification Report:')
        print(report)
        print('Read more about the classification report: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html and https://www.nb-data.com/p/breaking-down-the-classification')
    
    return None