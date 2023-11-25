# Starter code for CS 165B MP3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def compute_metric(labels, expected):
    tp = np.sum(labels[expected == 1])
    fp = np.sum(labels[expected == 0])
    tn = np.sum(1-labels[expected == 0])
    fn = np.sum(1-labels[expected == 1])
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    error_rate = (fp+fn)/(tp+fp+tn+fn)
    accuracy = (tp+tn)/(tp+fp+tn+fn)
    precision = tp/(tp+fp)
    f1 = 2*tp/(2*tp+fp+fn)

    return {
        "f1": f1,
        "accuracy": accuracy,
        "precision": precision,
        "tpr": tpr,
        "fpr": fpr,
        "error_rate": error_rate,
    }

def preprocess_data(data):
    X = data.drop("target", axis=1)

    # Identify categorical and numerical features
    numerical_features = X.select_dtypes(exclude=['object']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Pipeline: apply list of transformers to data
    # SimpleImputer: used for handling missing values, missing values filled with median (numerical), missing values filled with mode (categorical)
    # OneHotEncoder: used to convert categorical features to binary value (1 or 0), ignore categories present in test set but not in train set
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # ColumnTransformer: apply different transformers to different subsets of data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor

def run_train_test(training_data: pd.DataFrame, testing_data: pd.DataFrame):
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.

    Inputs:
        training_data: 
        testing_data: the same as training_data with "target" removed.

    Output:
        testing_prediction: List[int]
    Example output:
    return random.choices([0, 1, 2], k=len(testing_data))
    """

    preprocessor = preprocess_data(training_data)
    X_train = preprocessor.fit_transform(training_data)
    X_test = preprocessor.transform(testing_data)

    y_train = training_data["target"]

    # Edit Hyperparameters to get best possible F1 Score 
    param_grid = {
        'n_estimators': [7],
        'max_depth': [None],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'bootstrap': [True]
    }

    # Train a Random Forest Classifier
    clf = RandomForestClassifier(random_state=42)
    
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)

    # Make predictions on the testing set
    best_clf = grid_search.best_estimator_
    testing_prediction = best_clf.predict(X_test)

    return testing_prediction

if __name__ == '__main__':
    # Load data
    training = pd.read_csv('./data/train.csv')
    development = pd.read_csv('./data/dev.csv')

    # Extract target labels and drop from development set
    target_label = development['target']
    development.drop('target', axis=1, inplace=True)

    # Run training and testing
    prediction = run_train_test(training, development)

    # Compute and print metrics
    target_label = target_label.values
    status = compute_metric(prediction, target_label)
    print(status)
