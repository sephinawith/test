import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_data():
    train_log = pd.read_csv('prediction_log/train_log.csv').replace('', np.nan)
    train_truth = pd.read_csv('prediction_log/train_truth.csv').replace('', np.nan)
    test_log = pd.read_csv('prediction_log/test_log.csv').replace('', np.nan)
    test_truth = pd.read_csv('prediction_log/test_truth.csv').replace('', np.nan)
    user_info = pd.read_csv('prediction_log/user_info.csv').replace('', np.nan)
    course_info = pd.read_csv('prediction_log/course_info.csv').replace('', np.nan)
    return train_log, train_truth, test_log, test_truth, user_info, course_info

def merge_data(train_log, train_truth, test_log, test_truth, user_info, course_info):
    train_data = pd.merge(train_log, train_truth, on='enroll_id')
    test_data = pd.merge(test_log, test_truth, on='enroll_id')
    train_data = pd.merge(train_data, user_info, left_on='username', right_on='user_id')
    test_data = pd.merge(test_data, user_info, left_on='username', right_on='user_id')
    train_data = pd.merge(train_data, course_info, on='course_id')
    test_data = pd.merge(test_data, course_info, on='course_id')
    return train_data, test_data

def feature_engineering(train_data, test_data):
    le = LabelEncoder()
    
    train_grouped = train_data.groupby('enroll_id').agg({
        'action': ['count', 'nunique'],
        'gender': 'first',
        'education': 'first',
        'course_type': 'first',
        'category': 'first',
        'truth': 'first'
    }).reset_index()
    
    test_grouped = test_data.groupby('enroll_id').agg({
        'action': ['count', 'nunique'],
        'gender': 'first',
        'education': 'first',
        'course_type': 'first',
        'category': 'first',
        'truth': 'first'
    }).reset_index()
    
    train_grouped.columns = ['enroll_id', 'action_count', 'action_nunique', 'gender', 'education', 'course_type', 'category', 'truth']
    test_grouped.columns = ['enroll_id', 'action_count', 'action_nunique', 'gender', 'education', 'course_type', 'category', 'truth']
    
    for col in ['gender', 'education', 'course_type', 'category']:
        train_grouped[col] = le.fit_transform(train_grouped[col].astype(str))
        test_grouped[col] = le.transform(test_grouped[col].astype(str))
    
    train_grouped = train_grouped.fillna(0)
    test_grouped = test_grouped.fillna(0)
    
    X_train = train_grouped.drop(['enroll_id', 'truth'], axis=1)
    y_train = train_grouped['truth']
    X_test = test_grouped.drop(['enroll_id', 'truth'], axis=1)
    y_test = test_grouped['truth']
    
    return X_train, y_train, X_test, y_test


def train_evaluate_model(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f'Accuracy: {accuracy:.4f}')
    print('Classification Report:')
    print(report)
    
    return model


def main():
    train_log, train_truth, test_log, test_truth, user_info, course_info = load_data()
    train_data, test_data = merge_data(train_log, train_truth, test_log, test_truth, user_info, course_info)
    print(train_data.shape)
    print(train_data.head(2))
    X_train, y_train, X_test, y_test = feature_engineering(train_data, test_data)
    print(X_train.shape)
    model = train_evaluate_model(X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    main()