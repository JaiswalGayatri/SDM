import numpy as np
import pandas as pd 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import pdb 


feature_cols = [
        'annual_mean_temperature', 'mean_diurnal_range', 'isothermality',
        'temperature_seasonality', 'max_temperature_warmest_month', 'min_temperature_coldest_month',
        'temperature_annual_range', 'mean_temperature_wettest_quarter', 'mean_temperature_driest_quarter',
        'mean_temperature_warmest_quarter', 'mean_temperature_coldest_quarter', 'annual_precipitation',
        'precipitation_wettest_month', 'precipitation_driest_month', 'precipitation_seasonality',
        'precipitation_wettest_quarter', 'precipitation_driest_quarter', 'precipitation_warmest_quarter',
        'precipitation_coldest_quarter', 'aridity_index', 'topsoil_ph', 'subsoil_ph', 'topsoil_texture',
        'subsoil_texture', 'elevation'
    ]


class Models:
    def __init__(self):
        return 
    def load_data(self,presence_path = 'data/presence.csv' ,absence_path='data/pseudo_absence.csv'):
        presence_df = pd.read_csv(presence_path)
        absence_df = pd.read_csv(absence_path)


       

        presence_coords = presence_df[['longitude', 'latitude']].values

        absence_coords = absence_df[['longitude', 'latitude']].values
        

        # Get feature columns (bio01 through bio19)


        # Extract features
        # print(feature_cols)
        # print(presence_df.columns)
        presence_features = presence_df[feature_cols].values
        absence_features = absence_df[feature_cols].values

        # Combine features and create labels
        X = np.vstack([presence_features, absence_features])
        y = np.hstack([np.ones(len(presence_features)), np.zeros(len(absence_features))])
        coords = np.vstack([presence_coords, absence_coords])

        # Assign weights
        presence_weights = np.ones(len(presence_features))  # Weight 1 for all presence points

        # Normalize reliability for absence points
        reliability = absence_df['reliability'].values
        min_reliability = np.min(reliability)
        max_reliability = np.max(reliability)
        if max_reliability!=min_reliability:
            absence_weights = ((reliability - min_reliability) / (max_reliability - min_reliability))
        else:
            absence_weights = [1 for i in range(len(reliability))]
        absence_weights = [i**(0.1) for i in absence_weights]

        # Combine presence and absence weights
        sample_weights = np.hstack([presence_weights, absence_weights])

        # Shuffle the data
        X, y, coords, sample_weights = shuffle(X, y, coords, sample_weights, random_state=42)

        return X, y, coords, feature_cols, sample_weights
    

    def RandomForest(self,X, y):


        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train model

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        # Evaluate model
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        # print(f"Accuracy_RFC: {accuracy_score(y_test, y_pred):.4f}")
        # print("\nConfusion Matrix:")
        # print(confusion_matrix(y_test, y_pred))
        # print("\nClassification Report:")
        # print(classification_report(y_test, y_pred))

        # Print probabilities for the test set
        # print("\nProbabilities on the test set:")
        # for i, prob in enumerate(y_proba):
        #     print(f"Sample {i}: {prob:.4f}")

        # # Print feature importances
        # print("\nFeature Importances:")
        # for importance, feature in sorted(zip(clf.feature_importances_, feature_cols), reverse=True):
        #     print(f"{feature}: {importance:.4f}")

        return clf, X_test, y_test, y_pred, y_proba
    


    def logistic_regression_L2(self,X, y):

        imputer = SimpleImputer(strategy="mean")
        X = imputer.fit_transform(X)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train model with Logistic Regression and L2 regularization
        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', random_state=42)
        clf.fit(X_train, y_train)

        # Evaluate model
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        # print(f"Accuracy_logistic: {accuracy_score(y_test, y_pred):.4f}")
        # print("\nConfusion Matrix:")
        # print(confusion_matrix(y_test, y_pred))
        # print("\nClassification Report:")
        # print(classification_report(y_test, y_pred))

        # print("\nFeature Importances (Coefficients):")
        # for coef, feature in sorted(zip(clf.coef_[0], feature_cols), key=lambda x: abs(x[0]), reverse=True):
        #     print(f"{feature}: {coef:.4f}")

        return clf, X_test, y_test, y_pred, y_proba
    
    def train_and_evaluate_model_logistic_weighted(self,X, y, sample_weights=None):
        # First, ensure X is a numeric array
        # You might need to add debug statements to check what X contains
        print(f"Type of X: {type(X)}")
        print(f"Shape of X: {X.shape if hasattr(X, 'shape') else 'no shape'}")
        
        # If sample_weights is None, create an array of ones with the same length as y
        if sample_weights is None:
            sample_weights = np.ones(len(y))
        
        # Initialize the imputer
        imputer = SimpleImputer(strategy="mean")
        
        # Transform X
        X = imputer.fit_transform(X)
        
        # Split the data
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
            X, y, sample_weights, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model with Logistic Regression and L2 regularization
        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', random_state=42)
        clf.fit(X_train, y_train, sample_weight=weights_train)
        
        # Evaluate model
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        
        return clf, X_test, y_test, y_pred, y_proba
    

    def evaluate_model(clf: RandomForestClassifier, X_test, y_test, sample_weights=None, dataset_name='Test'):
        # If sample_weights are provided, use them in the prediction step.
        try:
            # If no sample_weights, just perform prediction directly
            y_pred = clf.predict(X_test)

   
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None

        # If sample_weights are provided, calculate accuracy and metrics with them
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        # Print the results
        print(f"\n{dataset_name} Set Evaluation:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        print("\nClassification Report:")
        print(metrics['classification_report'])
        
        # Return the evaluation metrics
        return metrics
