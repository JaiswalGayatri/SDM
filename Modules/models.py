import numpy as np
import pandas as pd 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score
from sklearn.linear_model import LogisticRegression
import geopandas as gpd
from shapely.geometry import Point
from shapely import wkt
import os
from .custom_losses import CustomNeuralNetwork, FocalLoss, DiceLoss, TverskyLoss, TverskyScorer, FocalScorer, DiceScorer, optimize_threshold_for_tpr

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

    def load_data(self, presence_path='data/presence.csv', absence_path='data/pseudo_absence.csv'):
        # ------------------------------------
        # Load presence and absence data
        # ------------------------------------
        presence_df = pd.read_csv(presence_path)
        absence_df = pd.read_csv(absence_path)
        # print('len',len(presence_df))
        # print('len',len(absence_df))
        
        # ------------------------------------
        # 1. Reliability-based Weights
        # ------------------------------------
        # For presence samples, set weight = 1
        reliability_presence = np.ones(len(presence_df))
        # For absence samples, use the 'reliability' column
        if 'reliability' in absence_df.columns:
            absence_df['reliability'] = absence_df['reliability'].fillna(1)
        reliability = absence_df['reliability'].values
        min_rel = np.min(reliability)
        max_rel = np.max(reliability)
        if max_rel != min_rel:
            reliability_absence = (reliability - min_rel) / (max_rel - min_rel)
        else:
            reliability_absence = np.ones(len(reliability))
        # Apply a mild power transformation to reduce extremes
        reliability_absence = np.array([w**(0.1) for w in reliability_absence])
        reliability_weights = np.hstack([reliability_presence, reliability_absence])
        
        # ------------------------------------
        # 2. Bias-correction Weights
        # ------------------------------------
        # Read eco-region counts file; expects columns "ecoregion" and "count"
        counts_file = "outputs/testing_SDM_out/species_ecoregion_count_1.csv"
        if os.path.exists(counts_file):
            region_counts_df = pd.read_csv(counts_file)
            # print('region counts df length', len(region_counts_df))
            # Compute raw weight: 1 / ln(count + 1)
            region_counts_df['raw_weight'] = 1 / (region_counts_df['count'] + 1)
            # Normalize raw weights to a subtle range, e.g., [0.5, 1.5]
            min_w = region_counts_df['raw_weight'].min()
            max_w = region_counts_df['raw_weight'].max()
            region_counts_df['eco_weight'] = 0.5 + region_counts_df['raw_weight'] 
            # Create mapping: eco_region -> eco_weight
            eco_weight_dict = region_counts_df.set_index('ecoregion')['eco_weight'].to_dict()
            # print("Eco-region weight mapping:", eco_weight_dict)
        else:
            print(f"Warning: {counts_file} not found. Defaulting eco weights to 1.")
            eco_weight_dict = {}
        
        # To assign bias weights, we need to know in which eco-region a point falls.
        # Since our presence/absence CSV files do not contain an 'eco_region' column,
        # we convert them into GeoDataFrames and perform a spatial join with the eco-region polygons.
        # Define path to the eco-region polygons folder:
        ecoregion_folder = "data/eco_regions_polygon"
        
        # Convert presence and absence data to GeoDataFrames.
        presence_df["geometry"] = presence_df.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)
        absence_df["geometry"] = absence_df.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)
        presence_gdf = gpd.GeoDataFrame(presence_df, geometry="geometry", crs="EPSG:4326")
        absence_gdf = gpd.GeoDataFrame(absence_df, geometry="geometry", crs="EPSG:4326")
        
        # Combine them to preserve the original ordering (presence first, then absence)
        combined_gdf = pd.concat([presence_gdf, absence_gdf], ignore_index=True)
        # print('len',len(presence_df))
        # print('len',len(absence_df))
        # print('len',len(combined_gdf))
        # Function to load eco-region polygons from WKT files
        def load_ecoregions(folder):
            ecoregions = []
            for file in os.listdir(folder):
                if file.endswith(".wkt"):
                    with open(os.path.join(folder, file), "r") as f:
                        wkt_text = f.read().strip()
                        poly = wkt.loads(wkt_text)
                        ecoregions.append({"ecoregion": file.replace(".wkt", ""), "geometry": poly})
            return gpd.GeoDataFrame(ecoregions, geometry="geometry", crs="EPSG:4326")
        
        # Load eco-region polygons
        ecoregion_gdf = load_ecoregions(ecoregion_folder)
        
        # Spatial join: assign each point to an eco-region (using the "within" predicate)
        combined_with_ecoregion = gpd.sjoin(combined_gdf, ecoregion_gdf, how="left", predicate="within")
        # The resulting DataFrame will have a column 'ecoregion' from the polygons (if matched)
        
        # Define a function to return the bias weight for a given eco-region:
        def get_bias_weight(eco):
            if pd.isna(eco):
                return 1
            else:
                return eco_weight_dict.get(eco, 1)
        
        # Apply the mapping: this returns an array of bias weights (one per point)
        bias_weights = combined_with_ecoregion['ecoregion'].apply(get_bias_weight).values
        
        # Save bias weights (with coordinates) to CSV for inspection
        coords_bias = np.column_stack((combined_with_ecoregion.geometry.x, combined_with_ecoregion.geometry.y))
        bias_df = pd.DataFrame(coords_bias, columns=["longitude", "latitude"])
        bias_df["bias_weight"] = bias_weights
        output_bias_file = "outputs/bias_weights.csv"
        os.makedirs(os.path.dirname(output_bias_file), exist_ok=True)
        # bias_df.to_csv(output_bias_file, index=False)
        # print(f"Bias weights saved to {output_bias_file}")
        
        # ------------------------------------
        # 3. Feature Extraction & Combination
        # ------------------------------------
        # For features, we use the original CSV data (without the geometry columns)
        presence_features = presence_df[feature_cols].values
        absence_features = absence_df[feature_cols].values
        X = np.vstack([presence_features, absence_features])
        y = np.hstack([np.ones(len(presence_features)), np.zeros(len(absence_features))])
        # For coordinates, we extract them from the combined GeoDataFrame (order preserved)
        coords = np.column_stack((combined_with_ecoregion.geometry.x, combined_with_ecoregion.geometry.y))
        
        # ------------------------------------
        # Shuffle the data along with both sets of weights
        # ------------------------------------
        X, y, coords, reliability_weights, bias_weights = shuffle(
            X, y, coords, reliability_weights, bias_weights, random_state=42
        )
        
        # Return features, labels, coordinates, feature column names,
        # reliability-based weights, and bias-correction weights.
        return X, y, coords, feature_cols, reliability_weights, bias_weights


    def RandomForest(self, X, y, sample_weights=None):
        # Split data along with indices
        indices = np.arange(len(y))
        X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
            X, y, indices, test_size=0.2, random_state=42, stratify=y
        )
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        if sample_weights is not None:
            # Ensure sample_weights is a 1D array
            sample_weights = np.ravel(sample_weights)
            # Subset the weights using the training indices
            weights_train = sample_weights[indices_train]
            clf.fit(X_train, y_train, sample_weight=weights_train)
        else:
            clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        return clf, X_test, y_test, y_pred, y_proba

    def logistic_regression_L2(self, X, y):
        X = np.array(X, dtype=float)
        mask = np.isfinite(X).all(axis=1)
        X, y = X[mask], y[mask]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        train_mask = np.isfinite(X_train).all(axis=1)
        test_mask = np.isfinite(X_test).all(axis=1)
        X_train, y_train = X_train[train_mask], y_train[train_mask]
        X_test, y_test = X_test[test_mask], y_test[test_mask]
        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        return clf, X_test, y_test, y_pred, y_proba

    def train_and_evaluate_model_logistic_weighted(self, X, y, sample_weights=None):
        # Convert to numpy arrays and handle NaN values
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        # Remove rows with NaN values
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y[mask]
        if sample_weights is not None:
            sample_weights = np.array(sample_weights, dtype=float)
            sample_weights = sample_weights[mask]
        
        # Split data
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
            X, y, sample_weights, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', random_state=42)
        if weights_train is not None:
            clf.fit(X_train, y_train, sample_weight=weights_train)
        else:
            clf.fit(X_train, y_train)
        
        # Get predictions and probabilities
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        
        return clf, X_test, y_test, y_pred, y_proba

    def evaluate_model(self, clf: RandomForestClassifier, X_test, y_test, sample_weights=None, dataset_name='Test'):
        try:
            y_pred = clf.predict(X_test)
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
        print(f"\n{dataset_name} Set Evaluation:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        print("\nClassification Report:")
        print(metrics['classification_report'])
        return metrics

    def train_with_focal_loss(self, X, y, sample_weights=None, alpha=0.25, gamma=2.0):
        """Train a model using focal loss to improve handling of class imbalance"""
        nn_model = CustomNeuralNetwork(loss_fn='focal', alpha=alpha, gamma=gamma)
        nn_model.fit(X, y, sample_weights=sample_weights)
        return nn_model

    def train_with_dice_loss(self, X, y, sample_weights=None, smooth=1.0):
        """Train a model using dice loss to focus on true positives"""
        nn_model = CustomNeuralNetwork(loss_fn='dice', smooth=smooth)
        nn_model.fit(X, y, sample_weights=sample_weights)
        return nn_model

    def train_with_tversky_loss(self, X, y, sample_weights=None, alpha=0.3, beta=0.7, smooth=1.0):
        """
        Train a model using Tversky loss to handle class imbalance with explicit control
        over false positives and false negatives.
        
        Parameters:
        -----------
        X: features
        y: labels
        sample_weights: optional sample weights
        alpha: penalty for false positives (default 0.3)
        beta: penalty for false negatives (default 0.7)
        smooth: smoothing factor (default 1.0)
        """
        nn_model = CustomNeuralNetwork(
            loss_fn='tversky',
            alpha=alpha,
            beta=beta,
            smooth=smooth
        )
        nn_model.fit(X, y, sample_weights=sample_weights)
        return nn_model

    def optimize_for_tpr(self, X, y, sample_weights=None, threshold_range=(0.1, 0.9), steps=20):
        """Optimize decision threshold to maximize true positive rate while maintaining reasonable accuracy"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        if sample_weights is not None:
            clf.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            clf.fit(X_train, y_train)
        
        # Get probabilities
        y_proba = clf.predict_proba(X_test)[:, 1]
        
        # Find optimal threshold
        best_threshold = 0.5
        best_tpr = 0
        best_accuracy = 0
        
        for threshold in np.linspace(threshold_range[0], threshold_range[1], steps):
            y_pred = (y_proba >= threshold).astype(int)
            tpr = recall_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            
            # We want to maximize TPR while keeping accuracy above 0.5
            if tpr > best_tpr and accuracy > 0.5:
                best_tpr = tpr
                best_threshold = threshold
                best_accuracy = accuracy
        
        # Set the optimal threshold
        self.optimal_threshold = best_threshold
        
        # Return model and metrics
        metrics = {
            'optimal_threshold': best_threshold,
            'true_positive_rate': best_tpr,
            'accuracy': best_accuracy
        }
        
        return clf, metrics

    def evaluate_model_with_tpr(self, clf, X_test, y_test, sample_weights=None, dataset_name='Test'):
        """Evaluate model with focus on true positive rate and true negative rate"""
        try:
            y_proba = clf.predict_proba(X_test)[:, 1]
            
            # Use optimal threshold if available
            if hasattr(self, 'optimal_threshold'):
                y_pred = (y_proba >= self.optimal_threshold).astype(int)
            else:
                y_pred = clf.predict(X_test)
            
            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            tpr = tp / (tp + fn)  # True Positive Rate
            tnr = tn / (tn + fp)  # True Negative Rate
            
            metrics = {
                'accuracy': accuracy,
                'true_positive_rate': tpr,
                'true_negative_rate': tnr,
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred)
            }
            
            print(f"\n{dataset_name} Set Evaluation:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"True Positive Rate: {metrics['true_positive_rate']:.4f}")
            print(f"True Negative Rate: {metrics['true_negative_rate']:.4f}")
            print("\nConfusion Matrix:")
            print(metrics['confusion_matrix'])
            print("\nClassification Report:")
            print(metrics['classification_report'])
            
            return metrics
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return None

    def train_with_tversky_scoring(self, X, y, sample_weights=None, alpha=0.3, beta=0.7, model_type='rf'):
        """
        Train model with Tversky scoring to optimize for TPR
        model_type: 'rf' for Random Forest, 'logistic' for Logistic Regression
        """
        # Convert to numpy arrays and handle NaN values
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        # Remove rows with NaN values
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y[mask]
        if sample_weights is not None:
            sample_weights = np.array(sample_weights, dtype=float)
            sample_weights = sample_weights[mask]
        
        # Split data
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
            X, y, sample_weights, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        if model_type == 'rf':
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', random_state=42)
            
        if weights_train is not None:
            clf.fit(X_train, y_train, sample_weight=weights_train)
        else:
            clf.fit(X_train, y_train)
        
        # Get probabilities
        y_proba = clf.predict_proba(X_test)[:, 1]
        
        # Optimize threshold using Tversky scorer
        tversky_scorer = TverskyScorer(alpha=alpha, beta=beta)
        thresholds = np.linspace(0.1, 0.9, 20)
        best_threshold = 0.5
        best_score = 0
        
        for threshold in thresholds:
            score = tversky_scorer(y_test, y_proba, threshold)
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        # Store optimal threshold
        self.optimal_threshold = best_threshold
        
        return clf

    def train_with_focal_scoring(self, X, y, sample_weights=None, alpha=0.25, gamma=2.0, model_type='rf'):
        """
        Train model with Focal scoring to handle class imbalance
        model_type: 'rf' for Random Forest, 'logistic' for Logistic Regression
        """
        # Convert to numpy arrays and handle NaN values
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        # Remove rows with NaN values
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y[mask]
        if sample_weights is not None:
            sample_weights = np.array(sample_weights, dtype=float)
            sample_weights = sample_weights[mask]
        
        # Split data
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
            X, y, sample_weights, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        if model_type == 'rf':
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', random_state=42)
            
        if weights_train is not None:
            clf.fit(X_train, y_train, sample_weight=weights_train)
        else:
            clf.fit(X_train, y_train)
        
        # Get probabilities
        y_proba = clf.predict_proba(X_test)[:, 1]
        
        # Optimize threshold using Focal scorer
        focal_scorer = FocalScorer(alpha=alpha, gamma=gamma)
        thresholds = np.linspace(0.1, 0.9, 20)
        best_threshold = 0.5
        best_score = 0
        
        for threshold in thresholds:
            score = focal_scorer(y_test, y_proba, threshold)
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        # Store optimal threshold
        self.optimal_threshold = best_threshold
        
        return clf

    def train_with_dice_scoring(self, X, y, sample_weights=None, smooth=1.0, model_type='rf'):
        """
        Train model with Dice scoring to optimize overlap
        model_type: 'rf' for Random Forest, 'logistic' for Logistic Regression
        """
        # Convert to numpy arrays and handle NaN values
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        # Remove rows with NaN values
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y[mask]
        if sample_weights is not None:
            sample_weights = np.array(sample_weights, dtype=float)
            sample_weights = sample_weights[mask]
        
        # Split data
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
            X, y, sample_weights, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        if model_type == 'rf':
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', random_state=42)
            
        if weights_train is not None:
            clf.fit(X_train, y_train, sample_weight=weights_train)
        else:
            clf.fit(X_train, y_train)
        
        # Get probabilities
        y_proba = clf.predict_proba(X_test)[:, 1]
        
        # Optimize threshold using Dice scorer
        dice_scorer = DiceScorer(smooth=smooth)
        thresholds = np.linspace(0.1, 0.9, 20)
        best_threshold = 0.5
        best_score = 0
        
        for threshold in thresholds:
            score = dice_scorer(y_test, y_proba, threshold)
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        # Store optimal threshold
        self.optimal_threshold = best_threshold
        
        return clf
