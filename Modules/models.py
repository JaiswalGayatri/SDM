# Import necessary libraries for data manipulation, machine learning, and geospatial analysis
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

# Define the environmental feature columns used for species distribution modeling
# These include bioclimatic variables, soil properties, and elevation
feature_cols = [
    'annual_mean_temperature', 'mean_diurnal_range', 'isothermality',
    'temperature_seasonality', 'max_temperature_warmest_month', 'min_temperatur_coldest_month',
    'temperature_annual_range', 'mean_temperature_wettest_quarter', 'mean_temperature_driest_quarter',
    'mean_temperature_warmest_quarter', 'mean_temperature_coldest_quarter', 'annual_precipitation',
    'precipitation_wettest_month', 'precipitation_driest_month', 'precipitation_seasonality',
    'precipitation_wettest_quarter', 'precipitation_driest_quarter', 'precipitation_warmest_quarter',
    'precipitation_coldest_quarter', 'aridity_index', 'topsoil_ph', 'subsoil_ph', 'topsoil_texture',
    'subsoil_texture', 'elevation'
]

class Models:
    """
    A comprehensive class for species distribution modeling with various machine learning algorithms
    and advanced weighting schemes to handle class imbalance and spatial bias.
    """
    
    def __init__(self):
        """Initialize the Models class"""
        return 

    def load_data(self, presence_path='data/presence.csv', absence_path='data/pseudo_absence.csv'):
        """
        Load and preprocess species presence/absence data with two types of sample weighting:
        1. Reliability-based weights for data quality
        2. Bias-correction weights based on ecoregion sampling density
        
        Parameters:
        -----------
        presence_path : str
            Path to CSV file containing species presence records
        absence_path : str  
            Path to CSV file containing pseudo-absence records
            
        Returns:
        --------
        tuple
            X (features), y (labels), coords (coordinates), feature_cols (column names),
            reliability_weights, bias_weights
        """
        
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
        # For presence samples, set weight = 1 (assume high reliability)
        reliability_presence = np.ones(len(presence_df))
        
        # For absence samples, use the 'reliability' column if available
        if 'reliability' in absence_df.columns:
            absence_df['reliability'] = absence_df['reliability'].fillna(1)
        
        # Extract reliability values and normalize them to [0,1] range
        reliability = absence_df['reliability'].values
        min_rel = np.min(reliability)
        max_rel = np.max(reliability)
        if max_rel != min_rel:
            reliability_absence = (reliability - min_rel) / (max_rel - min_rel)
        else:
            reliability_absence = np.ones(len(reliability))
        
        # Apply a mild power transformation to reduce extremes in reliability weights
        reliability_absence = np.array([w**(0.1) for w in reliability_absence])
        
        # Combine reliability weights for presence and absence data
        reliability_weights = np.hstack([reliability_presence, reliability_absence])
        
        # ------------------------------------
        # 2. Bias-correction Weights
        # ------------------------------------
        # Read eco-region counts file to understand sampling density per ecoregion
        # This helps correct for geographic sampling bias
        counts_file = "outputs/testing_SDM_out/species_ecoregion_count_1.csv"
        if os.path.exists(counts_file):
            region_counts_df = pd.read_csv(counts_file)
            # print('region counts df length', len(region_counts_df))
            
            # Compute raw weight: inverse relationship with count (fewer samples = higher weight)
            region_counts_df['raw_weight'] = 1 / (region_counts_df['count'] + 1)
            
            # Normalize raw weights to a subtle range [0.5, 1.5] to avoid extreme values
            min_w = region_counts_df['raw_weight'].min()
            max_w = region_counts_df['raw_weight'].max()
            region_counts_df['eco_weight'] = 0.5 + region_counts_df['raw_weight'] 
            
            # Create mapping dictionary: eco_region -> eco_weight
            eco_weight_dict = region_counts_df.set_index('ecoregion')['eco_weight'].to_dict()
            # print("Eco-region weight mapping:", eco_weight_dict)
        else:
            print(f"Warning: {counts_file} not found. Defaulting eco weights to 1.")
            eco_weight_dict = {}
        
        # To assign bias weights, we need to determine which eco-region each point falls into
        # Since CSV files don't contain eco-region info, we perform spatial join with polygons
        ecoregion_folder = "data/eco_regions_polygon"
        
        # Convert lat/lon coordinates to Point geometries for spatial operations
        presence_df["geometry"] = presence_df.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)
        absence_df["geometry"] = absence_df.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)
        
        # Create GeoDataFrames with proper coordinate reference system (WGS84)
        presence_gdf = gpd.GeoDataFrame(presence_df, geometry="geometry", crs="EPSG:4326")
        absence_gdf = gpd.GeoDataFrame(absence_df, geometry="geometry", crs="EPSG:4326")
        
        # Combine presence and absence data while preserving original ordering
        combined_gdf = pd.concat([presence_gdf, absence_gdf], ignore_index=True)
        # print('len',len(presence_df))
        # print('len',len(absence_df))
        # print('len',len(combined_gdf))
        
        # Function to load eco-region polygons from WKT (Well-Known Text) files
        def load_ecoregions(folder):
            """Load ecoregion polygons from WKT files in the specified folder"""
            ecoregions = []
            for file in os.listdir(folder):
                if file.endswith(".wkt"):
                    with open(os.path.join(folder, file), "r") as f:
                        wkt_text = f.read().strip()
                        poly = wkt.loads(wkt_text)  # Parse WKT to geometry
                        ecoregions.append({"ecoregion": file.replace(".wkt", ""), "geometry": poly})
            return gpd.GeoDataFrame(ecoregions, geometry="geometry", crs="EPSG:4326")
        
        # Load eco-region polygons from WKT files
        ecoregion_gdf = load_ecoregions(ecoregion_folder)
        
        # Perform spatial join to assign each point to its corresponding eco-region
        # Uses "within" predicate to find which polygon contains each point
        combined_with_ecoregion = gpd.sjoin(combined_gdf, ecoregion_gdf, how="left", predicate="within")
        
        # Define function to retrieve bias weight for a given eco-region
        def get_bias_weight(eco):
            """Return bias weight for ecoregion, default to 1 if not found or NaN"""
            if pd.isna(eco):
                return 1
            else:
                return eco_weight_dict.get(eco, 1)
        
        # Apply the mapping to get bias weights for each data point
        bias_weights = combined_with_ecoregion['ecoregion'].apply(get_bias_weight).values
        
        # Save bias weights with coordinates to CSV for inspection and debugging
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
        # Extract environmental features from the original CSV data
        presence_features = presence_df[feature_cols].values
        absence_features = absence_df[feature_cols].values
        
        # Combine features and create binary labels (1=presence, 0=absence)
        X = np.vstack([presence_features, absence_features])
        y = np.hstack([np.ones(len(presence_features)), np.zeros(len(absence_features))])
        
        # Extract coordinates from the combined GeoDataFrame (maintaining order)
        coords = np.column_stack((combined_with_ecoregion.geometry.x, combined_with_ecoregion.geometry.y))
        
        # ------------------------------------
        # Shuffle the data along with both sets of weights
        # ------------------------------------
        # Randomly shuffle all arrays together to ensure proper randomization
        X, y, coords, reliability_weights, bias_weights = shuffle(
            X, y, coords, reliability_weights, bias_weights, random_state=42
        )
        
        # Return all processed data components
        return X, y, coords, feature_cols, reliability_weights, bias_weights


    def RandomForest(self, X, y, sample_weights=None):
        """
        Train a Random Forest classifier with optional sample weighting.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like  
            Target labels
        sample_weights : array-like, optional
            Sample weights for training
            
        Returns:
        --------
        tuple
            Trained classifier, test features, test labels, predictions, probabilities
        """
        # Split data and track indices to properly subset sample weights
        indices = np.arange(len(y))
        X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
            X, y, indices, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize Random Forest classifier with 100 trees
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Train with sample weights if provided
        if sample_weights is not None:
            # Ensure sample_weights is a 1D array
            sample_weights = np.ravel(sample_weights)
            # Subset the weights using the training indices
            weights_train = sample_weights[indices_train]
            clf.fit(X_train, y_train, sample_weight=weights_train)
        else:
            clf.fit(X_train, y_train)
        
        # Generate predictions and class probabilities
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]  # Probability of positive class
        
        return clf, X_test, y_test, y_pred, y_proba

    def logistic_regression_L2(self, X, y):
        """
        Train a Logistic Regression model with L2 regularization.
        Includes data cleaning to handle infinite/NaN values.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target labels
            
        Returns:
        --------
        tuple
            Trained classifier, test features, test labels, predictions, probabilities
        """
        # Convert to numpy arrays and filter out rows with infinite/NaN values
        X = np.array(X, dtype=float)
        mask = np.isfinite(X).all(axis=1)  # Keep only finite values
        X, y = X[mask], y[mask]
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Additional filtering for training and test sets
        train_mask = np.isfinite(X_train).all(axis=1)
        test_mask = np.isfinite(X_test).all(axis=1)
        X_train, y_train = X_train[train_mask], y_train[train_mask]
        X_test, y_test = X_test[test_mask], y_test[test_mask]
        
        # Initialize and train Logistic Regression with L2 penalty
        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', random_state=42)
        clf.fit(X_train, y_train)
        
        # Generate predictions and probabilities
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        
        return clf, X_test, y_test, y_pred, y_proba

    def train_and_evaluate_model_logistic_weighted(self, X, y, sample_weights=None):
        """
        Train a weighted Logistic Regression model with comprehensive data preprocessing.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target labels
        sample_weights : array-like, optional
            Sample weights for training
            
        Returns:
        --------
        tuple
            Trained classifier, test features, test labels, predictions, probabilities
        """
        # Convert to numpy arrays and ensure proper data types
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        # Remove rows with NaN values to ensure clean training data
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y[mask]
        if sample_weights is not None:
            sample_weights = np.array(sample_weights, dtype=float)
            sample_weights = sample_weights[mask]
        
        # Split data while preserving sample weights alignment
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
            X, y, sample_weights, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize Logistic Regression model
        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', random_state=42)
        
        # Train with or without sample weights
        if weights_train is not None:
            clf.fit(X_train, y_train, sample_weight=weights_train)
        else:
            clf.fit(X_train, y_train)
        
        # Generate predictions and class probabilities
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        
        return clf, X_test, y_test, y_pred, y_proba

    def evaluate_model(self, clf: RandomForestClassifier, X_test, y_test, sample_weights=None, dataset_name='Test'):
        """
        Comprehensive model evaluation with multiple metrics.
        
        Parameters:
        -----------
        clf : sklearn classifier
            Trained classifier to evaluate
        X_test : array-like
            Test features
        y_test : array-like
            Test labels
        sample_weights : array-like, optional
            Sample weights (currently unused but kept for compatibility)
        dataset_name : str
            Name for the dataset being evaluated (for display purposes)
            
        Returns:
        --------
        dict or None
            Dictionary containing evaluation metrics, or None if error occurs
        """
        try:
            # Generate predictions using the trained classifier
            y_pred = clf.predict(X_test)
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None

        # Calculate comprehensive evaluation metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        # Display evaluation results
        print(f"\n{dataset_name} Set Evaluation:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        print("\nClassification Report:")
        print(metrics['classification_report'])
        
        return metrics

    def train_with_focal_loss(self, X, y, sample_weights=None, alpha=0.25, gamma=2.0):
        """
        Train a neural network model using focal loss to improve handling of class imbalance.
        Focal loss down-weights easy examples and focuses on hard examples.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target labels
        sample_weights : array-like, optional
            Sample weights for training
        alpha : float, default=0.25
            Weighting factor for rare class
        gamma : float, default=2.0
            Focusing parameter (higher gamma = more focus on hard examples)
            
        Returns:
        --------
        CustomNeuralNetwork
            Trained neural network model with focal loss
        """
        nn_model = CustomNeuralNetwork(loss_fn='focal', alpha=alpha, gamma=gamma)
        nn_model.fit(X, y, sample_weights=sample_weights)
        return nn_model

    def train_with_dice_loss(self, X, y, sample_weights=None, smooth=1.0):
        """
        Train a neural network model using dice loss to focus on true positives.
        Dice loss measures overlap between predicted and actual positive regions.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target labels
        sample_weights : array-like, optional
            Sample weights for training
        smooth : float, default=1.0
            Smoothing factor to avoid division by zero
            
        Returns:
        --------
        CustomNeuralNetwork
            Trained neural network model with dice loss
        """
        nn_model = CustomNeuralNetwork(loss_fn='dice', smooth=smooth)
        nn_model.fit(X, y, sample_weights=sample_weights)
        return nn_model

    def train_with_tversky_loss(self, X, y, sample_weights=None, alpha=0.3, beta=0.7, smooth=1.0):
        """
        Train a neural network model using Tversky loss to handle class imbalance 
        with explicit control over false positives and false negatives.
        
        Tversky loss is a generalization of Dice loss that allows asymmetric 
        weighting of false positives and false negatives.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target labels
        sample_weights : array-like, optional
            Sample weights for training
        alpha : float, default=0.3
            Penalty weight for false positives (lower = less penalty)
        beta : float, default=0.7
            Penalty weight for false negatives (higher = more penalty)
        smooth : float, default=1.0
            Smoothing factor to avoid division by zero
            
        Returns:
        --------
        CustomNeuralNetwork
            Trained neural network model with Tversky loss
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
        """
        Optimize decision threshold to maximize true positive rate while maintaining 
        reasonable accuracy. This is particularly useful for species distribution 
        modeling where detecting presence is more important than avoiding false positives.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target labels
        sample_weights : array-like, optional
            Sample weights for training
        threshold_range : tuple, default=(0.1, 0.9)
            Range of thresholds to test
        steps : int, default=20
            Number of threshold values to test
            
        Returns:
        --------
        tuple
            Trained classifier and dictionary of optimization metrics
        """
        # Split data for training and threshold optimization
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        if sample_weights is not None:
            clf.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            clf.fit(X_train, y_train)
        
        # Get class probabilities for threshold optimization
        y_proba = clf.predict_proba(X_test)[:, 1]
        
        # Initialize optimization variables
        best_threshold = 0.5
        best_tpr = 0
        best_accuracy = 0
        
        # Test different thresholds to find optimal balance
        for threshold in np.linspace(threshold_range[0], threshold_range[1], steps):
            y_pred = (y_proba >= threshold).astype(int)
            tpr = recall_score(y_test, y_pred)  # True Positive Rate
            accuracy = accuracy_score(y_test, y_pred)
            
            # Update best threshold if TPR improves and accuracy remains acceptable
            if tpr > best_tpr and accuracy > 0.5:
                best_tpr = tpr
                best_threshold = threshold
                best_accuracy = accuracy
        
        # Store optimal threshold for later use
        self.optimal_threshold = best_threshold
        
        # Return model and optimization results
        metrics = {
            'optimal_threshold': best_threshold,
            'true_positive_rate': best_tpr,
            'accuracy': best_accuracy
        }
        
        return clf, metrics

    def evaluate_model_with_tpr(self, clf, X_test, y_test, sample_weights=None, dataset_name='Test'):
        """
        Evaluate model with focus on true positive rate and true negative rate.
        Uses optimal threshold if available from previous optimization.
        
        Parameters:
        -----------
        clf : sklearn classifier
            Trained classifier to evaluate
        X_test : array-like
            Test features
        y_test : array-like
            Test labels
        sample_weights : array-like, optional
            Sample weights (currently unused)
        dataset_name : str
            Name for the dataset being evaluated
            
        Returns:
        --------
        dict or None
            Dictionary containing TPR-focused evaluation metrics
        """
        try:
            # Get class probabilities
            y_proba = clf.predict_proba(X_test)[:, 1]
            
            # Use optimal threshold if available, otherwise use default prediction
            if hasattr(self, 'optimal_threshold'):
                y_pred = (y_proba >= self.optimal_threshold).astype(int)
            else:
                y_pred = clf.predict(X_test)
            
            # Calculate confusion matrix components
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            
            # Calculate key metrics for species distribution modeling
            accuracy = accuracy_score(y_test, y_pred)
            tpr = tp / (tp + fn)  # True Positive Rate (Sensitivity)
            tnr = tn / (tn + fp)  # True Negative Rate (Specificity)
            
            # Compile comprehensive metrics
            metrics = {
                'accuracy': accuracy,
                'true_positive_rate': tpr,
                'true_negative_rate': tnr,
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred)
            }
            
            # Display detailed evaluation results
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
        Train model with Tversky scoring to optimize decision threshold for TPR.
        Combines traditional ML models with advanced threshold optimization.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target labels
        sample_weights : array-like, optional
            Sample weights for training
        alpha : float, default=0.3
            Tversky alpha parameter (false positive penalty)
        beta : float, default=0.7
            Tversky beta parameter (false negative penalty)
        model_type : str, default='rf'
            Type of model to train ('rf' for Random Forest, 'logistic' for Logistic Regression)
            
        Returns:
        --------
        sklearn classifier
            Trained classifier with optimized threshold stored
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
        
        # Split data while maintaining weight alignment
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
            X, y, sample_weights, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize classifier based on model type
        if model_type == 'rf':
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', random_state=42)
            
        # Train model with optional sample weights
        if weights_train is not None:
            clf.fit(X_train, y_train, sample_weight=weights_train)
        else:
            clf.fit(X_train, y_train)
        
        # Get class probabilities for threshold optimization
        y_proba = clf.predict_proba(X_test)[:, 1]
        
        # Optimize threshold using Tversky scorer
        tversky_scorer = TverskyScorer(alpha=alpha, beta=beta)
        thresholds = np.linspace(0.1, 0.9, 20)
        best_threshold = 0.5
        best_score = 0
        
        # Find threshold that maximizes Tversky score
        for threshold in thresholds:
            score = tversky_scorer(y_test, y_proba, threshold)
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        # Store optimal threshold for later use
        self.optimal_threshold = best_threshold
        
        return clf

    def train_with_focal_scoring(self, X, y, sample_weights=None, alpha=0.25, gamma=2.0, model_type='rf'):
        """
        Train model with Focal scoring to handle class imbalance through threshold optimization.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target labels
        sample_weights : array-like, optional
            Sample weights for training
        alpha : float, default=0.25
            Focal loss alpha parameter (class weighting)
        gamma : float, default=2.0
            Focal loss gamma parameter (focusing parameter)
        model_type : str, default='rf'
            Type of model to train ('rf' for Random Forest, 'logistic' for Logistic Regression)
            
        Returns:
        --------
        sklearn classifier
            Trained classifier with optimized threshold stored
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
        
        # Split data while maintaining weight alignment
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
            X, y, sample_weights, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize classifier based on model type
        if model_type == 'rf':
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', random_state=42)
            
        # Train model with optional sample weights
        if weights_train is not None:
            clf.fit(X_train, y_train, sample_weight=weights_train)
        else:
            clf.fit(X_train, y_train)
        
        # Get class probabilities for threshold optimization
        y_proba = clf.predict_proba(X_test)[:, 1]
        
        # Optimize threshold using Focal scorer
        focal_scorer = FocalScorer(alpha=alpha, gamma=gamma)
        thresholds = np.linspace(0.1, 0.9, 20)
        best_threshold = 0.5
        best_score = 0
        
        # Find threshold that maximizes Focal score
        for threshold in thresholds:
            score = focal_scorer(y_test, y_proba, threshold)
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        # Store optimal threshold for later use
        self.optimal_threshold = best_threshold
        
        return clf

    def train_with_dice_scoring(self, X, y, sample_weights=None, smooth=1.0, model_type='rf'):
        """
        Train model with Dice scoring to optimize overlap between predicted and actual positives.
        Dice score is particularly useful for imbalanced datasets in species distribution modeling.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target labels
        sample_weights : array-like, optional
            Sample weights for training
        smooth : float, default=1.0
            Smoothing factor to avoid division by zero
        model_type : str, default='rf'
            Type of model to train ('rf' for Random Forest, 'logistic' for Logistic Regression)
            
        Returns:
        --------
        sklearn classifier
            Trained classifier with optimized threshold stored
        """
        # Convert to numpy arrays and handle NaN values
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        # Remove rows with NaN values to ensure clean training data
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y[mask]
        if sample_weights is not None:
            sample_weights = np.array(sample_weights, dtype=float)
            sample_weights = sample_weights[mask]
        
        # Split data while maintaining weight alignment
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
            X, y, sample_weights, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize classifier based on model type
        if model_type == 'rf':
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', random_state=42)
            
        # Train model with optional sample weights
        if weights_train is not None:
            clf.fit(X_train, y_train, sample_weight=weights_train)
        else:
            clf.fit(X_train, y_train)
        
        # Get class probabilities for threshold optimization
        y_proba = clf.predict_proba(X_test)[:, 1]
        
        # Optimize threshold using Dice scorer
        dice_scorer = DiceScorer(smooth=smooth)
        thresholds = np.linspace(0.1, 0.9, 20)
        best_threshold = 0.5
        best_score = 0
        
        # Find threshold that maximizes Dice score
        for threshold in thresholds:
            score = dice_scorer(y_test, y_proba, threshold)
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        # Store optimal threshold for later use
        self.optimal_threshold = best_threshold
        
        return clf