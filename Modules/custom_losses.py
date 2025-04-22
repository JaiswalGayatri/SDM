import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.losses import binary_crossentropy

class FocalLoss:
    def __init__(self, alpha=0.25, gamma=2.0):
        self.alpha = alpha
        self.gamma = gamma
    
    def __call__(self, y_true, y_pred):
        # Ensure numerical stability
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Calculate focal loss
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        modulating_factor = tf.pow(1.0 - p_t, self.gamma)
        
        return alpha_factor * modulating_factor * cross_entropy

class DiceLoss:
    def __init__(self, smooth=1.0):
        self.smooth = smooth
    
    def __call__(self, y_true, y_pred):
        # Ensure numerical stability
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Calculate dice loss
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1. - dice

class TverskyLoss:
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        """
        Tversky Loss implementation
        alpha: controls penalty for false positives
        beta: controls penalty for false negatives
        smooth: smoothing factor to avoid division by zero
        """
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def __call__(self, y_true, y_pred):
        # Ensure numerical stability
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Calculate true positives, false positives, and false negatives
        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))
        
        # Calculate Tversky index
        numerator = tp + self.smooth
        denominator = tp + self.alpha * fp + self.beta * fn + self.smooth
        
        # Return Tversky loss
        return 1 - (numerator / denominator)

class CustomNeuralNetwork:
    def __init__(self, loss_fn='focal', alpha=0.25, gamma=2.0, smooth=1.0, beta=0.7):
        self.loss_fn = loss_fn
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.beta = beta
        self.model = None
        
    def build_model(self, input_shape):
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        if self.loss_fn == 'focal':
            loss = FocalLoss(alpha=self.alpha, gamma=self.gamma)
        elif self.loss_fn == 'dice':
            loss = DiceLoss(smooth=self.smooth)
        elif self.loss_fn == 'tversky':
            loss = TverskyLoss(alpha=self.alpha, beta=self.beta, smooth=self.smooth)
        else:
            loss = 'binary_crossentropy'
            
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=['accuracy', tf.keras.metrics.TruePositives()]
        )
        
        self.model = model
        return model
    
    def fit(self, X, y, sample_weights=None, epochs=50, batch_size=32, validation_split=0.2):
        if self.model is None:
            self.build_model(X.shape[1])
            
        return self.model.fit(
            X, y,
            sample_weight=sample_weights,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
    
    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype(int)
    
    def predict_proba(self, X):
        return self.model.predict(X)

class TverskyScorer:
    def __init__(self, alpha=0.3, beta=0.7):
        """
        Tversky scorer for scikit-learn models
        alpha: penalty for false positives
        beta: penalty for false negatives
        """
        self.alpha = alpha
        self.beta = beta
    
    def __call__(self, y_true, y_pred_proba, threshold=0.5):
        # Convert probabilities to binary predictions
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate true positives, false positives, and false negatives
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        # Calculate Tversky score
        numerator = tp
        denominator = tp + self.alpha * fp + self.beta * fn
        
        # Avoid division by zero
        if denominator == 0:
            return 0.0
            
        return numerator / denominator

class FocalScorer:
    def __init__(self, alpha=0.25, gamma=2.0):
        """
        Focal scorer for scikit-learn models
        alpha: class weight
        gamma: focusing parameter
        """
        self.alpha = alpha
        self.gamma = gamma
    
    def __call__(self, y_true, y_pred_proba, threshold=0.5):
        # Convert probabilities to binary predictions
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate focal score
        p_t = np.where(y_true == 1, y_pred_proba, 1 - y_pred_proba)
        alpha_t = np.where(y_true == 1, self.alpha, 1 - self.alpha)
        modulating_factor = np.power(1.0 - p_t, self.gamma)
        
        # Calculate weighted accuracy
        weighted_correct = np.sum(alpha_t * modulating_factor * (y_true == y_pred))
        total_weight = np.sum(alpha_t * modulating_factor)
        
        if total_weight == 0:
            return 0.0
            
        return weighted_correct / total_weight

class DiceScorer:
    def __init__(self, smooth=1.0):
        """
        Dice scorer for scikit-learn models
        smooth: smoothing factor
        """
        self.smooth = smooth
    
    def __call__(self, y_true, y_pred_proba, threshold=0.5):
        # Convert probabilities to binary predictions
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate intersection and union
        intersection = np.sum((y_true == 1) & (y_pred == 1))
        union = np.sum(y_true == 1) + np.sum(y_pred == 1)
        
        # Calculate Dice score
        return (2. * intersection + self.smooth) / (union + self.smooth)

def optimize_threshold_for_tpr(y_true, y_pred_proba, min_accuracy=0.5):
    """
    Find the optimal threshold that maximizes TPR while maintaining minimum accuracy
    """
    thresholds = np.linspace(0.1, 0.9, 20)
    best_threshold = 0.5
    best_tpr = 0
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        tpr = recall_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        
        if tpr > best_tpr and accuracy >= min_accuracy:
            best_tpr = tpr
            best_threshold = threshold
    
    return best_threshold, best_tpr 