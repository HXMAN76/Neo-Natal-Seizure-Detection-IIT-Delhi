import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, BatchNormalization, ReLU, MaxPooling1D, 
    GlobalMaxPooling1D, LSTM, Dropout, Dense, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("🧠 Neonatal Seizure Detection: CNN+LSTM Implementation")
print("="*60)

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_preprocess_data(csv_file='/content/drive/MyDrive/complete_preprocessed_dataset_shuffled_with_ica.csv'):
    """
    Load preprocessed EEG data and prepare for CNN+LSTM model
    
    Assumes CSV structure:
    - Columns 0-9727: EEG features arranged as [t1_ch1, t1_ch2, ..., t1_ch19, t2_ch1, ..., t512_ch19]
    - Column 9728: Labels (0=non-seizure, 1=seizure) 
    - Each row represents one 10-second EEG window with 512 time points × 19 channels
    """
    print("📁 Loading preprocessed dataset...")
    
    # Load data
    data = pd.read_csv(csv_file)
    print(f"   Original shape: {data.shape}")
    
    # Separate features and labels
    if 'label' in data.columns:
        X = data.drop('label', axis=1).values
        y = data['label'].values
    else:
        # Assume last column is labels
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
    
    print(f"   Features shape: {X.shape}")
    print(f"   Labels shape: {y.shape}")
    
    # Check class distribution
    class_counts = Counter(y)
    print(f"   Class distribution: {dict(class_counts)}")
    if len(y) > 0:
        seizure_percentage = class_counts[1]/len(y)*100
        print(f"   Seizure percentage: {seizure_percentage:.2f}%")
        
        # Warning for small dataset
        if len(y) < 1000:
            print(f"   ⚠️  WARNING: Small dataset ({len(y)} samples). Consider:")
            print(f"      - Data augmentation techniques")
            print(f"      - Transfer learning approaches") 
            print(f"      - Cross-validation instead of train/test split")
        
        if class_counts[1] < 50:
            print(f"   ⚠️  WARNING: Very few seizure samples ({class_counts[1]}). Consider:")
            print(f"      - SMOTE or other oversampling techniques")
            print(f"      - Adjusting class weights in model compilation")
    else:
        print("   ERROR: No valid samples remaining after cleaning!")
        return None, None
    
    return X, y

def reshape_for_cnn_lstm(X, n_channels=19, n_timesteps=512):
    """
    Reshape flattened EEG data for CNN+LSTM input
    
    Current: (batch, 9728) - flattened as [t1_ch1, t1_ch2, ..., t1_ch19, t2_ch1, ..., t512_ch19]
    Target: (batch, n_timesteps, n_channels) - for Conv1D processing
    
    Your data structure: 512 time points × 19 channels stacked at each time point
    """
    print(f"🔄 Reshaping data from {X.shape} to CNN+LSTM format...")
    
    # Reshape from (batch, 9728) to (batch, 512, 19)
    # Your data is arranged as: [t1_ch1, t1_ch2, ..., t1_ch19, t2_ch1, ..., t512_ch19]
    X_reshaped = X.reshape(-1, n_timesteps, n_channels)
    
    print(f"   Reshaped to: {X_reshaped.shape}")
    print(f"   ✅ 10-second windows: {n_timesteps} time points × {n_channels} channels")
    
    return X_reshaped

# ============================================================================
# 2. MODEL ARCHITECTURE
# ============================================================================

def create_cnn_lstm_model(input_shape=(512, 19), num_classes=2):
    """
    Create the CNN+LSTM model as per your architecture specification
    
    Architecture matches your detailed specification:
    - 3 Conv1D layers with increasing filters (64, 128, 256)
    - BatchNormalization and ReLU after each conv layer
    - MaxPooling after first two conv layers
    - GlobalMaxPooling before LSTM
    - Single LSTM layer (64 units)
    - Dense layers for classification
    """
    print("🏗️ Building CNN+LSTM Architecture...")
    
    # Update architecture comments to reflect 512 time points
    model = Sequential([
        # Input Layer
        Input(shape=input_shape, name='eeg_input'),
        
        # Conv1D Layer 1 - Extract local temporal patterns (512 → 512)
        Conv1D(filters=64, kernel_size=7, padding='same', name='conv1d_1'),
        BatchNormalization(momentum=0.99, epsilon=1e-3, name='batch_norm_1'),
        ReLU(name='relu_1'),
        MaxPooling1D(pool_size=2, strides=2, name='maxpool_1'),  # 512 → 256
        
        # Conv1D Layer 2 - Extract higher-level patterns (256 → 256)
        Conv1D(filters=128, kernel_size=5, padding='same', name='conv1d_2'),
        BatchNormalization(momentum=0.99, epsilon=1e-3, name='batch_norm_2'),
        ReLU(name='relu_2'),
        MaxPooling1D(pool_size=2, strides=2, name='maxpool_2'),  # 256 → 128
        
        # Conv1D Layer 3 - Extract complex patterns (128 → 128)
        Conv1D(filters=256, kernel_size=3, padding='same', name='conv1d_3'),
        BatchNormalization(momentum=0.99, epsilon=1e-3, name='batch_norm_3'),
        ReLU(name='relu_3'),
        
        # Global Max Pooling - Aggregate temporal info (128 → 1)
        GlobalMaxPooling1D(name='global_maxpool'),
        
        # LSTM Layer - Model long-term dependencies
        # Note: Reshaping needed since LSTM expects 3D input (batch, timesteps, features)
        tf.keras.layers.Reshape((1, 256), name='reshape_for_lstm'),
        LSTM(units=64, return_sequences=False, name='lstm'),
        
        # Classification Layers
        Dropout(rate=0.5, name='dropout'),
        Dense(units=128, activation='relu', name='dense_1'),
        Dense(units=64, activation='relu', name='dense_2'),
        Dense(units=num_classes, activation='softmax', name='output')
    ])
    
    # Compile model with more robust metrics
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']  # Simplified metrics to avoid shape issues
    )
    
    return model

def print_model_summary(model):
    """Print detailed model architecture summary"""
    print("\n📊 Model Architecture Summary:")
    print("-" * 80)
    model.summary()
    
    total_params = model.count_params()
    print(f"\n📈 Model Statistics:")
    print(f"   Total Parameters: {total_params:,}")
    print(f"   Trainable Parameters: {total_params:,}")
    
    # Calculate model size (approximate)
    model_size_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
    print(f"   Estimated Model Size: {model_size_mb:.2f} MB")

# ============================================================================
# 3. TRAINING UTILITIES
# ============================================================================

def create_callbacks(model_name='best_seizure_model'):
    """Create training callbacks for monitoring and model saving"""
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=f'{model_name}.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    ]
    return callbacks

def plot_training_history(history):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    
    # Precision
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Training Precision')
        axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
    
    # Recall
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='Training Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# 4. EVALUATION UTILITIES
# ============================================================================

def evaluate_model(model, X_test, y_test, class_names=['Non-Seizure', 'Seizure']):
    """Comprehensive model evaluation"""
    print("\n🔍 Model Evaluation Results:")
    print("=" * 50)
    
    # Predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Classification Report
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # ROC AUC Score
    if len(np.unique(y_test)) > 1:
        auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
        print(f"\n🎯 ROC AUC Score: {auc_score:.4f}")
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    return {
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'auc_score': auc_score if len(np.unique(y_test)) > 1 else None
    }

# ============================================================================
# 5. MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Main training pipeline"""
    print("🚀 Starting Neonatal Seizure Detection Training Pipeline")
    print("=" * 70)
    
    # Load and preprocess data
    X, y = load_and_preprocess_data()
    
    # Check if data loading failed
    if X is None or y is None:
        print("❌ Data loading failed. Exiting...")
        return None, None, None
    
    # Final validation before proceeding
    print(f"\n🔍 Final validation before training:")
    print(f"   X shape: {X.shape}, dtype: {X.dtype}")
    print(f"   y shape: {y.shape}, dtype: {y.dtype}")
    print(f"   NaN in X: {np.isnan(X).sum()}")
    print(f"   NaN in y: {np.isnan(y).sum()}")
    print(f"   Inf in X: {np.isinf(X).sum()}")
    print(f"   y unique values: {np.unique(y)}")
    
    # If still problems, exit
    if np.isnan(X).any() or np.isnan(y).any():
        print("❌ Still have NaN values after cleaning. Cannot proceed.")
        return None, None, None
    
    # Reshape for CNN+LSTM
    X_reshaped = reshape_for_cnn_lstm(X)
    
    # Train-test split (stratified to preserve class distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print(f"\n📊 Data Split Summary:")
    print(f"   Training samples: {X_train.shape[0]} ({Counter(y_train)})")
    print(f"   Test samples: {X_test.shape[0]} ({Counter(y_test)})")
    print(f"   Input shape per sample: {X_train.shape[1:]} (512 time points × 19 channels)")
    print(f"   Total 10-second EEG windows: {X_train.shape[0] + X_test.shape[0]}")
    
    # Create model
    model = create_cnn_lstm_model(input_shape=X_train.shape[1:])
    print_model_summary(model)
    
    # Create callbacks
    callbacks = create_callbacks()
    
    # Train model with validation_split instead of separate validation data
    print("\n🏃 Starting model training...")
    
    # Use validation_split to ensure consistent batch handling
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,  # Use 20% of training data for validation
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    results = evaluate_model(model, X_test, y_test)
    
    # Save final model
    model.save('final_seizure_detection_model.h5')
    print(f"\n💾 Model saved as 'final_seizure_detection_model.h5'")
    
    return model, history, results

# ============================================================================
# 6. USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Run the complete pipeline
    model, history, results = main()
    
    print("\n🎉 Training Complete!")
    print(f"Final AUC Score: {results['auc_score']:.4f}")
    
    # Optional: Patient-independent validation using cross-validation
    print("\n💡 For clinical validation, consider implementing:")
    print("   - Leave-one-patient-out cross-validation")
    print("   - Stratified K-fold validation")
    print("   - Temporal validation (early vs late recordings)")