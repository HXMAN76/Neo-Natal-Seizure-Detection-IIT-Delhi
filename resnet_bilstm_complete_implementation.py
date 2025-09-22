
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv1D, BatchNormalization, ReLU, MaxPooling1D, 
    GlobalAveragePooling1D, LSTM, Bidirectional, Dropout, Dense, Input,
    Add, Lambda, Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("🏗️ Neonatal Seizure Detection: ResNet+BiLSTM Implementation")
print("="*70)

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv1D, BatchNormalization, ReLU, MaxPooling1D, 
    GlobalAveragePooling1D, LSTM, Bidirectional, Dropout, Dense, Input,
    Add, Lambda, Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("🏗️ Neonatal Seizure Detection: ResNet+BiLSTM Implementation")
print("="*70)

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_preprocess_data(csv_file='complete_preprocessed_dataset_shuffled_with_ica.csv'):
    """
    Load preprocessed EEG data and prepare for ResNet+BiLSTM model
    
    Assumes CSV structure:
    - Columns 0-9727: EEG features arranged as [t1_ch1, t1_ch2, ..., t1_ch19, t2_ch1, ..., t320_ch19]
    - Column 9728: Labels (0=non-seizure, 1=seizure) 
    - Each row represents one 10-second EEG window with 320 time points × 19 channels
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

def reshape_for_resnet_bilstm(X, n_channels=19, n_timesteps=320):
    """
    Reshape flattened EEG data for ResNet+BiLSTM input
    
    Current: (batch, 9728) - flattened as [t1_ch1, t1_ch2, ..., t1_ch19, t2_ch1, ..., t320_ch19]
    Target: (batch, n_timesteps, n_channels) - for Conv1D processing
    """
    print(f"🔄 Reshaping data from {X.shape} to ResNet+BiLSTM format...")
    
    # If needed, adjust the shape to match expected input dimensions
    expected_flat_size = n_timesteps * n_channels
    if X.shape[1] != expected_flat_size:
        print(f"   ⚠️ WARNING: Input shape mismatch. Expected {expected_flat_size} features, got {X.shape[1]}")
        print(f"   ⚠️ Attempting to reshape - may truncate or pad data")
        
        # If larger, truncate; if smaller, pad with zeros
        if X.shape[1] > expected_flat_size:
            X = X[:, :expected_flat_size]
        else:
            pad_width = expected_flat_size - X.shape[1]
            X = np.pad(X, ((0, 0), (0, pad_width)), 'constant')
    
    # Reshape from flat to (batch, timesteps, channels)
    X_reshaped = X.reshape(-1, n_timesteps, n_channels)
    
    print(f"   Reshaped to: {X_reshaped.shape}")
    print(f"   ✅ 10-second windows: {n_timesteps} time points × {n_channels} channels")
    
    return X_reshaped

# ============================================================================
# 2. RESIDUAL BLOCK IMPLEMENTATION
# ============================================================================

def residual_block_1d(inputs, filters, kernel_size=3, stride=1, name='resblock'):
    """
    1D Residual Block for EEG signal processing
    Implements the basic residual block with skip connections
    """
    # First convolutional layer
    x = Conv1D(filters=filters, kernel_size=kernel_size, strides=stride, 
               padding='same', use_bias=False, name=f'{name}_conv1')(inputs)
    x = BatchNormalization(name=f'{name}_bn1')(x)
    x = ReLU(name=f'{name}_relu1')(x)
    
    # Second convolutional layer
    x = Conv1D(filters=filters, kernel_size=kernel_size, strides=1,
               padding='same', use_bias=False, name=f'{name}_conv2')(x)
    x = BatchNormalization(name=f'{name}_bn2')(x)
    
    # Skip connection
    if stride != 1 or inputs.shape[-1] != filters:
        # Need to match dimensions for skip connection
        shortcut = Conv1D(filters=filters, kernel_size=1, strides=stride,
                         padding='same', use_bias=False, name=f'{name}_shortcut_conv')(inputs)
        shortcut = BatchNormalization(name=f'{name}_shortcut_bn')(shortcut)
    else:
        shortcut = inputs
    
    # Add skip connection and apply ReLU
    x = Add(name=f'{name}_add')([x, shortcut])
    x = ReLU(name=f'{name}_relu2')(x)
    
    return x

# ============================================================================
# 3. MODEL ARCHITECTURE
# ============================================================================

def create_resnet_bilstm_model(input_shape=(320, 19), num_classes=2, lstm_units=128, dropout_rate=0.5):
    """
    Create ResNet+BiLSTM model as per the specified architecture
    
    Architecture:
    1. Initial Conv1D + MaxPool
    2. 3 Residual Blocks with increasing filters (64, 128, 256)
    3. Global Average Pooling
    4. BiLSTM layer
    5. Dense classification layers
    """
    print("🏗️ Building ResNet+BiLSTM Architecture...")
    
    # Input Layer
    inputs = Input(shape=input_shape, name='eeg_input')
    
    # Initial convolutional layer
    x = Conv1D(filters=64, kernel_size=7, strides=2, padding='same', 
               use_bias=False, name='initial_conv')(inputs)
    x = BatchNormalization(name='initial_bn')(x)
    x = ReLU(name='initial_relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2, name='initial_maxpool')(x)
    
    # Residual blocks
    x = residual_block_1d(x, filters=64, name='resblock1')
    x = residual_block_1d(x, filters=128, name='resblock2')
    x = residual_block_1d(x, filters=256, name='resblock3')
    
    # Global Average Pooling
    x = GlobalAveragePooling1D(name='global_avgpool')(x)
    
    # Reshape for BiLSTM (add sequence dimension)
    x = Lambda(lambda x: tf.expand_dims(x, axis=1), name='reshape_for_lstm')(x)
    
    # BiLSTM layer
    x = Bidirectional(LSTM(units=lstm_units, return_sequences=False), 
                     name='bilstm')(x)
    
    # Classification layers
    x = Dropout(rate=dropout_rate, name='dropout')(x)
    outputs = Dense(units=num_classes, activation='softmax', name='output')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='ResNet_BiLSTM')
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
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
# 4. TRAINING UTILITIES
# ============================================================================

def create_callbacks(model_name='resnet_bilstm_model'):
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
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# 5. EVALUATION UTILITIES
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
    auc_score = None
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
        'auc_score': auc_score
    }

# ============================================================================
# 6. VISUALIZATION UTILITIES
# ============================================================================

def visualize_feature_maps(model, X_sample, layer_name='resblock3_relu2'):
    """Visualize feature maps from ResNet layers"""
    print(f"\n👁️ Feature Map Visualization from {layer_name}:")
    
    try:
        # Create a model that outputs the feature maps
        feature_model = Model(
            inputs=model.input,
            outputs=model.get_layer(layer_name).output
        )
        
        # Get feature maps
        feature_maps = feature_model.predict(X_sample[0:1])
        
        # Plot feature maps
        n_features = min(16, feature_maps.shape[-1])  # Show up to 16 feature maps
        fig, axes = plt.subplots(4, 4, figsize=(12, 10))
        
        for i in range(n_features):
            row = i // 4
            col = i % 4
            axes[row, col].plot(feature_maps[0, :, i])
            axes[row, col].set_title(f'Feature Map {i+1}')
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle(f'Feature Maps from {layer_name}', y=1.02)
        plt.show()
        
    except Exception as e:
        print(f"   ⚠️ Feature map visualization failed: {e}")

def compare_models_performance(results_cnn_lstm, results_resnet_bilstm):
    """Compare performance between CNN+LSTM and ResNet+BiLSTM models"""
    models = ['CNN+LSTM+Attention', 'ResNet+BiLSTM']
    auc_scores = [results_cnn_lstm.get('auc_score', 0), results_resnet_bilstm.get('auc_score', 0)]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, auc_scores, color=['skyblue', 'lightcoral'])
    plt.title('Model Performance Comparison')
    plt.ylabel('AUC Score')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, score in zip(bars, auc_scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=12)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ============================================================================
# 7. MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Main training pipeline for ResNet+BiLSTM model"""
    print("🚀 Starting ResNet+BiLSTM Neonatal Seizure Detection Training Pipeline")
    print("=" * 80)
    
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
    
    # Reshape for ResNet+BiLSTM
    X_reshaped = reshape_for_resnet_bilstm(X)
    
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
    print(f"   Input shape per sample: {X_train.shape[1:]} (320 time points × 19 channels)")
    print(f"   Total 10-second EEG windows: {X_train.shape[0] + X_test.shape[0]}")
    
    # Create ResNet+BiLSTM model
    model = create_resnet_bilstm_model(
        input_shape=X_train.shape[1:],
        num_classes=2,
        lstm_units=128,
        dropout_rate=0.5
    )
    print_model_summary(model)
    
    # Create callbacks
    callbacks = create_callbacks(model_name='resnet_bilstm_seizure_model')
    
    # Train model
    print("\n🏃 Starting ResNet+BiLSTM model training...")
    
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
    
    # Visualize feature maps (if applicable)
    if len(X_test) > 0:
        visualize_feature_maps(model, X_test)
    
    # Save final model
    model.save('resnet_bilstm_seizure_detection_model.h5')
    print(f"\n💾 ResNet+BiLSTM model saved as 'resnet_bilstm_seizure_detection_model.h5'")
    
    return model, history, results

# ============================================================================
# 8. USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Run the complete pipeline
    model, history, results = main()
    
    if results is not None and 'auc_score' in results and results['auc_score'] is not None:
        print("\n🎉 Training Complete!")
        print(f"Final AUC Score: {results['auc_score']:.4f}")
        
        # Clinical validation suggestions
        print("\n💡 For clinical validation, consider implementing:")
        print("   - Leave-one-patient-out cross-validation")
        print("   - Stratified K-fold validation")
        print("   - Temporal validation (early vs late recordings)")
        print("   - Feature importance analysis from ResNet layers")
    else:
        print("\n⚠️ Training did not complete successfully")
