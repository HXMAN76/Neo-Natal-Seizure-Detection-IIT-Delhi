import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv1D, BatchNormalization, ReLU, MaxPooling1D, 
    GlobalMaxPooling1D, LSTM, Dropout, Dense, Input,
    LayerNormalization, Reshape, Permute, Multiply, 
    Add, Concatenate, Lambda, Dot, Activation
)
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("🧠 Enhanced Neonatal Seizure Detection: CNN+LSTM+Attention Implementation")
print("="*80)

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_preprocess_data(csv_file='complete_preprocessed_dataset_shuffled_with_ica.csv'):
    """
    Load preprocessed EEG data and prepare for enhanced CNN+LSTM+Attention model
    
    Assumes CSV structure:
    - Columns 0-6079: EEG features arranged as [t1_ch1, t1_ch2, ..., t1_ch19, t2_ch1, ..., t320_ch19]
    - Column 6080: Labels (0=non-seizure, 1=seizure) 
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

def reshape_for_cnn_lstm(X, n_channels=19, n_timesteps=320):
    """
    Reshape flattened EEG data for CNN+LSTM+Attention input
    
    Current: (batch, 6080) - flattened as [t1_ch1, t1_ch2, ..., t1_ch19, t2_ch1, ..., t320_ch19]
    Target: (batch, n_timesteps, n_channels) - for Conv1D processing with attention
    """
    print(f"🔄 Reshaping data from {X.shape} to CNN+LSTM+Attention format...")
    
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
# 2. ATTENTION MECHANISMS
# ============================================================================

def channel_attention_eca(inputs, kernel_size=3, name='eca_block'):
    """
    Efficient Channel Attention (ECA) for adaptive channel weighting
    Focuses on important frequency bands in EEG
    """
    # Global average pooling across time dimension using Lambda layer
    avg_pool = Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True), name=f'{name}_avgpool')(inputs)
    
    # Transpose to shape [batch, channels, 1] using Permute
    transposed = Permute((2, 1), name=f'{name}_permute1')(avg_pool)
    
    # 1D convolution for efficient channel attention
    conv = Conv1D(
        filters=1, 
        kernel_size=kernel_size,
        padding='same',
        activation='sigmoid',
        name=f'{name}_conv'
    )(transposed)
    
    # Transpose back to original shape using Permute
    attention = Permute((2, 1), name=f'{name}_permute2')(conv)
    
    # Apply channel attention weights
    output = Multiply(name=f'{name}_scale')([inputs, attention])
    
    return output

def temporal_attention(inputs, num_heads=4, query_dim=None, key_dim=None, name='temporal_attention'):
    """
    Multi-head temporal attention to focus on important segments in the EEG signal
    """
    if query_dim is None:
        query_dim = inputs.shape[-1]
    if key_dim is None:
        key_dim = inputs.shape[-1]
    
    # Define dimensions
    time_steps = inputs.shape[1]
    hidden_dim = inputs.shape[2]
    
    # Create query, key, value projections
    query = Dense(query_dim, name=f'{name}_query')(inputs)
    key = Dense(key_dim, name=f'{name}_key')(inputs)
    value = Dense(hidden_dim, name=f'{name}_value')(inputs)
    
    # For simplicity, use a single head attention instead of multi-head
    # This avoids complex tensor operations that are difficult in Functional API
    
    # Calculate attention scores using Dot product
    # Reshape for matrix multiplication: (batch, time, dim) -> (batch, time, dim)
    attention_scores = Dot(axes=[2, 2], name=f'{name}_scores')([query, key])
    
    # Apply softmax
    attention_weights = Activation('softmax', name=f'{name}_weights')(attention_scores)
    
    # Apply attention weights to values
    # Use Lambda layer for more complex operations
    def apply_attention(tensors):
        weights, values = tensors
        return tf.matmul(weights, values)
    
    context = Lambda(apply_attention, name=f'{name}_context')([attention_weights, value])
    
    # Final projection
    output = Dense(hidden_dim, name=f'{name}_output')(context)
    
    return output

def spatial_temporal_attention(inputs, spatial_heads=4, temporal_heads=4, name='spatiotemporal_attention'):
    """
    Combined spatial (channel) and temporal attention for EEG seizure patterns
    """
    # Temporal attention (across time)
    temporal_out = temporal_attention(
        inputs, 
        num_heads=temporal_heads,
        name=f'{name}_temporal'
    )
    
    # Spatial attention (across channels)
    # Transpose to make channels the time dimension for attention
    transposed = Permute((2, 1), name=f'{name}_transpose')(inputs)  # [batch, channels, time]
    spatial_out = temporal_attention(
        transposed,
        num_heads=spatial_heads, 
        name=f'{name}_spatial'
    )
    spatial_out = Permute((2, 1), name=f'{name}_transpose_back')(spatial_out)  # [batch, time, channels]
    
    # Combine both attentions with residual connection
    combined = Add(name=f'{name}_combine')([temporal_out, spatial_out, inputs])
    
    return combined

def multi_head_self_attention(inputs, heads=8, d_model=256, d_ff=512, name='self_attention'):
    """
    Simplified self-attention for global pattern recognition
    """
    # Since we're dealing with 1D vector after GlobalMaxPooling, simplify the attention
    
    # Project to query, key, value
    query = Dense(d_model, name=f'{name}_query')(inputs)
    key = Dense(d_model, name=f'{name}_key')(inputs)
    value = Dense(d_model, name=f'{name}_value')(inputs)
    
    # Simple attention mechanism for 1D vectors
    # Calculate attention scores using dot product
    scores = Dot(axes=1, name=f'{name}_scores')([query, key])
    
    # Scale the scores
    def scale_scores(x):
        return x / tf.math.sqrt(tf.cast(d_model, tf.float32))
    
    scaled_scores = Lambda(scale_scores, name=f'{name}_scale')(scores)
    
    # Apply softmax to get attention weights
    weights = Activation('softmax', name=f'{name}_weights')(scaled_scores)
    
    # Apply attention weights to values
    context = Multiply(name=f'{name}_context')([weights, value])
    
    # Feed-forward network
    ffn_output = Dense(d_ff, activation='relu', name=f'{name}_ffn1')(context)
    ffn_output = Dense(d_model, name=f'{name}_ffn2')(ffn_output)
    
    # Add & Norm (residual connection)
    output = Add(name=f'{name}_residual')([context, ffn_output])
    
    return output

def attention_weight_aggregation(inputs, context, context_dim=256, attention_heads=4, name='attn_aggregation'):
    """
    Weight LSTM outputs with attention scores based on context
    """
    # Project inputs and context to common space
    inputs_proj = Dense(context_dim, name=f'{name}_input_proj')(inputs)
    context_proj = Dense(context_dim, name=f'{name}_context_proj')(context)
    
    # Calculate attention scores using Dot layer
    scores = Dot(axes=1, name=f'{name}_scores')([inputs_proj, context_proj])
    
    # Scale the scores
    def scale_scores(x):
        return x / tf.math.sqrt(tf.cast(context_dim, tf.float32))
    
    scaled_scores = Lambda(scale_scores, name=f'{name}_scale')(scores)
    
    # Apply softmax
    weights = Activation('softmax', name=f'{name}_weights')(scaled_scores)
    
    # Weight inputs by attention scores
    weighted = Multiply(name=f'{name}_multiply')([inputs, weights])
    
    return weighted

# ============================================================================
# 3. MODEL ARCHITECTURE
# ============================================================================

def create_enhanced_model(input_shape=(320, 19), num_classes=2):
    """
    Create the enhanced CNN+LSTM+Attention model as per the specified architecture
    """
    print("🏗️ Building Enhanced CNN+LSTM+Attention Architecture...")
    
    # Input Layer
    inputs = Input(shape=input_shape, name='eeg_input')
    
    # Conv1D Layer 1 with Channel Attention
    x = Conv1D(filters=64, kernel_size=7, padding='same', name='conv1d_1')(inputs)
    x = BatchNormalization(momentum=0.99, epsilon=1e-3, name='batch_norm_1')(x)
    x = ReLU(name='relu_1')(x)
    x = channel_attention_eca(x, kernel_size=3, name='channel_attention_1')
    x = MaxPooling1D(pool_size=2, strides=2, name='maxpool_1')(x)
    
    # Conv1D Layer 2 with Temporal Attention
    x = Conv1D(filters=128, kernel_size=5, padding='same', name='conv1d_2')(x)
    x = BatchNormalization(momentum=0.99, epsilon=1e-3, name='batch_norm_2')(x)
    x = ReLU(name='relu_2')(x)
    x = temporal_attention(x, num_heads=4, query_dim=128, key_dim=128, name='temporal_attention')
    x = MaxPooling1D(pool_size=2, strides=2, name='maxpool_2')(x)
    
    # Conv1D Layer 3 with Spatial-Temporal Attention
    x = Conv1D(filters=256, kernel_size=3, padding='same', name='conv1d_3')(x)
    x = BatchNormalization(momentum=0.99, epsilon=1e-3, name='batch_norm_3')(x)
    x = ReLU(name='relu_3')(x)
    x = spatial_temporal_attention(x, spatial_heads=4, temporal_heads=4, name='spatiotemporal_attention')
    
    # Global Max Pooling
    x_pooled = GlobalMaxPooling1D(name='global_maxpool')(x)
    
    # Multi-Head Self-Attention
    x_attention = multi_head_self_attention(x_pooled, heads=8, d_model=256, d_ff=512, name='self_attention')
    
    # Layer Normalization
    x_norm = LayerNormalization(epsilon=1e-6, name='layer_norm')(x_attention)
    
    # Reshape for LSTM
    x_reshaped = Reshape((1, 256), name='reshape_for_lstm')(x_norm)
    
    # LSTM Layer
    lstm_out = LSTM(units=64, return_sequences=False, name='lstm')(x_reshaped)
    
    # Attention Weight Aggregation
    weighted_lstm = attention_weight_aggregation(
        lstm_out, x_pooled, context_dim=256, attention_heads=4, name='attn_aggregation'
    )
    
    # Classification Layers
    x = Dropout(rate=0.5, name='dropout')(weighted_lstm)
    x = Dense(units=128, name='dense_1')(x)
    x = ReLU(name='relu_dense_1')(x)
    x = Dense(units=64, name='dense_2')(x)
    x = ReLU(name='relu_dense_2')(x)
    outputs = Dense(units=num_classes, activation='softmax', name='output')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='EnhancedCNN_LSTM_Attention')
    
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
    print(f"   Trainable Parameters: {model.count_params():,}")
    
    # Calculate model size (approximate)
    model_size_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
    print(f"   Estimated Model Size: {model_size_mb:.2f} MB")

# ============================================================================
# 4. TRAINING UTILITIES
# ============================================================================

def create_callbacks(model_name='enhanced_seizure_model'):
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
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    
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
# 6. VISUALIZATION UTILITIES
# ============================================================================

def visualize_attention_weights(model, X_sample, layer_name=None):
    """Visualize attention weights to interpret model focus"""
    print("\n👁️ Attention Visualization:")
    
    # Get model outputs with intermediate layers
    if layer_name:
        attention_model = Model(
            inputs=model.input,
            outputs=model.get_layer(layer_name).output
        )
        attention_weights = attention_model.predict(X_sample[0:1])
    else:
        # If no specific layer, try to extract from temporal attention
        try:
            attention_model = Model(
                inputs=model.input,
                outputs=model.get_layer('temporal_attention').output
            )
            attention_weights = attention_model.predict(X_sample[0:1])
        except:
            print("   ⚠️ Could not extract attention weights. No compatible layer found.")
            return
    
    # Plot the weights
    plt.figure(figsize=(12, 6))
    if len(attention_weights.shape) >= 3:
        # Take mean across appropriate dimension for visualization
        if attention_weights.shape[1] > attention_weights.shape[2]:
            # More timesteps than channels
            weights = np.mean(attention_weights[0], axis=1)
            plt.plot(weights)
            plt.title('Average Attention Weights Across Time')
            plt.xlabel('Channel')
            plt.ylabel('Attention Weight')
        else:
            # More channels than timesteps or similar
            weights = np.mean(attention_weights[0], axis=0)
            plt.plot(weights)
            plt.title('Average Attention Weights Across Channels')
            plt.xlabel('Time')
            plt.ylabel('Attention Weight')
    else:
        # Already aggregated weights
        plt.plot(attention_weights[0])
        plt.title('Attention Weights')
        plt.xlabel('Feature')
        plt.ylabel('Weight')
    
    plt.grid(True, alpha=0.3)
    plt.show()

def gradcam_visualization(model, X_sample, y_sample, layer_name='conv1d_3'):
    """
    Generate Grad-CAM visualization for CNN layer activations
    Shows which parts of the EEG signal the model focuses on
    """
    try:
        # Create a model that maps input to both the prediction and the convolution output
        grad_model = Model(
            inputs=model.inputs,
            outputs=[model.get_layer(layer_name).output, model.output]
        )
        
        # Get the score for target class
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(X_sample[0:1])
            target = predictions[:, y_sample[0]]
            
        # Get the gradients of the target class with respect to the convolution outputs
        grads = tape.gradient(target, conv_outputs)
        
        # Average the gradients spatially
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
        
        # Apply the heatmap to the convolution output
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        
        # Normalize heatmap
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
        
        # Plot the heatmap
        plt.figure(figsize=(12, 4))
        plt.plot(heatmap[0])
        plt.title('Grad-CAM: Model Focus Areas')
        plt.xlabel('Time')
        plt.ylabel('Activation')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"   ⚠️ Grad-CAM visualization failed: {e}")

# ============================================================================
# 7. MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Main training pipeline for enhanced model"""
    print("🚀 Starting Enhanced Neonatal Seizure Detection Training Pipeline")
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
    
    # Reshape for CNN+LSTM+Attention
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
    print(f"   Input shape per sample: {X_train.shape[1:]} (320 time points × 19 channels)")
    print(f"   Total 10-second EEG windows: {X_train.shape[0] + X_test.shape[0]}")
    
    # Create enhanced model
    model = create_enhanced_model(input_shape=X_train.shape[1:])
    print_model_summary(model)
    
    # Create callbacks
    callbacks = create_callbacks(model_name='enhanced_seizure_model_attention')
    
    # Train model with validation_split instead of separate validation data
    print("\n🏃 Starting enhanced model training...")
    
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
    
    # Visualize attention mechanisms (if applicable)
    if len(X_test) > 0:
        visualize_attention_weights(model, X_test)
        gradcam_visualization(model, X_test, y_test)
    
    # Save final model
    model.save('enhanced_seizure_detection_model.h5')
    print(f"\n💾 Enhanced model saved as 'enhanced_seizure_detection_model.h5'")
    
    return model, history, results

# ============================================================================
# 8. USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Run the complete pipeline
    model, history, results = main()
    
    if results is not None and 'auc_score' in results:
        print("\n🎉 Training Complete!")
        print(f"Final AUC Score: {results['auc_score']:.4f}")
        
        # Clinical validation suggestions
        print("\n💡 For clinical validation, consider implementing:")
        print("   - Leave-one-patient-out cross-validation")
        print("   - Stratified K-fold validation")
        print("   - Temporal validation (early vs late recordings)")
        print("   - Interpret attention weights for explainability")
    else:
        print("\n⚠️ Training did not complete successfully")