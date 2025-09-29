# Neonatal Seizure Detection using CNN+LSTM+Attention

## Overview

This project implements an enhanced deep learning model for neonatal seizure detection using EEG data. The model combines Convolutional Neural Networks (CNN), Long Short-Term Memory (LSTM), and multiple attention mechanisms to achieve high accuracy in detecting seizures in neonatal EEG signals.

## Model Architecture

The Enhanced CNN+LSTM+Attention model includes:

- **Conv1D Layers**: Extract spatial features from EEG channels
- **Channel Attention (ECA)**: Focus on important frequency bands
- **Temporal Attention**: Identify critical time segments
- **Spatial-Temporal Attention**: Combined channel and time attention
- **Multi-Head Self-Attention**: Global pattern recognition
- **LSTM**: Sequential pattern learning
- **Attention Weight Aggregation**: Combine features intelligently

## Key Features

✅ **Proper Binary Classification**: Uses sigmoid activation for seizure detection  
✅ **Scaled Dot-Product Attention**: Mathematically correct attention mechanisms  
✅ **Adaptive Components**: Kernel sizes adapt to input dimensions  
✅ **Robust Architecture**: Handles dimensional mismatches gracefully  
✅ **GPU Support**: Optimized for GPU acceleration  

## Files Structure

```
├── cnn_lstm_attention.py      # Main model implementation
├── preprocess.ipynb           # Data preprocessing pipeline
├── check.ipynb               # Data analysis and validation
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

### Key Dependencies:
- TensorFlow >= 2.8.0
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- Scikit-learn >= 1.0.0
- MNE >= 1.0.0 (for EEG processing)
- Matplotlib, Seaborn (for visualization)

## Usage

### 1. Data Preprocessing
Use `preprocess.ipynb` to:
- Load raw EEG data from .edf files
- Apply filtering and artifact removal
- Extract features and create datasets
- Apply temporal stabilization

### 2. Model Training
```python
from cnn_lstm_attention import main

# Run the complete pipeline
model, history, results = main()
```

### 3. Custom Model Creation
```python
from cnn_lstm_attention import create_enhanced_model

# Create model with custom parameters
model = create_enhanced_model(
    input_shape=(320, 19),  # 320 time points, 19 channels
    num_classes=2           # Binary classification
)
```

## Data Format

The model expects preprocessed EEG data in the following format:
- **Input Shape**: (batch_size, 320, 19)
  - 320 time points (10 seconds at 32 Hz sampling rate)
  - 19 EEG channels
- **Labels**: Binary (0=non-seizure, 1=seizure)

## Model Performance

The enhanced model achieves:
- High accuracy in seizure detection
- Robust performance across different patients
- Interpretable attention mechanisms
- Real-time inference capability

## Architecture Improvements

Recent fixes include:
1. **Corrected Attention Mechanisms**: Proper softmax normalization
2. **Binary Classification**: Appropriate loss function and output layer
3. **Dimensional Compatibility**: Robust tensor operations
4. **Adaptive Parameters**: Self-adjusting kernel sizes
5. **Error Handling**: Graceful handling of edge cases

## Clinical Applications

This model is designed for:
- Automated neonatal seizure detection
- Clinical decision support
- Continuous EEG monitoring
- Research in neonatal neurology


## Project Team
- [Raghav N](https://github.com/Rag-795)
- [Hari Heman V K](https://github.com/HXMAN76)
- [Mathivanan S](https://github.com/Rag-795)
- [Rashwanth Ram](https://github.com/Rag-795)

## Citation

If you use this code in your research, please cite:
```
@misc{neonatal_seizure_cnn_lstm_attention,
  title={Enhanced Neonatal Seizure Detection using CNN+LSTM+Attention},
  author={Your Name},
  year={2025},
  publisher={GitHub},
  url={https://github.com/HXMAN76/Neo-Natal-Seizure-Detection-IIT-Delhi}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For questions or issues, please:
1. Check the documentation
2. Open an issue on GitHub
3. Contact the maintainers

---

**Note**: This is research code. For clinical applications, please ensure proper validation and regulatory approval.
