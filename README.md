# Neural Network for Pima Indians Diabetes Prediction

## 🎯 Problem Overview

This project implements a **neural network classifier** to predict diabetes in Pima Indian women based on diagnostic measurements. The dataset originates from the **National Institute of Diabetes and Digestive and Kidney Diseases** and is widely used for binary classification benchmarking.

### Dataset Information
- **Source**: UCI Machine Learning Repository
- **Samples**: 768 patients
- **Features**: 8 diagnostic measurements
- **Target**: Diabetes diagnosis (0 = No, 1 = Yes)
- **Class Distribution**: ~65% negative, ~35% positive (imbalanced)

### Features Description
| Feature | Description |
|---------|-------------|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration (2 hours in oral glucose tolerance test) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body mass index (weight in kg/(height in m)²) |
| DiabetesPedigreeFunction | Diabetes pedigree function (genetic influence) |
| Age | Age in years |

---

## 🧠 Approach & Methodology

### Data Preprocessing
1. **Missing Value Handling**: Replaced zero values in Glucose, BloodPressure, SkinThickness, Insulin, and BMI with median values (zeros are biologically implausible)
2. **Feature Scaling**: Applied StandardScaler to normalize all features to zero mean and unit variance
3. **Train-Test Split**: 80% training, 20% testing with stratification to preserve class distribution

### Neural Network Architecture
```
Model: Sequential
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
dense_1 (Dense)             (None, 64)                576       
dense_2 (Dense)             (None, 32)                2,080     
dense_3 (Dense)             (None, 16)                528       
dropout (Dropout)           (None, 16)                0         
dense_4 (Dense)             (None, 8)                 136       
dense_5 (Dense)             (None, 1)                 9         
=================================================================
Total params: 3,329
Trainable params: 3,329
Non-trainable params: 0
```

### Training Configuration
- **Optimizer**: Adam (learning_rate=0.001)
- **Loss Function**: Binary Crossentropy
- **Batch Size**: 32
- **Early Stopping**: patience=5, monitoring validation loss
- **Dropout Rate**: 0.5 (to reduce overfitting)

---

## 📊 Results & Analysis

### Model Performance
The neural network achieved the following results on the test set:

| Metric | Score |
|--------|-------|
| **Test Accuracy** | ~72-75% |
| **Precision (Class 1)** | ~65-70% |
| **Recall (Class 1)** | ~55-65% |
| **F1-Score** | ~60-65% |

### Key Findings

1. **Dataset Challenges**: The Pima Indians Diabetes dataset is inherently difficult due to:
   - Class imbalance (more non-diabetic cases)
   - Missing values encoded as zeros
   - Limited sample size (768 instances)
   - High feature correlation

2. **Model Behavior**: The training curves show:
   - Rapid initial learning in the first 10-15 epochs
   - Validation loss stabilization indicating proper generalization
   - Early stopping prevented overfitting by terminating training at optimal point

3. **Confusion Matrix Insights**:
   - The model tends to have higher accuracy on the majority class (non-diabetic)
   - Some false negatives occur due to class imbalance
   - Overall balanced performance considering dataset limitations

4. **Comparison with Benchmarks**: The achieved accuracy of 72-75% is consistent with published results on this dataset. Simple logistic regression typically achieves ~70%, while more complex models rarely exceed 78% without extensive feature engineering.

5. **Clinical Relevance**: In a medical context, the recall for diabetic patients is crucial. While the model shows reasonable performance, real-world deployment would require careful consideration of the cost of false negatives (missed diabetes cases).

---

## 📁 Repository Structure

```
neural-network-diabetes/
├── pima_diabetes_nn.ipynb      # Complete Jupyter notebook with code and analysis
├── README.md                    # This documentation file
├── LICENSE                      # License information
└── results/
    ├── training_curves.png      # Loss and accuracy curves over epochs
    ├── confusion_matrix.png     # Test set confusion matrix visualization
    └── metrics_summary.txt      # Detailed performance metrics
```

---

## 🚀 How to Run

### Prerequisites
```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn
```

### Execution
1. Clone the repository:
   ```bash
   git clone https://github.com/MoHazem02/neural-network-diabetes.git
   cd neural-network-diabetes
   ```

2. Run the Jupyter notebook:
   ```bash
   jupyter notebook pima_diabetes_nn.ipynb
   ```

3. Execute all cells sequentially to reproduce results

---

## 📹 Video Presentation

[[Link to recorded video presentation explaining the project]](https://drive.google.com/file/d/1EGwTGw8OPMDgxMjQ4zy6w6cZfVOF3_JT/view?usp=sharing)

---

## 👤 Author

Mohamed - Deep Learning Course Assignment

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
