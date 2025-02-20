
# Income Prediction Using Machine Learning

## Overview
This project aims to predict whether an individual's income exceeds $50,000 per year based on various demographic and economic features. The dataset used is the Adult Income Dataset from the UCI Machine Learning Repository. The dataset contains information about individuals' income levels based on features such as age, education, occupation, and more.

## Findings

### Optimization Techniques and Results

| Training Instance | Optimizer Used | Regularizer Used | Epochs | Early Stopping | Number of Layers | Learning Rate | Accuracy | F1 Score | Recall | Precision | ROC AUC |
|-------------------|----------------|------------------|--------|----------------|------------------|---------------|----------|----------|--------|-----------|---------|
| Simple Model      | Adam           | None             | 10     | No             | 2                | 0.001         | 0.85     | 0.86     | 0.87    | 0.84      | 0.91    |
| Instance 1        | Adam           | L2               | 10     | Yes            | 2                | 0.001         | 0.88     | 0.89     | 0.90    | 0.87      | 0.93    |
| Instance 2        | RMSprop        | L1               | 10     | No             | 2                | 0.001         | 0.86     | 0.87     | 0.88    | 0.85      | 0.92    |
| Instance 3        | Adam           | L1               | 10     | Yes            | 3                | 0.01          | 0.90     | 0.91     | 0.92    | 0.89      | 0.94    |
| Instance 4        | RMSprop        | L2               | 10     | Yes            | 3                | 0.001         | 0.91     | 0.92     | 0.93    | 0.90      | 0.95    |

### Summary of Optimization Techniques

The combination of Adam optimizer, L2 regularization, and early stopping worked the best. This combination provided the highest accuracy and F1 score among the tested models. The neural network implementation with these techniques outperformed the classical ML algorithms in terms of accuracy and F1 score.

### Comparison of ML Algorithm and Neural Network

The neural network models generally outperformed the classical ML algorithms (Logistic Regression, SVM, XGBoost) in terms of accuracy and F1 score. The best neural network model (Instance 4) achieved an accuracy of 0.91 and an F1 score of 0.92, while the best classical ML algorithm (XGBoost) achieved an accuracy of 0.89 and an F1 score of 0.90.

### Hyperparameters of Classical ML Algorithms

- **Logistic Regression**: `max_iter=1000`
- **SVM**: `probability=True`
- **XGBoost**: Default hyperparameters

## Instructions

### Running the Notebook

1. **Install Required Libraries**: Ensure you have the required libraries installed. You can install them using pip:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow xgboost

2. **Load the Dataset**: The dataset is loaded from the UCI Machine Learning Repository. Ensure you have an active internet connection.

3. **Run the Notebook**: Open the notebook in a Jupyter environment and run each cell sequentially.
Loading the Best Saved Model
The best model (Instance 4) is saved in the saved_models directory. You can load it using the following code:
```bash 
from tensorflow.keras.models import load_model

best_model = load_model('saved_models/model_4.h5') 
```
### Making Predictions

To make predictions using the best model, use the following code:
```python
# Load the best model
best_model = load_model('saved_models/model_4.h5')

# Make predictions on the test data
predictions = best_model.predict(X_test)
predictions = (predictions > 0.5).astype(int)
```
### Summary
The combination of RMSprop optimizer, L2 regularization, and early stopping worked the best. The neural network implementation outperformed the classical ML algorithms in terms of accuracy and F1 score.

#### Instructions for Running the Notebook

1. Install Required Libraries: Ensure you have the required libraries installed. You can install them using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```
2. Load the Dataset: The dataset is loaded from the UCI Machine Learning Repository. Ensure you have an active internet connection.

3. Run the Notebook: Open the notebook in a Jupyter environment and run each cell sequentially.

4. Load the Best Model: The best model is saved in the saved_models directory. You can load it using the following code:
```python
from tensorflow.keras.models import load_model
best_model = load_model('saved_models/best_model.h5')
```

### Walkthrough Video 

Link to walkthrough vide: https://drive.google.com/drive/folders/1QxsAvpemVIdrM1ifNkOp9E3qisBGk6Jy?usp=sharing 