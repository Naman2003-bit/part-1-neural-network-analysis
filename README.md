# Part 1: Neural Network Fundamentals and Training Behavior Analysis

## Objective
Analyze how a feed-forward neural network learns to predict customer churn
from structured telecom data, and study the effect of different hyperparameters
on model performance.

## Dataset Details
- Source: customer_churn_nn.csv
- Total Records: 2000+ customer entries
- Input Features: 15 (after dropping customer_id)
- Target: churn (0 = retained, 1 = churned)
- Class imbalance handled via stratified split

## Preprocessing Steps
- Dropped non-predictive column: customer_id
- Label encoded: region, plan_type, contract_type, payment_method, autopay_enabled
- Scaled all numerical features using StandardScaler
- Train-Test Split: 85% train, 15% test (stratified)

## Model Architecture
- 3 hidden layers with decreasing neurons (128 → 64 → 32)
- BatchNormalization after each Dense layer
- Dropout rate: 0.3 per layer
- Output: Sigmoid activation for binary classification
- Optimizer: Adam | LR: 0.0005 | Loss: Binary Crossentropy

## Hyperparameter Experiments
5 configurations tested varying layers, neurons, learning rate, and batch size.
Full results saved in results/model_comparison_table.csv

## Reflection
- Weights define connection strength between neurons; biases allow
  each neuron to shift its output independently of inputs
- Without activation functions, stacking layers gives no benefit since
  the entire network collapses into a single linear transformation
- A very high learning rate causes the optimizer to overshoot minima,
  making loss spike unpredictably
- Overfitting: growing gap between train and val accuracy curves
- Underfitting: both curves stay flat and low from the beginning

## Repository Structure
part-1-neural-network-analysis/
├── README.md
├── notebook.ipynb
├── requirements.txt
└── results/
    ├── eda_plots.png
    ├── evaluation_outputs.png
    ├── model_comparison_table.png
    └── model_comparison_table.csv
