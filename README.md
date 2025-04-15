# Diabetes Prediction using Machine Learning

This project predicts whether a patient has diabetes based on diagnostic measurements using Machine Learning. The model is trained on the Pima Indians Diabetes Dataset, which includes features like glucose levels, BMI, and insulin levels.

## Features

- Data preprocessing (handling missing values, scaling)
- Multiple model implementations (Logistic Regression, Random Forest, SVM)
- Comprehensive model evaluation
- Visualization of results
- Easy-to-use prediction interface

## Dataset

The dataset (`diabetes.csv`) contains 768 samples with 8 features and a binary outcome (1 = diabetic, 0 = non-diabetic).

### Features:
1. Pregnancies
2. Glucose
3. Blood Pressure
4. Skin Thickness
5. Insulin
6. BMI
7. Diabetes Pedigree Function
8. Age
9. Outcome (Target)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/diabetes-prediction.git
cd diabetes-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your `diabetes.csv` file in the project directory.

2. Run the main script:
```bash
python diabetes_prediction.py
```

The script will:
- Load and preprocess the data
- Train multiple models
- Evaluate and compare model performance
- Generate confusion matrix visualizations
- Make a sample prediction

## Model Performance

The script evaluates three different models:
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

Each model's performance is measured using:
- Accuracy
- Classification Report (Precision, Recall, F1-score)
- Confusion Matrix

## Making Predictions

You can make predictions for new patients by modifying the example data in the `main()` function:

```python
example_data = [[2, 120, 70, 20, 80, 25, 0.3, 30]]  # Sample input
prediction, probability = make_prediction(model, scaler, example_data)
```

## Project Structure

```
├── diabetes_prediction.py  # Main Python script
├── diabetes.csv            # Dataset
├── README.md              # Project documentation
└── requirements.txt       # Dependencies
```

## Future Improvements

- Implement Neural Networks using TensorFlow/Keras
- Add hyperparameter tuning using GridSearchCV
- Create a web interface using Flask/Streamlit
- Add more advanced feature engineering
- Implement cross-validation
- Add model persistence (save/load trained models)

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 