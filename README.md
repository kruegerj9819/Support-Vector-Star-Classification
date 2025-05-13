# Star Classification using Support Vector Classifiers
This project uses a **Support Vector Classifier (SVC)** to classify celestial objects into three categories: **Galaxy**, **QSO** (Quasi-Stellar Object), and **Star**. The dataset contains various features of celestial objects, and the goal is to predict their classification based on these features.

## Dataset
The dataset used for this project is sourced from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17), which includes data on celestial objects. The objects are categorized into three classes:
- Galaxy
- QSO (Quasi-Stellar Object)
- Star
The dataset is pre-processed and saved as `cleaned_star_classification.csv`.

## Installation
To run this project, clone this repository and install the required dependencies. You can use the following command to install the necessary libraries:
```
pip install -r requirements.txt
```

### Requirements
- pandas
- numpy
- matplotlib
- scikit-learn

## How to Run
1. Clone this repository or download the necessary files.
2. Ensure that the dataset cleaned_star_classification.csv is available in the same directory as the Python script.
3. Run the Python script SVC.py to train and evaluate the SVC model.
```
python SVC.py
```
The script will:
- Load the dataset and preprocess the data.
- Split the data into training and test sets.
- Use a Support Vector Classifier with an optimal grid search to find the best `C` and `gamma` hyperparameters.
- Evaluate the model using the F1 Score and display a normalized confusion matrix.

## Results
The script will output:
- The weighted F1 Score, which is used to evaluate the model’s performance across the classes.
- A normalized confusion matrix to visualize the model's performance on the test data.

### Example Output
```
Score: 0.920
```

### Confusion Matrix
This project demonstrates how to use **Support Vector Classifiers (SVC)** to classify celestial objects. Hyperparameter tuning is performed via grid search to optimize the `C` and `gamma` parameters. The results are evaluated using the **F1 Score** and **Confusion Matrix** to assess the model's classification accuracy.

## Interpretation
![star-classification-confusion matrix](https://github.com/user-attachments/assets/1936e67e-016b-4c28-9f3b-ced786bfaabb)

The Support Vector Classifier did an excellent job at accurately classifying the start based on the data it was given. There does not seem to be any major overfitting or underfitting occuring, as the number of missclassifications is very low. The only missclassification that stands out is between stars and galaxies, but there is only around 10% of misscalssifications there.
