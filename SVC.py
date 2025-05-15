import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

# 0 -> Galaxy
# 1 -> QSO
# 2 -> Star
star_labels = [
    'Galaxy', 'QSO', 'Star'
]

# https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17
df = pd.read_csv("cleaned_star_classification.csv")

X = df.iloc[:, :-1].copy().to_numpy()
y = df.iloc[:, -1].copy().to_numpy()

# normalize the data
X = (X - np.average(X, axis=0)) / np.std(X, axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Support Vector Classifier with C and Gamma values picked by an optimal gridsearch
reg = SVC(kernel='rbf', class_weight='balanced')
parameters = {"C": np.linspace(0.5, 1.5, num=5), "gamma": np.linspace(0.5, 1.5, num=5)}
grid_search = GridSearchCV(reg, param_grid = parameters, cv=5, scoring="f1_weighted")
grid_search.fit(X_train,y_train)
score_dif = pd.DataFrame(grid_search.cv_results_)

c = grid_search.best_params_['C']
Gamma = grid_search.best_params_['gamma']

clf = SVC(C=c, gamma=Gamma, kernel='rbf', class_weight='balanced')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"Score: {f1_score(y_test, y_pred, average='weighted'):.3f}")

cm = confusion_matrix(y_test, clf.predict(X_test), normalize="true")
disp_cm = ConfusionMatrixDisplay(cm, display_labels=star_labels)
disp_cm.plot()
plt.show()
