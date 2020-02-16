import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import GridSearchCV

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')
sample_submission = pd.read_csv('input/sample_submission.csv')

target = train['target']

#dropping the id column
train = train.drop(['id','target'], axis=1)
test = test.drop('id', axis=1)

#Scaling the data
sc = StandardScaler()
X_train = sc.fit_transform(train)
X_test = sc.fit_transform(test)


lasso = Lasso(random_state=42, max_iter=10000, tol=0.01)
# alphas = np.logspace(-3, -0.1, 50)
alphas = np.arange(0.01,0.1,0.01)
tuned_parameters = [{'alpha': alphas}]

clf = GridSearchCV(lasso, tuned_parameters, cv=9, refit=True, scoring='roc_auc', return_train_score=True)
clf.fit(X_train, target)

scores_train = clf.cv_results_['mean_train_score']
scores_test = clf.cv_results_['mean_test_score']

plt.figure().set_size_inches(8, 6)
plt.semilogx(alphas, scores_train)
plt.semilogx(alphas, scores_test)
plt.xlabel('alpha')
plt.ylabel('score')

plt.show()
plt.figure().savefig("output/lasso_alphas.png")