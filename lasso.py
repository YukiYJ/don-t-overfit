import pandas as pd

from sklearn.linear_model import Lasso
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE, RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from matplotlib import pyplot as plt

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

# lasso
splits = 10
# folds = RepeatedStratifiedKFold(n_splits=splits, n_repeats=20, random_state=42)
# oof_preds = np.zeros(X_train.shape[0])
# sub_preds = np.zeros(X_test.shape[0])

model = Lasso(alpha=0.03, tol=0.01, selection='random', random_state=42)
# model = RFECV(clf, step=1, cv=(splits - 1))
model.fit(X_train, target)
sub_preds = model.predict(X_test).clip(0, 1)

# model_lasso = LassoCV(alphas=[0.01,0.03,0.05,0.001,0.0005], tol=0.01, selection='random', random_state=42)
# model_lasso.fit(X_train, target)
# lasso_train_pred = model_lasso.predict(X_train)
# lasso_pred = np.expm1(model_lasso.predict(X_test))

# plt.plot(model_lasso.alphas_, model_lasso.mse_path_)
# plt.show()

sample_submission['target'] = sub_preds
sample_submission.to_csv('output/submission_lasso_2.csv', index=False)