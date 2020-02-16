import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import roc_auc_score

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


#cv
def cv_score(model,cv):
    # kfolds = KFold(n_splits=5, shuffle=True)
    return cross_val_score(model, X_train, target, scoring='roc_auc', cv=cv)


# X_train += np.random.normal(0, 0.01, X_train.shape)

# lasso
for cv in range(3,11):

    clf = Lasso(alpha=0.03, tol=0.01, selection='random', random_state=42)
    model = RFECV(clf, step=1, cv=cv, scoring='roc_auc')
    model.fit(X_train, target)
    sub_preds = model.predict(X_test).clip(0, 1)
    # y_score = model.predict(X_train).clip(0, 1)
    # y_true = target.array

    # score = cv_score(model,cv)
    print("cv=", cv)
    print("max grid scores:", max(model.grid_scores_))
    print("num of features:", model.n_features_ )
    # print("ROCAUC score=", score)
    # print("score=", sum(score) / cv)

    sample_submission['target'] = sub_preds
    sample_submission.to_csv('output/lassoRFECV3_cv' + str(cv) + '.csv', index=False)



