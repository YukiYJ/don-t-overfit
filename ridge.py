import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
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
    score=0
    n=10
    for i in range(n):
        kfolds = KFold(n_splits=cv, shuffle=True)
        score=score+cross_val_score(model, X_train, target, scoring='roc_auc', cv=kfolds)
    return score/n


cv = 5
# alphas = [0.1,1.0,10.0, 100.0, 1000.0, 10000.0]
# alphas = np.logspace(3, 4, 30)
# alphas = [2400]
# tuned_parameters = [{'alpha': alphas}]
#
# ridge = Ridge(random_state=42, max_iter=10000)
# model = GridSearchCV(ridge, tuned_parameters, cv=cv, refit=True, scoring='roc_auc', return_train_score=True)
# model.fit(X_train, target)
#
# score = cv_score(model,cv)
# print("Ridge: cv=", cv)
#
# print("best parameters:",model.best_params_ )
# print("best estimator:",model.best_estimator_)
# print("best score:",model.best_score_)
#
# print("ROCAUC scores：", score)
# print("avg score=", sum(score) / cv)



#ridge
for cv in range(3,11):

    ridge = Ridge(random_state=42, max_iter=10000, alpha=2400)
    model = RFECV(ridge, step=1, cv=cv, scoring='roc_auc')
    model.fit(X_train, target)
    sub_preds = model.predict(X_test).clip(0, 1)

    score = cv_score(model,cv)
    print("cv=", cv)
    print("max grid scores:", max(model.grid_scores_))
    print("num of features:", model.n_features_ )
    print("ROCAUC scores=", score)
    print("avg score=", sum(score) / cv)

    sample_submission['target'] = sub_preds
    sample_submission.to_csv('output/ridgeRFECV1_cv' + str(cv) + '.csv', index=False)



