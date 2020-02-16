import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_selection import RFECV
import statsmodels.api as sm
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_classif

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')
sample_submission = pd.read_csv('input/sample_submission.csv')

y_train = train['target']

#dropping the id column
train = train.drop(['id','target'], axis=1)
test = test.drop('id', axis=1)

#Scaling the data
sc = StandardScaler()
X_train = sc.fit_transform(train)
X_test = sc.fit_transform(test)


def cv_score(model,cv):
    score=0
    n=5
    for i in range(n):
        # kfolds = KFold(n_splits=cv, shuffle=True)
        score=score+cross_val_score(model, X_train, y_train, scoring='roc_auc', cv=cv)
    return score/n


#logreg
lr = LogisticRegression(class_weight='balanced', solver='liblinear', penalty ='l1', C= 0.1, max_iter=10000)
lr.fit(X_train, y_train)
preds_logreg = lr.predict_proba(X_test)


#lasso
lasso_model = Lasso(alpha=0.03, tol=0.01, selection='random', random_state=42)
lasso = RFECV(lasso_model, step=1, cv=9, scoring='roc_auc')
lasso.fit(X_train, y_train)
preds_lasso = lasso.predict(X_test).clip(0, 1)


#ridge
cv=6
ridge_model = Ridge(random_state=42, max_iter=10000, alpha=2400)
ridge = RFECV(ridge_model, step=1, cv=cv, scoring='roc_auc')
ridge.fit(X_train, y_train)
preds_ridge = ridge.predict(X_test).clip(0, 1)


#GLM
select = SelectKBest(k=15, score_func=f_classif)
select.fit(X_train, y_train)

up_X_train = pd.DataFrame(select.transform(X_train))
up_X_test = pd.DataFrame(select.transform(X_test))
model = sm.GLM(y_train, up_X_train,family=sm.families.Binomial())
model_results = model.fit()
preds_glm = model_results.predict(up_X_test)

sample_submission['target'] = preds_glm
sample_submission.to_csv('output/GLM_k15.csv', index=False)


def blend_models_predict():
    return 0.6 * preds_lasso + 0.2 * preds_logreg[:,1] + 0.2*preds_ridge


sample_submission['target'] = blend_models_predict()
sample_submission.to_csv('output/lasso6_logreg2_ridge2.csv', index=False)

# #stack
# cv=5
# stack = StackingCVRegressor(regressors=(ridge, lr, lasso),
#                                 meta_regressor=lasso,
#                                 use_features_in_secondary=True)
# stack.fit(np.array(X_train), np.array(target))
#
#
# sample_submission['target'] = stack.predict(np.array(X_test))
# sample_submission.to_csv('output/stack.csv', index=False)
#
# #score
# print("cv=", cv)
# for clf, label in zip([ridge, lr, lasso, stack], ['Ridge', 'Lasso', 'LogisticRegression', 'StackingCVRegressor']):
#     scores = cv_score(clf,cv)
#     print("ROCAUC score: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
#
#
# #visualization
# with plt.style.context(('seaborn-whitegrid')):
#     plt.scatter(X_train, target, c='lightgray')
#     plt.plot(X_train, sample_submission['target'], c='darkgreen', lw=2)
# plt.show()



