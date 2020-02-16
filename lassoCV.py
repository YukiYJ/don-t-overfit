import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
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

# lasso
for cv in range(2,30,2):

    model = LassoCV(alphas=[0.1,0.05,0.03,0.01,0.005,0.001], tol=0.01, cv=cv, selection='random', random_state=42)
    # model = RFECV(clf, step=1, cv=(splits - 1))
    model.fit(X_train, target)
    sub_preds = model.predict(X_test).clip(0, 1)
    y_score = model.predict(X_train).clip(0, 1)
    y_true = target.array

    print("cv=", cv, "ROCAUC score=", roc_auc_score(y_true, y_score))

    sample_submission['target'] = sub_preds
    sample_submission.to_csv('output/lasso_cv'+str(cv)+'.csv', index=False)