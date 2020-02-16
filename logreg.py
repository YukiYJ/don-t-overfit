import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')
sample_submission = pd.read_csv('input/sample_submission.csv')
target = train['target']

#dropping the id column
# train = train.drop(['id','target'], axis=1)
# test = test.drop('id', axis=1)

# selected by RFECV with lasso
features = [ '16', '33', '43', '45', '52', '63', '65', '73', '90', '91', '117', '133', '134', '149', '189', '199', '217', '237', '258', '295']

train = train[features]
test = test[features]

#Scaling the data
sc = StandardScaler()
X_train = sc.fit_transform(train)
X_test = sc.fit_transform(test)


def cv_score(model,cv):
    score=0
    n=5
    for i in range(n):
        kfolds = KFold(n_splits=cv, shuffle=True)
        score=score+cross_val_score(model, X_train, target, scoring='roc_auc', cv=kfolds)
    return score/n


#logreg

clf = LogisticRegression(class_weight='balanced', solver='liblinear', penalty ='l1', C= 0.2, max_iter=10000, random_state=300)
clf.fit(X_train, target)
y_pred_logreg = clf.predict_proba(X_test)
sample_submission['target'] = y_pred_logreg[:,1]
sample_submission.to_csv('output/logreg_C2_features.csv', index=False)

# for cv in range(3,11):
#
#     clf = LogisticRegressionCV(class_weight='balanced', solver='liblinear', penalty ='l1', Cs=100, max_iter=10000, scoring='roc_auc', cv=cv)
#     clf.fit(X_train, target)
#     y_pred_logreg = clf.predict_proba(X_test)
#
#     print("cv=", cv)
#     print("best C:", clf.C_)
#
#     scores = cv_score(clf,cv)
#     print("ROCAUC scores=", scores)
#     print("score=", sum(scores) / cv)
#
#     sample_submission['target'] = y_pred_logreg[:,1]
#     sample_submission.to_csv('output/LogRegCV_cv' + str(cv) + '.csv', index=False)

