# 01.03 classifier model (train)

# train initial classifier, then improve with grid search
# https://stackoverflow.com/questions/30102973/how-to-get-best-estimator-on-gridsearchcv-random-forest-classifier-scikit
rfc = RandomForestClassifier(
    n_jobs=-1,
    max_features='sqrt',
    n_estimators=100,
    oob_score=True,
    random_state=12345
)
# run grid search to find best parameters
# note: grid search takes ~5 min to run, so uncomment to run
# param_grid = {
#     'n_estimators': [200, 300, 400, 500],
#     'max_features': ['auto', 'sqrt', 'log2']
# }
# rfc_cv = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
# rfc_cv.fit(X_train, y_train)
# output best parameters, then refit model
# https://www.kaggle.com/sociopath00/random-forest-using-gridsearchcv
# print('Grid Search Results - Best Parameters:')
# print(rfc_cv.best_params_)
# output:
# Grid Search Results - Best Parameters:
# {'max_features': 'auto', 'n_estimators': 200}
# refit model with best parameters from grid search
rfc_params = RandomForestClassifier(
    n_jobs=-1,
    max_features='auto',
    n_estimators=200,
    oob_score=True,
    random_state=12345
)
rfc_params.fit(X_train, y_train)
