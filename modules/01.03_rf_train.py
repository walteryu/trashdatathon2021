# 01.03 rf classifier model (train)

# filter data for model train/predict
# https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html
df_subset = df[[
    'OBJECTID',
    'SegmentID',
    'Seg_ID_Sco',
    'Seg_Bk_Sco',
    'Seg_LL_Sco',
    'Seg_Wd_Sco',
    'SegScore'
]]

# split train and test data
# https://stackoverflow.com/questions/24147278/how-do-i-create-test-and-train-samples-from-one-dataframe-with-pandas
print('Output df shape:')
print(df.shape, '\n')
train, test = train_test_split(df_subset, test_size=0.2)
print('Output train df shape:')
print(train.shape, '\n')
print('Output test df shape:')
print(test.shape, '\n')

# train rf classifier
# https://stackabuse.com/classification-in-python-with-scikit-learn-and-pandas/
y_train = train.iloc[:,6]
X_train = train.iloc[:,:6]
y_test = test.iloc[:,6]
X_test = test.iloc[:,:6]

# encode labels as continuous
# https://stackoverflow.com/questions/41925157/logisticregression-unknown-label-type-continuous-using-sklearn-in-python
# lab_enc = preprocessing.LabelEncoder()
# y_train = lab_enc.fit_transform(y_train)
# y_test = lab_enc.fit_transform(y_test)
y_train = y_train.astype('int64')
y_test = y_test.astype('int64')
print('Output X_train df info:')
print(X_train.info(), '\n')
print('Output X_test df info:')
print(X_test.info(), '\n')
print('Output y_train df info:')
print(y_train.dtype, '\n')
print('Output y_test df info:')
print(y_test.dtype, '\n')

# train rf classifier
RF = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,
    # max_depth=5,
    random_state=12345
)
RF.fit(X_train, y_train)
# RF.predict(X_test)
# print('Random Classifier Accuracy Score:')
# round(RF.score(X_test,y_test), 4)
