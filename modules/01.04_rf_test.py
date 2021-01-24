# 01.04 rf classifier model (test)

# make predictions and calculate accuracy score
# https://www.blopig.com/blog/2017/07/using-random-forests-in-python-with-scikit-learn/
predicted = RF.predict(X_test)
accuracy = accuracy_score(y_test, predicted)
print(f'Out-of-bag score estimate: {RF.oob_score_:.4}')
print(f'Mean accuracy score: {accuracy:.4}')

# create/plot confusion matrix
# https://www.blopig.com/blog/2017/07/using-random-forests-in-python-with-scikit-learn/
cm = pd.DataFrame(
    confusion_matrix(y_test, predicted),
    # columns=df['SegScore'],
    # index=df['SegScore']
)
sns.heatmap(cm,
    annot=True,
    cmap='coolwarm',
    fmt='g'
)
