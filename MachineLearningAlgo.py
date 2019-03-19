# Load libraries
import pandas
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "/Users/sravanig/Downloads/Wine.csv"

dataset = pandas.read_csv(url)

# shape
print(dataset.shape)

# head
print(dataset.head(20))

# descriptions
print(dataset.describe())

# class distribution
print(dataset.groupby('Customer_Segment').size())

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(7,3), sharex=False, sharey=False)
plt.show()

# histograms
dataset.hist()
plt.show()

# scatter plot matrix
scatter_matrix(dataset, alpha=0.2, figsize=(11, 11))
plt.show()

# Split-out validation dataset
array = dataset.values
X = array[:,0:13]
Y = array[:,13]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Make predictions on validation dataset using LogisticRegression
# lr = LogisticRegression(solver='liblinear', multi_class='ovr')
lr = models[0]
# lr.fit(X_train, Y_train)
predictionsLR = lr.predict(X_validation)
print('*****************************LogisticRegression*******************************')
print('Accuracy Score:')
print(accuracy_score(Y_validation, predictionsLR))
print('Confusion Matrix: ')
print(confusion_matrix(Y_validation, predictionsLR))
print('Classification Report:')
print(classification_report(Y_validation, predictionsLR))
print('*******************************END OF LogisticRegression*****************************')

# Make predictions on validation dataset using LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, Y_train)
predictionsLDA = lda.predict(X_validation)
print('*****************************LDA*******************************')
print('Accuracy Score:')
print(accuracy_score(Y_validation, predictionsLDA))
print('Confusion Matrix: ')
print(confusion_matrix(Y_validation, predictionsLDA))
print('Classification Report:')
print(classification_report(Y_validation, predictionsLDA))
print('*******************************END OF LDA*****************************')

# Make predictions on validation dataset using KNN
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print('**************************** KNN ********************************')
print('Accuracy Score:')
print(accuracy_score(Y_validation, predictions))
print('Confusion Matrix: ')
print(confusion_matrix(Y_validation, predictions))
print('Classification Report:')
print(classification_report(Y_validation, predictions))
print('*****************************END OF KNN*******************************')


# Make predictions on validation dataset using DecisionTreeClassifier
cart = DecisionTreeClassifier()
cart.fit(X_train, Y_train)
predictionsCART = cart.predict(X_validation)
print('*****************************DecisionTreeClassifier*******************************')
print('Accuracy Score:')
print(accuracy_score(Y_validation, predictionsCART))
print('Confusion Matrix: ')
print(confusion_matrix(Y_validation, predictionsCART))
print('Classification Report:')
print(classification_report(Y_validation, predictionsCART))
print('*******************************END OF DecisionTreeClassifier*****************************')

# Make predictions on validation dataset using Naive Bayes
nb = GaussianNB()
nb.fit(X_train, Y_train)
predictionsNb = nb.predict(X_validation)
print('*****************************NAIVE BAYES*******************************')
print('Accuracy Score:')
print(accuracy_score(Y_validation, predictionsNb))
print('Confusion Matrix: ')
print(confusion_matrix(Y_validation, predictionsNb))
print('Classification Report:')
print(classification_report(Y_validation, predictionsNb))
print('*******************************END OF NAIVE BAYES*****************************')


# Make predictions on validation dataset using SVC
svc = SVC(gamma='auto')
svc.fit(X_train, Y_train)
predictionsSVC = svc.predict(X_validation)
print('*****************************SVC*******************************')
print('Accuracy Score:')
print(accuracy_score(Y_validation, predictionsSVC))
print('Confusion Matrix: ')
print(confusion_matrix(Y_validation, predictionsSVC))
print('Classification Report:')
print(classification_report(Y_validation, predictionsSVC))
print('*******************************END OF SVC*****************************')