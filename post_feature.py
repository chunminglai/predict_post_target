
# coding: utf-8

# In[99]:

print(__doc__)
get_ipython().magic('matplotlib inline')
edit here to test github
import matplotlib.pyplot as plt
import numpy as np
from numpy  import array
from itertools import cycle
import pandas as pd
from sklearn import svm, datasets
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.cross_validation import cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
# import some data to play with
#iris = datasets.load_iris()
#X = iris.data
#y = iris.target
data = pd.read_csv('post_feature_new.txt',header = None)
y1 = data[0].tolist()
y = np.array(y1)
data = data.drop(data.columns[[0,1,2,5]], axis=1)
#print (data)
data2 = data.as_matrix()
#nor_data = preprocessing.normalize(data2, norm='l2')
#print (data2)
# setup plot details
#colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
#lw = 2

#n_classes = y.shape[1]
#print (Ｘ)
# Add noisy features
random_state = np.random.RandomState(0)
#n_samples, n_features = X.shape
#X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
#print (Ｘ)
# Split into training and test
sm = SMOTE(random_state=random_state)
X_train, X_test, y_train, y_test = train_test_split(data2, y, test_size=.25,
                                                    random_state=random_state, stratify=y)
#clf = GaussianNB()
X_res, y_res = sm.fit_sample(X_train, y_train)
X_res2, y_res2 = sm.fit_sample(X_test, y_test)
#clf = LinearSVC(C=0.01, penalty="l1", dual=False, class_weight="auto")
print(len(X_res))
print(len(y_res))
clf = GaussianNB()
#clf = AdaBoostClassifier()
clf.fit(X_res,y_res)
#clf.fit(X_train,y_train)
#clf = KNeighborsClassifier(n_neighbors=3)
#clf = AdaBoostClassifier()
#clf.fit(X_train,y_train)
#y_fitted = clf.predict_proba(X_res2)



#print(pro)
#cross_val_score(clf, data2, y)
#model = SVR(cache_size=7000)

#y_score = clf.fit(X_train, y_train).decision_function(X_test)
#svc.fit(X_train, y_train) 
#predicts = clf.predict(X_test)
predicted = cross_val_predict(clf, X_res2, y_res2, cv=10)
#target_names = ['class 0', 'class 1']
#print(len(y_test))
print(classification_report(y_res2, predicted, target_names=target_names))
#count = len(["ok" for idx, label in enumerate(y_test) if label == predicts[idx]])
#print ("Accuracy Rate, which is calculated manually is: %f" % (float(count) / len(y_test)))


# In[ ]:



