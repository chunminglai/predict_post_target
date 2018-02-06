print(__doc__)
%matplotlib inline

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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.cross_validation import cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn import tree
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
# import some data to play with
#iris = datasets.load_iris()
#X = iris.data
#y = iris.target
def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print "def tree({}):".format(", ".join(feature_names))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print "{}if {} <= {}:".format(indent, name, threshold)
            recurse(tree_.children_left[node], depth + 1)
            print "{}else:  # if {} > {}".format(indent, name, threshold)
            recurse(tree_.children_right[node], depth + 1)
        else:
            print "{}return {}".format(indent, tree_.value[node])

    recurse(0, 1)



data = pd.read_csv('post_feature_new.txt',header = None)
y1 = data[0].tolist()
y = np.array(y1)
#combine global and dynamic
#data = data.drop(data.columns[[0,1,2,5]], axis=1)
#global only
#data = data.drop(data.columns[[0,1,2,5,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]], axis=1)
#dynamic only
data = data.drop(data.columns[[0,1,2,3,4,5,6,7,8,21,22,23,24,25,26,27,28,29,30,31,32]], axis=1)
#data = data.ix[:,9:16]
#print (data)
data2 = data.as_matrix()


random_state = np.random.RandomState(0)
# Split into training and test
sm = SMOTE(random_state=random_state)
data2 = preprocessing.normalize(data2, norm='l2')
data2,y = sm.fit_sample(data2,y)
X_train, X_test, y_train, y_test = train_test_split(data2, y, test_size=.25,
                                                    random_state=random_state)
#clf = GaussianNB()
#X_res, y_res = sm.fit_sample(X_train, y_train)
#X_res2, y_res2 = sm.fit_sample(X_test, y_test)
#normalize
#clf1 = GaussianNB()
#clf2 = AdaBoostClassifier()
clf3 = tree.DecisionTreeClassifier()
#clf1.fit(X_train,y_train)
#clf2.fit(X_train,y_train)
clf3.fit(X_train,y_train)
#clf1 = LinearSVC(C=0.01, penalty="l1", dual=False, class_weight="auto")
#clf = KNeighborsClassifier(n_neighbors=3)
#clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
#predicts1 = clf1.predict(X_test)
#predicts2 = clf2.predict(X_test)
predicts3 = clf3.predict(X_test)
#print(classification_report(y_test, predicts1))
#print(classification_report(y_test, predicts2))
print(classification_report(y_test, predicts3))
#print(f1_score(y_test, predicts1))
#print(f1_score(y_test, predicts2))
#print(f1_score(y_test, predicts3))
#fig = plt.figure()
#fig.set_size_inches(6.5, 4.0)

###########################Draw 1####################################
#fpr = dict()
#tpr = dict()
#roc_auc = dict()
#fpr[0], tpr[0], _ = roc_curve(y_test, predicts1)
#print(_[0])
#roc_auc[0] = auc(fpr[0], tpr[0])

# Compute micro-average ROC curve and ROC area
#fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), predicts1.ravel())
#roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

#plt.figure()
#lw = 2
#plt.plot(fpr[0], tpr[0], 
         #lw=lw, label='Naive Baynes (area = %0.2f)' % roc_auc[0])
#plt.plot([0, 1], [0, 1], lw=lw, linestyle='--')
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic example')
#plt.legend(loc="lower right")
#plt.show()
###########################Draw 2####################################
