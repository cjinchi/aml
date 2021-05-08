import numpy
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# X, y = load_iris(return_X_y=True)
# clf = LogisticRegression(solver="liblinear").fit(X, y)
# y2 = np.argmax(clf.predict_proba(X),axis=1)
# print(roc_auc_score(y, clf.predict_proba(X), multi_class='ovr'))
# print(X)
# print(y2)
# print(y)
# x = np.argmax(X)
# print(X)

# print(np.argmax(X,axis=1))
# print(x)
# print(y)
# print(y2)
# print(accuracy_score(y, y2))


# real = np.asarray([random.choice([0,1,2]) for _ in range(50)])
# predict = np.asarray([random.choice([0,1,2]) for _ in range(50)])
# print(real)
# print(predict)
#
# print(roc_auc_score(real, predict,multi_class='ovr'))
#
# # unique, counts = numpy.unique(real, return_counts=True)
# # print(unique)
# # print(counts)
# # print(1/(counts[0]*counts[1]))
# num = 0
# total = 0
# for i in range(len(real)):
#     for j in range(i + 1, len(real)):
#         if real[i] + real[j] == 1:
#             total += 1
#         if real[i] == 1 and real[j] == 0 and predict[i] > predict[j]:
#             num += 1
#         elif real[i] == 0 and real[j] == 1 and predict[i] < predict[j]:
#             num += 1
#         elif real[i]+real[j]==1 and predict[i]==predict[j]:
#             num+=0.5
# print(num / total)
# print(num,total)

# from sklearn.datasets import load_iris
# X, y = load_iris(return_X_y=True)
# clf = LogisticRegression(solver="liblinear").fit(X, y)
# y2 = clf.predict_proba(X)
# roc_auc_score(y, y2, multi_class='ovr')
#
# print(y)
# print(y2)


# y = np.zeros((5,3))
# x = np.asarray([0,2,1,2,0])
#
# print(y)

def to_one_hot(a):
    b = np.zeros((a.size, 3))
    b[np.arange(a.size), a] = 1
    return b


def my_auc(real,predict):
    total,num = 0,0
    for i in range(len(real)):
        for j in range(i + 1, len(real)):
            if real[i] + real[j] == 1:
                total += 1
            if real[i] == 1 and real[j] == 0 and predict[i] > predict[j]:
                num += 1
            elif real[i] == 0 and real[j] == 1 and predict[i] < predict[j]:
                num += 1
            elif real[i] + real[j] == 1 and predict[i] == predict[j]:
                num += 0.5
    return num/total

real = np.asarray([random.choice([0, 1, 2]) for _ in range(50)])
predict = np.asarray([random.choice([0, 1, 2]) for _ in range(50)])
print(real)
print(predict)
predict_one_hot = to_one_hot(predict)

print(roc_auc_score(real, predict_one_hot, multi_class='ovr'))

total_auc = 0
for i in range(3):
    one_real = np.asarray(real==i).astype(int)
    one_predict = np.asarray(predict==i).astype(int)
    total_auc+=my_auc(one_real,one_predict)
print(total_auc/3)