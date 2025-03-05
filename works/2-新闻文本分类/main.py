import time

t0 = time.time()
print('程序开始的时间:', time.strftime('%H:%M:%S', time.localtime(time.time())))

import numpy as np
from sklearn.datasets import fetch_20newsgroups  # 获取数据集
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF文本特征提取
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report

# 平时常用的一些分类方法
from sklearn.naive_bayes import MultinomialNB  # 朴素贝叶斯
from sklearn.neighbors import KNeighborsClassifier  # K近邻
from sklearn.linear_model import SGDClassifier  # 随机梯度下降(适合稀疏矩阵)
from sklearn.tree import DecisionTreeClassifier  # 决策树conda
from sklearn.linear_model import LogisticRegression  # 逻辑回归
from sklearn.ensemble import RandomForestClassifier  # 随机森林
from sklearn.ensemble import AdaBoostClassifier  # 平时就是叫AdaBoost
from sklearn.ensemble import GradientBoostingClassifier  # 梯度提升决策树（慢）
from sklearn.neural_network import MLPClassifier  # 多层感知器(慢)
from sklearn.svm import SVC  # 支持向量机（慢）

# #选取20个类中7种比较典型的类别进行实验
# select = ['alt.atheism',
#  'comp.graphics',
#  'comp.os.ms-windows.misc',
#  'comp.sys.ibm.pc.hardware',
#  'comp.sys.mac.hardware',
#  'comp.windows.x',
#  'misc.forsale',
#  'rec.autos',
#  'rec.motorcycles',
#  'rec.sport.baseball',
#  'rec.sport.hockey',
#  'sci.crypt',
#  'sci.electronics', 
#  'sci.med',
#  'sci.space',
#  'soc.religion.christian',
#  'talk.politics.guns',
#  'talk.politics.mideast',
#  'talk.politics.misc',
#  'talk.religion.misc']
#
# train=fetch_20newsgroups(subset='train',categories=select)
# test=fetch_20newsgroups(subset='test',categories=select)

train = fetch_20newsgroups(subset='train')
test = fetch_20newsgroups(subset='test')

# 将文章数据向量化（TF-IDF算法）
vectorizer = TfidfVectorizer()
train_v = vectorizer.fit_transform(train.data)
test_v = vectorizer.transform(test.data)

# Classifier = [MultinomialNB(),DecisionTreeClassifier(),KNeighborsClassifier(),
#               LogisticRegression(),SGDClassifier(),RandomForestClassifier()]
Classifier_str = ['MultinomialNB()',  'SGDClassifier()']
# options:'MultinomialNB()'   'DecisionTreeClassifier()'       'KNeighborsClassifier()'         'LogisticRegression()'   'MLPClassifier()'
#         'SGDClassifier()'   'RandomForestClassifier()'       'GradientBoostingClassifier()'   'AdaBoostClassifier()'   'SVC()'
for i in Classifier_str:
    t2 = time.time()
    model = eval(i)
    model.fit(train_v, train.target)
    # 初始化五折交叉验证
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 用于存储预测结果
    y_pred = np.zeros(len(train.target))

    # 进行五折交叉验证
    for train_index, test_index in kf.split(train_v, train.target):
        X_train, X_test = train_v[train_index], train_v[test_index]
        y_train, y_test = train.target[train_index], train.target[test_index]

        model.fit(X_train, y_train)
        y_pred[test_index] = model.predict(X_test)

    # 打印分类报告和混淆矩阵
    print('\n\n' + i)
    print(i + "Classification Report:")
    print(classification_report(train.target, y_pred))

    print(i + "Confusion Matrix:")
    print(confusion_matrix(train.target, y_pred))
    print(i + "准确率为:", model.score(test_v, test.target))
    print(i + '用时:%.6fs' % (time.time() - t2))
    # y_predict=model.predict(test_v)
    # print(np.mean(y_predict==test.target))

t1 = time.time()
print('程序结束的时间:', time.strftime('%H:%M:%S', time.localtime(time.time())))
print("用时：%.2fs" % (t1 - t0))

