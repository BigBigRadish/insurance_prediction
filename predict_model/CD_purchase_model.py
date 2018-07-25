# -*- coding: utf-8 -*-
'''
Created on 2018年7月24日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
import pandas as pd
import pydot
import numpy as np
from pyspark.mllib.tree import DecisionTree
from sklearn.model_selection import cross_val_score
from xgboost.sklearn import XGBClassifier
no_survey=pd.read_csv('../no_survey_3.csv')
no_survey=no_survey.drop(columns=['1rigion','8worry_risk','CM_insurance_conditiones','CM_insurance_notworry','CN_insurance_conditiones','CN_insurance_notworry','CD_insurance_conditiones','CD_insurance_notworry'])
print(no_survey)
no_survey=no_survey.astype(float)
from sklearn import tree
from sklearn.cross_validation import train_test_split  #这里是引用了交叉验证 
FeatureSet=no_survey.drop(columns=['CN_purchase'])
featureName=FeatureSet.columns.values.tolist()
print(featureName)
Label=no_survey['CN_purchase']
X_train,X_test, y_train, y_test = train_test_split(FeatureSet, Label, random_state=1)#将数据随机分成训练集和测试集
print (X_train)
print (X_test)
print (y_train)
print (y_test)
#print iris
clf = tree.DecisionTreeClassifier()#DecisionTree
clf = clf.fit(X_train, y_train)
# # from sklearn.externals.six import StringIO
# # with open("isBuy.dot", 'w') as f:
# #     f = tree.export_graphviz(clf, out_file=f)
import os
os.environ['PATH'] = os.environ['PATH'] + (';E:\\graphviz-2.38\\release\\bin\\')
# # 
# # #  
# # from sklearn.externals.six import StringIO  
# #  #注意要安装pydot2这个python插件。否则会报错。
# # dot_data = StringIO() 
# # tree.export_graphviz(clf, out_file=dot_data) 
# # graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
# # graph[0].write_pdf("isBuy.pdf") #将决策树以pdf格式输出
# #   
# pre_labels=clf.predict(X_test)
# score=clf.score(X_test, y_test)#
# print ('DecisionTree prediction score:'+str(score))
# scores = cross_val_score(clf, X_test, y_test, cv=5)
# print(scores)
# with open("cd_model_score.txt", 'a') as f:
#     f.write("decision tree :"+str(scores)+'\n')
########################################################
# from sklearn import svm
# from sklearn import metrics
# clf = svm.SVC(kernel='rbf', C=2)
# clf = clf.fit(X_train, y_train)
# scores = cross_val_score(clf, X_test, y_test, cv=5)
# print("svm rbf :"+str(scores))
# with open("cd_model_score.txt", 'a') as f:
#     f.write("svm rbf c=2 :"+str(scores)+'\n')
############################################随机森林#######

from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation, metrics
import matplotlib.pylab as plt
from IPython.display import Image
# 
# #不管任何参数，都用默认的，拟合下数据看看
# rf0 = RandomForestClassifier(random_state=10)
# rf0.fit(X_train,y_train)
# scores = cross_val_score(rf0, X_test, y_test, cv=5)
# with open("cd_model_score.txt", 'a') as f:
#     f.write("rf random  :"+str(scores)+'\n')
#首先对n_estimators进行网格搜索
# param_test1= {'n_estimators':list(range(10,71,5))}
# gsearch1= GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100,
#                                  min_samples_leaf=20,max_depth=8,max_features='sqrt' ,random_state=10),
#                         param_grid =param_test1, scoring='accuracy',cv=5)
# gsearch1.fit(X_train,y_train)
# print(gsearch1.grid_scores_,gsearch1.best_params_, gsearch1.best_score_,'\n')
# #n_estimators': 65效果比较好
# param_test2= {'max_depth':list(range(3,14,2)), 'min_samples_split':list(range(50,201,20))}
# gsearch2= GridSearchCV(estimator = RandomForestClassifier(n_estimators= 65,
#                                  min_samples_leaf=20,max_features='sqrt' ,oob_score=True,random_state=10),
#    param_grid = param_test2,scoring='accuracy',iid=False, cv=5)
# gsearch2.fit(X_train,y_train)
# print(gsearch2.grid_scores_,gsearch2.best_params_, gsearch2.best_score_)
# #{'max_depth': 11, 'min_samples_split': 50} 0.7594296748847198
#再对内部节点再划分所需最小样本数min_samples_split和叶子节点最少样本数min_samples_leaf一起调参
# param_test3= {'min_samples_split':list(range(10,150,20)), 'min_samples_leaf':list(range(10,60,10))}
# gsearch3= GridSearchCV(estimator = RandomForestClassifier(n_estimators= 65,max_depth=11,
#                                  max_features='sqrt' ,oob_score=True, random_state=10),
#    param_grid = param_test3,scoring='accuracy',iid=False, cv=5)
# gsearch3.fit(X_train,y_train)
# print(gsearch3.grid_scores_,gsearch3.best_params_, gsearch3.best_score_)
# #{'min_samples_split': 10, 'min_samples_leaf': 10} 0.7627227268245397


# param_test4= {'max_features':list(range(3,11,2))}
# gsearch4= GridSearchCV(estimator = RandomForestClassifier(n_estimators= 65,max_depth=11, min_samples_split=10,
#                                  min_samples_leaf=10 ,oob_score=True, random_state=10),
#    param_grid = param_test4,scoring='accuracy',iid=False, cv=5)
# gsearch4.fit(X_train,y_train)
# print(gsearch4.grid_scores_,gsearch4.best_params_, gsearch4.best_score_)
# #{'max_features': 9} 0.7632950751866703

estimator = RandomForestClassifier(n_estimators= 65,max_depth=11, min_samples_split=10,
                                 min_samples_leaf=10 ,max_features=9,oob_score=True, random_state=10)
estimator.fit(X_train,y_train)
scores = cross_val_score(estimator, X_test, y_test, cv=5)
# print(scores)
# with open("cd_model_score.txt", 'a') as f:
#       f.write("rf random optimaor  :"+str(scores)+'\n')
###############################################xgb#################################
# xgb=XGBClassifier(learning_rate =0.1,
#  n_estimators=1000,
#  max_depth=11,
#  min_child_weight=1,
#  gamma=0,
#  subsample=0.8,
#  colsample_bytree=0.8,
#  objective= 'binary:logistic',
#  nthread=4,
#  scale_pos_weight=1,
#  seed=27)
# xgb.fit(X_train,y_train)
# scores = cross_val_score(xgb, X_test, y_test, cv=5)
# print(scores)
# with open("cd_model_score.txt", 'a') as f:
#       f.write("xgb 1  :"+str(scores)+'\n')
#效果一般 就用随机森林了
#存一哈65个随机森林结构
# Estimators = estimator.estimators_
# for index, model in enumerate(Estimators):
#     filename = 'iris_' + str(index) + '.pdf'
#     dot_data = tree.export_graphviz(model , out_file=None,
#                          feature_names=featureName,
#                          class_names='CM_purchase',
#                          filled=True, rounded=True,
#                          special_characters=True)
#     graph = pydot.graph_from_dot_data(dot_data)
#     # 使用ipython的终端jupyter notebook显示。
#     #Image(graph.create_png())
#     graph[0].write_pdf(filename)
############特征重要度可视化#######################
y_importances = estimator.feature_importances_
x_importances = featureName
y_pos = np.arange(len(x_importances))
# 横向柱状图
plt.barh(y_pos, y_importances, align='center')
plt.yticks(y_pos, x_importances)
plt.xlabel('Importances')
plt.xlim(0,1)
plt.title('Features Importances')
plt.show()
 
# 竖向柱状图
plt.bar(y_pos, y_importances, width=0.4, align='center', alpha=0.4)
plt.xticks(y_pos, x_importances)
plt.ylabel('Importances')
plt.ylim(0,1)
plt.title('Features Importances')
plt.show()