# -*- coding: utf-8 -*-
'''
Created on 2018年7月23日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
import pandas as pd
from pandas import DataFrame
import numpy as np
import math
import scipy.stats
#分析每个特征的相关性
from numpy import linalg as la  
  
#欧式距离  
def euclidSimilar(inA,inB):  
    return 1.0/(1.0+la.norm(inA-inB))  
#皮尔逊相关系数  
def pearsonSimilar(inA,inB):  #pearson相似系数
    if len(inA)<3:  
        return 1.0  
    return 0.5+0.5*np.corrcoef(inA,inB,rowvar=0)[0][1]  
#余弦相似度  
def cosSimilar(inA,inB):  #余弦相似性
    inA=np.mat(inA)  
    inB=np.mat(inB)  
    num=float(inA*inB.T)  
    denom=la.norm(inA)*la.norm(inB)  
    return 0.5+0.5*(num/denom)  
def computecorrelation(x,y):#kl散度相似系数
    x_bar=np.mean(x)
    y_bar=np.mean(y)
    SSR=0
    Varx=0
    Vary=0
    for i in range(0,len(x)):
        SSR+=(x[i]-x_bar)*(y[i]-y_bar)
        Varx+=(x[i]-x_bar)**2
        Vary+=(y[i]-y_bar)**2
    SST=math.sqrt(Varx*Vary)
    return SSR/SST
if __name__ == "__main__":
    no_survey=pd.read_csv('../no_survey_3.csv')
    no_survey=no_survey.drop(columns=['1rigion','8worry_risk','CM_insurance_conditiones','CM_insurance_notworry','CN_insurance_conditiones','CN_insurance_notworry','CD_insurance_conditiones','CD_insurance_notworry'])
    print(no_survey)
    no_survey=no_survey.astype(float)
    #labels=list(no_survey)
    #print(labels[0:])
    #y=list(no_survey['CM_purchase'])
#     for i in labels:
#         x=list(no_survey[str(i)])
#         print(i+'与CM_purchase Kl散度：')
#         f=open('../cos_similar.txt','a',encoding='utf-8')#存入文件
#         similar=pearsonSimilar(x, y)
#         #similar=computecorrelation(x,y)
#         cos=cosSimilar(x,y)
#         KL = scipy.stats.entropy(x, y) 
#         print(cos)
#         f.write(str(i+'与CM_purchase cosin 相关度：'+str(cos))+'\n')
