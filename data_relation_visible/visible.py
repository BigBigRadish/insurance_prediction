# -*- coding: utf-8 -*-
'''
Created on 2018年7月23日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
# 首先载入pandas
import pandas as pd
import relativity
# 我们将载入seaborn,但是因为载入时会有警告出现，因此先载入warnings，忽略警告
import warnings 
#from data_relation_visible.relativity import no_survey
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)
#  pairplot显示不同特征之间的关系
no_survey=pd.read_csv('../no_survey_3.csv')
no_survey=no_survey.drop(columns=['1rigion','8worry_risk','CM_insurance_conditiones','CM_insurance_notworry','CN_insurance_conditiones','CN_insurance_notworry','CD_insurance_conditiones','CD_insurance_notworry'])
print(no_survey)
no_survey=no_survey.astype(float)
plt.show(sns.pairplot(no_survey, hue="CM_purchase", size=3))