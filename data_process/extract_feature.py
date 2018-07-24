# -*- coding: utf-8 -*-
'''
Created on 2018年7月21日

@author: Zhukun Luo
Jiangxi university of finance and economics
'''
from pandas import DataFrame
from pandas import concat
import pandas as pd
import re
import jieba,math
import jieba.analyse
'''''''''''''''地区异常值处理''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
no_survey=pd.read_csv('../no_survey.csv')
print(no_survey['1调查地区'])
#print(no_survey)
rigion=[]
j=1
for i in no_survey['1调查地区']:
    if re.findall('(.{1,3})-', str(i)) :
        province=re.findall('(.{2,3})-', str(i))
    else :
        province=[]

    if province==[]:
        rigion.append('')
    else:
        rigion.append(province[0])

no_survey['1调查地区']=rigion
no_survey.to_csv('../no_survey_1.csv')
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''担心的未来问题数据处理'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#将其未来担心的问题划分为收入、开销、居住环境、家庭关系、心理情绪、疾病、自理或活动能力下降，其他，无，作为特征
#处理这些文本数据的步骤：构建每一类特征的词典，然后进行筛选
no_survey=pd.read_csv('../no_survey_2.csv')
# print(no_survey['1调查地区'].isnull().sum())
# no_survey=no_survey.dropna()#去除含有空值的行
# print(no_survey['1调查地区'].isnull().sum())
#no_survey.to_csv('../no_survey_1.csv')
# income=[]
# 
# with open('../walk_ability.txt', mode='r', encoding='UTF-8') as f:
#     lines = f.read()
# print(lines)
# lines=str(lines[0]).split(',')
# for i in lines:
#     print(i)
# for j in no_survey['8worry_risk']:
#     flag='0'
#     for i in lines:
#         print(i)
#         if str(i) in str(j):
#             flag='1'
#             break
#     income.append(flag)
# print(income)
# no_survey['8income'] =income 
# no_survey.to_csv('../no_survey_2.csv')#插入income 
##########################################
# lines=str(lines).split(',')
# for i in lines:
#     print(i)
# for j in no_survey['8worry_risk']:
#     flag='0'
#     for i in lines:
#         print(i)
#         if str(i) in str(j):
#             flag='1'
#             break
#     income.append(flag)
# print(income)
# no_survey['8cost'] =income 
# no_survey.to_csv('../no_survey_2.csv')#插入cost  
#######################################################   
# lines=str(lines).split(',')
# for i in lines:
#     print(i)
# for j in no_survey['8worry_risk']:
#     flag='0'
#     for i in lines:
#         print(i)
#         if str(i) in str(j):
#             flag='1'
#             break
#     income.append(flag)
# print(income)
# no_survey['8live_envr'] =income 
# no_survey.to_csv('../no_survey_2.csv')#插入live_envr 
#################################################
# lines=str(lines).split(',')
# for i in lines:
#     print(i)
# for j in no_survey['8worry_risk']:
#     flag='0'
#     for i in lines:
#         print(i)
#         if str(i) in str(j):
#             flag='1'
#             break
#     income.append(flag)
# print(income)
# no_survey['8family'] =income 
# no_survey.to_csv('../no_survey_2.csv')#插入family
#####################################################
# lines=str(lines).split(',')
# for i in lines:
#     print(i)
# for j in no_survey['8worry_risk']:
#     flag='0'
#     for i in lines:
#         print(i)
#         if str(i) in str(j):
#             flag='1'
#             break
#     income.append(flag)
# print(income)
# no_survey['8moody'] =income 
# no_survey.to_csv('../no_survey_2.csv')#插入moody
######################################################
# lines=str(lines).split(',')
# for i in lines:
#     print(i)
# for j in no_survey['8worry_risk']:
#     flag='0'
#     for i in lines:
#         print(i)
#         if str(i) in str(j):
#             flag='1'
#             break
#     income.append(flag)
# print(income)
# no_survey['8desase'] =income 
# no_survey.to_csv('../no_survey_2.csv')#插入desase
# lines=str(lines).split(',')
# for i in lines:
#     print(i)
# for j in no_survey['8worry_risk']:
#     flag='0'
#     for i in lines:
#         print(i)
#         if str(i) in str(j):
#             flag='1'
#             break
#     income.append(flag)
# print(income)
# no_survey['8walk_ability'] =income 
# no_survey.to_csv('../no_survey_2.csv')#插入walk_ability
#######################################################
#对所有的风险文本进行分词
# def seg_sentence(sentence):  #分词函数
#     sentence_seged = jieba.cut(sentence.strip(),cut_all = False)  #jieba分词
#     stopwords = stopwordslist('../哈工大停用词表.txt')  # 这里加载停用词的路径  
#     outstr = ''
#     word_freq = {}  
#     for ele in sentence_seged:  #统计词频
#         if ele not in stopwords:  
#             if ele in word_freq:
#                 word_freq[ele] += 1
#             else:
#                 word_freq[ele] = 1
#     freq_word = []
#     for ele, freq in word_freq.items():
#         freq_word.append((ele,freq))
#     freq_word.sort(key = lambda x: x[1], reverse = True)
#     for ele, freq in freq_word:
# 
#         word=ele;
#         
#         freq=freq;
#         worry_risk = str(word)+','+str(freq)
#         print(worry_risk)
#         f=open('../CD_insurance_notworry.txt','a',encoding='utf-8')#存入文件
#         f.write(str(worry_risk)+'\n')
# def stopwordslist(filepath):  
#     stopwords = [line.strip() for line in open(filepath, 'rb').readlines()]  
#     return stopwords     
# ''''''''
# if __name__ == "__main__":
#     i=''
#     for j in no_survey['CD_insurance_notworry']:
#         if str(j)=='-3':
#             continue
#         else:
#             i+=str(j)+' '
#     print(i)
#     i=re.sub("[\s+\.\!\/_,$%^*(+\"\'：]+|[+——！，。？、~@#￥%……&*（）:】-一’()【-]+", " ",i) 
#     seg_sentence(i)
########################################################################################
#对省份进行数值化
province=no_survey['1rigion'].drop_duplicates()
print(province)
list={}
j=0
for i in province:
    list[str(i)]=str(j)
    j+=1
print(list)
list1=[] 
for i in  no_survey['1rigion']: 
    for k ,v in list.items():
    #print(k,v)  
        if str(i)==str(k):
            list1.append(v)
            break
        else:
            continue
print(list1)
no_survey['province']=list1
no_survey.to_csv('../no_survey_3.csv')
         