import numpy as np 
import pandas as pd
import os
import csv
import FitAndPredictNews as fpn
import FitAndPredictIMDB as fpimdb
import re
from sklearn.datasets import fetch_20newsgroups
import NewsCharts as chartnews
import IMDBCharts as chartimdb

#------------------------------------------------------------------------------------------------------------------------------#

#import news_groups data
#categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
news_groups_train = fetch_20newsgroups(subset='train', shuffle=True, remove=['headers', 'footers', 'quotes'])#, categories=categories)
news_groups_test = fetch_20newsgroups(subset='test', shuffle=True, remove=['headers', 'footers', 'quotes'])##, categories=categories)
print("-----------------------------\nNEWS GROUP DATA\n-----------------------------")

#get diagrams
#graphs = chartnews.NewsCharts(news_groups_train, news_groups_test)
#graphs.__run__()

#run methods
news = fpn.FitAndPredictNews(news_groups_train, news_groups_test)
news.__run__()

#------------------------------------------------------------------------------------------------------------------------------#

#import IMDB data to csv
#train data
dirpath_neg = os.path.dirname(os.getcwd())+'/IMDB Data/train/neg'
dirpath_pos = os.path.dirname(os.getcwd())+'/IMDB Data/train/pos'
output = 'train_data.csv'
with open(output, 'w') as outfile:
    csvout = csv.writer(outfile)

    files = os.listdir(dirpath_neg)

    for filename in files:
        with open(dirpath_neg + '/' + filename) as afile:
            csvout.writerow(['neg', afile.read()])
            afile.close()

    files = os.listdir(dirpath_pos)

    for filename in files:
        with open(dirpath_pos + '/' + filename) as afile:
            csvout.writerow(['pos', afile.read()])
            afile.close()
       
    outfile.close()

#test data
dirpath_neg = os.path.dirname(os.getcwd())+'/IMDB Data/test/neg'
dirpath_pos = os.path.dirname(os.getcwd())+'/IMDB Data/test/pos'
output = 'test_data.csv'
with open(output, 'w') as outfile:
    csvout = csv.writer(outfile)

    files = os.listdir(dirpath_neg)

    for filename in files:
        with open(dirpath_neg + '/' + filename) as afile:
            csvout.writerow(['neg', afile.read()])
            afile.close()

    files = os.listdir(dirpath_pos)

    for filename in files:
        with open(dirpath_pos + '/' + filename) as afile:
            csvout.writerow(['pos', afile.read()])
            afile.close()
       
    outfile.close()


#import to dataframe
col_names = ['Class', 'Content']
imdb_train = pd.read_csv('train_data.csv', header=None, names=col_names)
imdb_test = pd.read_csv('test_data.csv', header=None, names=col_names)

#run methods
print("-----------------------------\nIMDB DATA\n-----------------------------")

imdb = fpimdb.FitAndPredictIMDB(imdb_train, imdb_test)
imdb.__run__()

#get diagrams
#graphs2 = chartimdb.IMDBCharts(imdb_train, imdb_test)
#graphs2.__run__()

#------------------------------------------------------------------------------------------------------------------------------#
