# -*- coding: utf-8 -*-
"""
Load libraries
"""

from logging import warning
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import warnings
import pickle
warnings.filterwarnings("ignore")


#load dataset
raw_mail_data=pd.read_csv('spam.csv',encoding="ISO-8859-1")

raw_mail_data.columns

raw_mail_data.head()

mail_data=raw_mail_data.where((pd.notnull(raw_mail_data)),' ')

mail_data.shape

#sample data
mail_data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1).head(5)

mail_data.loc[mail_data['v1']=='spam','v1',]=0
mail_data.loc[mail_data['v1']=='ham','v1',]=1

#seperate the data as feature and target X=> feature(v2)  Y=> target(v1)
X=mail_data['v2']
Y=mail_data['v1']

#split into train and test
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=3)

"""Feature Extraction"""

#tranform the data to feature vector so that it can be used to give data to dvm 
#convert the text to lower case
count_vect=CountVectorizer()
feature_extraction=TfidfVectorizer(min_df=1,stop_words='english',lowercase='True')

X_train_feature=feature_extraction.fit_transform(x_train)
X_test_feature=feature_extraction.transform(x_test)

X_train_feature.shape

X_test_feature.shape

#convert y values to int
y_train=y_train.astype('int')
y_test=y_test.astype('int')

"""Training the model --> **Suport Vector Machine**"""

#train machine
model=LinearSVC()
model.fit(X_train_feature,y_train)

#save the vectoriser

pickle.dump(feature_extraction,open('vectoriser.pickle','wb'))
loaded_vect=pickle.load(open('vectoriser.pickle','rb'))

#save the model
pickle.dump(model,open('model.pkl','wb'))
pred_model=pickle.load(open('model.pkl','rb'))


'''
#performace of model on data which has it has seen before

train_data_predict=model.predict(X_train_feature)
train_data_accuracy=accuracy_score(y_train,train_data_predict)

print(train_data_accuracy)

#test data accuracy
test_data_predict=model.predict(X_test_feature)
test_data_accuracy=accuracy_score(y_test,test_data_predict)

test_data_accuracy

"""Predict weather mail is sapm or ham"""

mail_input=[input("Enter your mail:")]
mail_input_feature=feature_extraction.transform(mail_input)
ans = model.predict(mail_input_feature)

if ans[0]==1:
  print('ham')
else:
  print('spam')'''