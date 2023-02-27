
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 14:14:59 2022

@author: Mohsin Sarker Raihan
"""

#%%                           Import Library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.metrics import cohen_kappa_score

#%%                         Import Dataset
dataset = pd.read_csv('data.csv', encoding= 'unicode_escape')
#print(dataset.head(5))
#dataset = shuffle(dataset)
print("Data shape:",dataset.shape)

# iterating the columns 
for col in dataset.columns: 
    print(col)
    
#%% Missing value

from sklearn.impute import KNNImputer
imputer=KNNImputer(n_neighbors=5)
dataset= pd.DataFrame(imputer.fit_transform(dataset),columns=dataset.columns)

#%%                    Create X and Y variables 

X=dataset[['cough', 'breathlessness', 'headache', 'mild_fever',
       'throat_irritation', 'runny_nose', 'sinus_pressure',
       'chest_pain', 'blood_in_sputum']]



y=dataset['prognosis']


#%%                        Feature Correlation

import seaborn as sn
import matplotlib.pyplot as plt
sn.set(font_scale=1.8)
sn.set_style("darkgrid")
fig_dims = (20, 12)
fig, ax = plt.subplots(figsize=fig_dims)
sn.heatmap(dataset.corr(),annot=True, ax=ax)
plt.show()




#%%               Spliting the dataset into Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#%%                               Feature Scaling
'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''

#%%-------------------------- Train The Model------------------------


#%%                      Initialize the model

# ANN
#%% Functions

# Plot 
def plott_ann(train,val,x_test,x_val,x):
    import seaborn as sn
    import matplotlib.pyplot as plt
    
    sn.set(font_scale=4)
    plt.rcParams['figure.figsize']=20,10
    sn.set_style("whitegrid")
    plt.plot(train,linewidth=5, label  = x_test)
    plt.plot(val, linewidth=5,  label  = x_val)
    plt.xlabel('epoch',fontsize = 50)
    plt.ylabel(x,fontsize = 50 )
    plt.grid(True)
    plt.legend()
    #plt.title("Model Performance", fontsize = 20)
    plt.rc('xtick', labelsize=30) 
    plt.rc('ytick', labelsize=30)
    plt.legend(loc=5, prop={'size': 30})
    plt.show()

# model score function 
def model_score (y_test,y_pred):
    # Statistical Score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    st = 'macro'  # 'micro',  'macro' , 'weighted' None
    ac  = accuracy_score(y_test, y_pred)  # for multy class
    pre = precision_score(y_test, y_pred, average = st) # use [accuracy_score(y_test, y_pred, average = st)]
    re  = recall_score(y_test, y_pred, average = st)
    f1  = f1_score(y_test, y_pred, average = st) 
    results = [ac,pre,re, f1]
    return results

# print function 
def print_summary(ac,pr,re,f):
    print("Accuracy    =", ac)
    print("Precision   =", pr)
    print("Recall      =", re)
    print("F1 Score    =", f)
    
                                                              
#%%  Importing the Keras libraries and packages

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

#%%                         Create the Model

# Adding the input layer and the first hidden layer
classifier.add(Dense( 30, input_dim = 9,  activation = 'relu' ))

# # Adding the input layer and the first hidden layer
classifier.add(Dense( 12, input_dim = 30,  activation = 'relu' ))

# # Adding the second hidden layer
# classifier.add(Dense( 16, input_dim = 32, activation = 'relu'))

# # Adding the input layer and the first hidden layer
# classifier.add(Dense( 8, input_dim = 16,  activation = 'relu' ))

# # Adding the input layer and the first hidden layer
# classifier.add(Dense( 6, input_dim = 3,  activation = 'relu' ))

# # Adding the input layer and the first hidden layer
# classifier.add(Dense( 12, input_dim = 6,  activation = 'relu' ))

# # Adding the input layer and the first hidden layer
# classifier.add(Dense( 18, input_dim = 12,  activation = 'relu' ))

# Adding the output layer
classifier.add(Dense(6, input_dim = 12, activation = 'softmax'))

#%%                      Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#%%                     Let's have a Fun 

batch_size= 10
epochs = 100

history = classifier.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test, y_test))

# Evaluate the performance of our trained model
scores = classifier.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


#%%              results section 

#model_score
history_dict = history.history
train_acc  = history_dict['accuracy']
train_loss = history_dict['loss']
val_acc    = history_dict['val_accuracy']
val_loss   = history_dict['val_loss']

#%%  Plot section 

plott_ann(train_acc,val_acc,'Training Accuracy','Validation Accuracy','Accuracy')
plott_ann(train_loss,val_loss,'Training loss','Validation loss','Loss')
plott_ann(val_acc,val_loss,'Validation Accuracy','Valodation Loss','Score')
plott_ann(train_acc,train_loss,'Training Accuracy','Training Loss','Score')

#%%    Model performance 

# Part 3 - Making the predictions and evaluating the model

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
y_pred = classifier.predict(X_test) 
predictions = np.argmax(y_pred, axis=-1) 

label_encoder = LabelEncoder().fit(y_test)
label_y = label_encoder.transform(y_test)

cm = confusion_matrix(label_y, predictions)
print(cm)

from sklearn.metrics import classification_report
print(classification_report(label_y,predictions))



#%% plot ANN configuration

from eiffel2 import builder
# python -m pip install eiffel2  

builder([9, 30, 12, 6])
# or the following if you want to have a dark theme
#builder([14, 28, 12, 6], bmode="night")


#%% Plot results Summary

plott_ann(train_acc,val_acc,'Training_Accuracy','Validation_Accuracy','Accuracy')
plott_ann(train_loss,val_loss,'Training_loss','Validation_loss','Loss')
plott_ann(val_acc,val_loss,'Validation Accuracy','Valodation Loss','Score')
plott_ann(train_acc,train_loss,'Training_Accuracy','Training_Loss','Score')

#%% plot all results

import seaborn as sn
import matplotlib.pyplot as plt

sn.set(font_scale=1)
plt.rcParams['figure.figsize']=20,10
sn.set_style("darkgrid")
plt.plot(train_acc, linewidth=4, label  = 'Train Accuracy')
plt.plot(train_loss,linewidth=4, label  = 'Train Loss')
plt.plot(val_acc   ,linewidth=4, label  = 'Validation Accuracy')
plt.plot(val_loss  ,linewidth=4, label  = 'Validation Loss')
plt.ylim(0, 1.05)

plt.xlabel('epoch',fontsize = 20)
plt.ylabel('Score',fontsize = 20 )
plt.grid(True)
plt.legend()
plt.title("Model Performance", fontsize = 20)
plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20)
plt.legend(loc=5, prop={'size': 20})
plt.show()

#%% End - Happy Coding