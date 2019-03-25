#!/anaconda3/bin/python

import pandas as pd
from scipy.io import arff

import plotly
plotly.tools.set_credentials_file(username='artemii-yanushevskyi', api_key='aRmQfG7U4SAlhISYVym7')

import plotly.plotly as py
import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
from IPython.display import Image
import plotly.io as pio

import numpy as np
import seaborn as sns

from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
colormap = plt.cm.RdBu

exporting = True

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from sklearn.preprocessing import OneHotEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

display = print

# In[2]:


data, meta = arff.loadarff('cworkTrain.arff')
df = pd.DataFrame(data)

# convert attribute values with type "object" to regular strings
objects_df = df.select_dtypes([object]) # select only atributes of object type
stack_df = objects_df.stack() # means create one column
decoded_stack_df = stack_df.str.decode('utf-8') # decode the values in the column
decoded_objects_df = decoded_stack_df.unstack() # separate into columns

# replace in df
for col in decoded_objects_df.columns:
    df[col] = decoded_objects_df[col]

# replace month with a value
df['month'] = pd.to_datetime(df.month, format='%b').dt.month


# In[3]:


df['termDeposit'] = df['termDeposit'].apply(lambda x: 0 if x == 'no' else 1)
df_features = df.drop(columns='termDeposit')
df_categorical = df_features.select_dtypes([object])
df_numerical = df_features.select_dtypes([int, float])
# create dummy variables for df_categorical

df_one_hot = pd.concat([df_numerical, df['termDeposit']], axis=1)
catergorical_attributes = dict()
for atr in df_categorical.columns:
    df_dummies = pd.get_dummies(df[atr], prefix = atr)
    catergorical_attributes[atr] = [col.split('_')[1] for col in df_dummies.columns]
    df_one_hot = pd.concat([df_one_hot, df_dummies], axis=1) # the dataset ready to appy decision tree algorithm

display(df_one_hot.head())


# In[4]:


attribute_order_one_hot = []
for i in range(len(df.columns)):
    if df.columns[i] in catergorical_attributes.keys():
        print(i, df.columns[i])
        categories = [df.columns[i] + '_' + cat for cat in catergorical_attributes[df.columns[i]]]
        attribute_order_one_hot.extend(categories)
    else:
        attribute_order_one_hot.append(df.columns[i])
        
df_one_hot_ordered = df_one_hot[attribute_order_one_hot]
df_one_hot_ordered.head()


# In[5]:


df_shuffle = shuffle(df_one_hot_ordered)
X_train, X_test, y_train, y_test = train_test_split(df_shuffle.drop('termDeposit', axis=1), df_shuffle['termDeposit'], test_size=0.33, random_state=45)
display(df_shuffle.head(), X_train.shape)


# # Keras

# In[6]:


import tensorflow as tf
X_train_norm = tf.keras.utils.normalize(X_train.values, axis=1)
X_test_norm = tf.keras.utils.normalize(X_test.values, axis=1)


# In[7]:


model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:



model.fit(X_train_norm, y_train.values, epochs=3, batch_size=10)

val_loss, val_acc = model.evaluate(X_test_norm, y_test)
print(val_loss)
print(val_acc)


