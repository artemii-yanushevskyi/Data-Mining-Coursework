import plotly.graph_objs as go
import plotly.io as pio
from IPython.display import Image
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

exporting = False

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

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def plotattributes(df, attributes=['balance', 'newbalance'], size=50):
    for atr in attributes:
        trace0 = go.Histogram(
            x=df[df['termDeposit'] == 0][atr],
            name='No subscribtion',
            xbins=dict(
                size=size,
            ),
            marker=dict(color='red'),
        )
        trace1 = go.Histogram(
            x=df[df['termDeposit'] == 1][atr],
            name='Subscribtion',
            xbins=dict(
                size=size,
            ),
            marker=dict(color='green'),
        )
        data = [trace0, trace1]
        layout = go.Layout(barmode='stack', title=atr.capitalize())
        fig = go.Figure(data=data, layout=layout)
        static_image_bytes = pio.to_image(fig, format='png')
        if exporting == True:
            display(Image(static_image_bytes))
        else:
            display(iplot(fig))
            
def decode_dataframe(df):
    # convert attribute values with type "object" to regular strings
    objects_df = df.select_dtypes([object]) # select only atributes of object type
    stack_df = objects_df.stack() # means create one column
    decoded_stack_df = stack_df.str.decode('utf-8') # decode the values in the column
    decoded_objects_df = decoded_stack_df.unstack() # separate into columns

    # replace in df
    for col in decoded_objects_df.columns:
        df[col] = decoded_objects_df[col]

    return df

def one_hot_encode_categorical(df):
    df_categorical = df.select_dtypes([object])
    df_numerical = df.select_dtypes([int, float])

    # create dummy variables for df_categorical
    df_one_hot = df_numerical.copy(deep=True)
    catergorical_attributes = dict()

    for atr in df_categorical.columns:
        df_dummies = pd.get_dummies(df[atr], prefix = atr)
        catergorical_attributes[atr] = [col.split('_')[1] for col in df_dummies.columns]
        df_one_hot = pd.concat([df_one_hot, df_dummies], axis=1) # the dataset ready to appy decision tree algorithm

    # one hot encoded attributes place in place of the original,
    # so that the order is preserved

    attribute_order_one_hot = []
    for i in range(len(df.columns)):
        if df.columns[i] in catergorical_attributes.keys():
            print(i, df.columns[i])
            categories = [df.columns[i] + '_' + cat for cat in catergorical_attributes[df.columns[i]]]
            attribute_order_one_hot.extend(categories)
        else:
            attribute_order_one_hot.append(df.columns[i])

    df_one_hot_ordered = df_one_hot[attribute_order_one_hot]
    return df_one_hot_ordered

def classification(X, y, test=False):
    scores = pd.DataFrame(index=range(10))
    
    cls = DecisionTreeClassifier(min_impurity_decrease=0.0001)
    scorescv = cross_val_score(cls, X, y, cv=10)
    name = 'Tree ' + str(0.0001)
    scores[name] = pd.Series(scorescv)
    print('finished', name)
    
    if test == True:
        return scores

    n = 15
    cls = KNeighborsClassifier(n_neighbors=n)
    scorescv = cross_val_score(cls, X, y, cv=10)
    name = '{}-NN'.format(n)
    scores[name] = pd.Series(scorescv)
    print('finished', name)

    
    cls = LogisticRegression(random_state=0, solver='sag', max_iter=10000)
    scorescv = cross_val_score(cls, X, y, cv=10)
    name = 'Logistic R'
    scores[name] = pd.Series(scorescv)
    print('finished', name)

    
    n = 50
    cls = RandomForestClassifier(n_estimators=n, n_jobs=-1)
    scorescv = cross_val_score(cls, X, y, cv=10)
    name = 'RandomF {} trees'.format(n)
    scores[name] = pd.Series(scorescv)
    print('finished', name)
    
    display(scores.describe())
    
    return scores

def cs_classification(X, y, test=False):
    scores = pd.DataFrame(index=range(10))
    cls = DecisionTreeClassifier(min_impurity_decrease=0.0001)
    name = 'Tree ' + str(0.0001)
    threshold = thresholding(X, y, cls, name=name)
    scores = pd.concat([scores, threshold['mincost_df']], axis=1)
    if test == True:
        return scores

    n = 15
    cls = KNeighborsClassifier(n_neighbors=n)
    name = '{}-NN'.format(n)
    threshold = thresholding(X, y, cls, name=name)
    scores = pd.concat([scores, threshold['mincost_df']], axis=1)

    cls = LogisticRegression(random_state=0, solver='sag', max_iter=10000)
    name = 'Logistic R'
    threshold = thresholding(X, y, cls, name=name)
    scores = pd.concat([scores, threshold['mincost_df']], axis=1)
    
    
    n = 50
    cls = RandomForestClassifier(n_estimators=n, n_jobs=-1)
    name = 'RandomF {} trees'.format(n)
    threshold = thresholding(X, y, cls, name=name)
    scores = pd.concat([scores, threshold['mincost_df']], axis=1)
    
    display(scores.describe())
    
    return scores

def thresholding(X, y, cls, verbose=False, name='Classifier'):
    print(cls, "\n")
    skf = KFold(n_splits=10)
    i = 0
    costs = []
    for train_index, test_index in skf.split(X, y):
        if verbose:
            print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        cls.fit(X_train, y_train)

        y_pred_proba = cls.predict_proba(X_test)[:, 1]

        if verbose:
            print(confusion_matrix(y_test, y_pred_proba > 0.5))
        
        costs_row = []
        
        for t in np.linspace(0, 1, 101):
            y_pred = y_pred_proba >= t # all above or equal t is considered to be 'yes'
            
            cost = sum((y_pred == 1) & (y_test == 0)) + 10 * sum((y_pred == 0) & (y_test == 1))
            costs_row.append(cost)
            
            if t*100 % 10 == 0 and verbose:
                print('%.2f' % t, 'TP {0:<10} TN {1:<10} FP {2:<10} FN {3:<27} cost {4:<10}'.format(
                    sum((y_pred == 1) & (y_test == 1)),
                    sum((y_pred == 0) & (y_test == 0)),
                    sum((y_pred == 1) & (y_test == 0)),
                    sum((y_pred == 0) & (y_test == 1)),
                    cost) 
                )
        
        costs.append(costs_row)
        
        t = np.argmin(costs_row)/100
        cost = costs_row[np.argmin(costs_row)]
        
        if not verbose:
            print('Fold %d/10 for %s with lowest cost %.2f at t = %.2f' % (i + 1, name, cost, t))
            i += 1
            
    costs = np.array(costs)
    costmeans = [np.mean(col) for col in costs.T]
    t = np.argmin(costmeans)/100
    mincost = min(costmeans)
    print('The lowest cost will be reached if t is equal to %10.2f\nThe lowest average cost would be %32.1f.' % (t, mincost))
    df = pd.DataFrame(costmeans, index=np.linspace(0, 1, 101), columns=[name])
    df.plot()
    plt.show()
    print("\n"*2)
    
    
    return {
        'mincost': mincost,
        't': t,
        'mincost_df': pd.DataFrame(costs[:,np.argmin(costmeans)], columns=[name]), 
    }

def boxplot_scores(scores, name='Cost'):
    data = []
    for col in scores.columns:
        data.append(go.Box(y=scores[col], name=col, showlegend=False))
    
    layout = go.Layout(
        title=go.layout.Title(
            text='%s Boxplot' % name,
            xref='paper',
            x=0
        ),
    )

    fig = go.Figure(data=data, layout=layout)
    
    if exporting == True:
        static_image_bytes = pio.to_image(fig, format='png')
        display(Image(static_image_bytes))
    else:
        display(iplot(fig))