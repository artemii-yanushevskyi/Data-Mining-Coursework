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

exporting = False


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