import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from modeler.modeler import Modeler
st.set_page_config(layout="wide")

## Backend
m = Modeler()
m.prepro()
m.fit()

## Frontend

st.title('Our first streamlit app with ml')

params = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'radius error',
       'perimeter error', 'area error', 'compactness error', 'concavity error',
       'concave points error', 'worst radius', 'worst texture',
       'worst perimeter', 'worst area', 'worst smoothness',
       'worst compactness', 'worst concavity', 'worst concave points',
       'worst symmetry', 'worst fractal dimension']

default_values = [1.799e+01, 1.038e+01, 1.228e+02, 1.001e+03, 1.184e-01, 2.776e-01,
       3.001e-01, 1.471e-01, 2.419e-01, 1.095e+00, 8.589e+00, 1.534e+02,
       4.904e-02, 5.373e-02, 1.587e-02, 2.538e+01, 1.733e+01, 1.846e+02,
       2.019e+03, 1.622e-01, 6.656e-01, 7.119e-01, 2.654e-01, 4.601e-01,
       1.189e-01]
d = []

ncol = 8
nrow = 4

col1, col2, col3, col4, col5, col6, col7, col8 = st.beta_columns(ncol)
cols = [col1, col2, col3, col4, col5, col6, col7, col8]

for index, param in enumerate(params):
    d.append(cols[int(index/nrow)].number_input(param,0.0,10000.0,default_values[index]))

# Barchart plot function
def barplot(pred):
    category_names = ['No', 'Yes']
    results = {
        'Breast Cancer': pred,
    }

    labels = ['']
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn_r')(np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(16, 1))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5,
                label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            ax.text(x, y, str(int(100*c))+'%', ha='center', va='center',
                    color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
                loc='lower left', fontsize='x-large')
    return fig

st.header('Does the patient have breast cancer?')
st.pyplot(barplot(m.predictproba(d)))