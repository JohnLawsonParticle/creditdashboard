# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 20:29:35 2022

@author: Guillaume
"""

import streamlit as st
import pandas as pd 
from xgboost import XGBClassifier
from pycaret.classification import load_model
import shap
import streamlit.components.v1 as components
from sklearn.pipeline import Pipeline
import plotly.graph_objects as go
import requests


@st.cache(allow_output_mutation=True)
def import_df():
    df = pd.read_csv('data_streamlit/data_api.csv')
    return df

@st.cache
def get_index_from_SK_ID(df,sk_id) :
    loc_id = df.loc[df["SK_ID_CURR"] == sk_id,"SK_ID_CURR"]
    index_user = loc_id.index.values[0]
    return index_user

@st.cache(hash_funcs={XGBClassifier: id},allow_output_mutation=True)
def import_model():
    model_complete = load_model("final_model_v2")
    model = load_model("final_model_v2").steps[-1][1]
    pipeline = load_model("final_model_v2").steps[:-1]
    return model,pipeline, model_complete

@st.cache(hash_funcs={XGBClassifier: id},allow_output_mutation=True)
def pipeline_transform(pipeline,df) : 
    
    pipe = Pipeline(pipeline)
    
    X = pipe.fit_transform(df)
    X['SK_ID_CURR'] = df['SK_ID_CURR']
    
    return X

@st.cache
def plot_score(score=0.5):
    ot_value =  0.25470000000000004
    decision_threshold = 1 - ot_value

    fig = go.Figure()

    fig.add_shape(type="rect",
        x0=0, y0=0, x1=decision_threshold-0.05, y1=1, fillcolor="red",
        line=dict(
            width=0,
        )
    )

    fig.add_shape(type="rect",
        x0=decision_threshold-0.05, y0=0, x1=decision_threshold+0.05, y1=1, fillcolor="orange",
        line=dict(
            width=0,
        )
    )

    fig.add_shape(type="rect",
        x0=decision_threshold+0.05, y0=0, x1=1, y1=1, fillcolor="blue",
        line=dict(
            width=0,
        )
    )

    fig.add_shape(type="line",
        x0=score, y0=-0.5, x1=score, y1=1.5,
        line=dict(color="Black",width=2)
    )

    fig.update_layout(xaxis_range=[-0.1,1.1], yaxis_range=[-1,2], plot_bgcolor ="white", 
                      title="Predicted Score and Decision Boundary")
    fig.update_yaxes(visible=False)

    return fig

def fetch(session, url):
    try:
        result = session.get(url)
        return result.json()
    except Exception:
        return {}
    
@st.cache
def predict_df(session, df): 
    
    targets = []
    scores =  []
    
    for id_client in df["SK_ID_CURR"] :
        
        data = fetch(session, f'https://credit-dash-api.herokuapp.com/get_predict?id_client={id_client}')
        targets.append(data["prediction"])
        scores.append(data["score"])
        
    df["TARGET"] = targets  
    df["Score"] = scores
    
    return df

@st.cache
def predict_simu(session, id_client,new_credit_amt,new_credit_ann,new_income_percent,new_credit_term) :
    
    data = fetch(session, f'https://credit-dash-api.herokuapp.com/get_predict_simu?id_client={id_client}&new_credit_amt={new_credit_amt}&new_credit_ann={new_credit_ann}&new_income_percent={new_income_percent}&new_credit_term={new_credit_term}')
    score = data["score"]
    prediction = data["prediction"]
    
    return score, prediction
        
#@st.cache(allow_output_mutation=True)
def get_neighs(session, id_client): 
    
    data = fetch(session, f'https://credit-dash-api.herokuapp.com/get_neighbours?id_client={id_client}')
    neighs_list = [data["neigh_1"],data["neigh_2"],data["neigh_3"],data["neigh_4"]]
    dist_list =[data["dist_1"],data["dist_2"],data["dist_3"],data["dist_4"]] 
    
    return neighs_list, dist_list
    

def plot_neighs(session,df,input_user_code) : 
    ot_value =  0.25470000000000004
    decision_threshold = 1 - ot_value
    
    return_neighs, dist_list = get_neighs(session,input_user_code)
    
    neighs = return_neighs.copy()
    dist = dist_list.copy()
    
    neighs.insert(0, get_index_from_SK_ID(df,input_user_code))
    dist.insert(0, 0)
      
    var_tooltip = ['DAYS_BIRTH', 'AMT_CREDIT', 'AMT_ANNUITY', 'CREDIT_INCOME_PERCENT',"Score","TARGET","SK_ID_CURR"]
    
    nn_df = df.loc[neighs,var_tooltip]
    nn_df['DAYS_BIRTH'] = round(abs(nn_df['DAYS_BIRTH'])/365,0)
    nn_df['Status'] = ["Candidate","Neighbour","Neighbour","Neighbour","Neighbour"]
    nn_df["Size"] = [35,25,25,25,25]
    nn_df["theta"] = [0,0,90,180,270]
    nn_df["radius"] = dist
    nn_df["Score"] = nn_df["Score"].astype(float)
    
    fig = go.Figure(data=
        go.Scatterpolar(customdata = nn_df,
                      hovertemplate="<br><b> %{customdata[7]} : %{customdata[5]} </b> \
                    <br> Id : %{customdata[6]} \
                    <br> Age : %{customdata[0]} <br> Credit Amount : %{customdata[1]} \
                    <br> Credit Annuity : %{customdata[2]} \
                    <br> Credit Income % : %{customdata[3]:.2f} \
                    <br> Score : %{customdata[4]:.2f}  <extra></extra>", 
            r = nn_df["radius"],
            theta = nn_df["theta"],
            mode = 'markers',
            marker=dict(
                            color=nn_df.loc[:,'Score'],
                            colorscale = "RdBu",
                            cmin = 0,
                            cmax = 1,
                            cmid = decision_threshold,
                            size=nn_df.loc[:,'Size'])
        ))
    
    fig.update_layout(title="Nearest Neighbours" , polar = dict(angularaxis =
                                   dict(showgrid = False,showline = False, showticklabels=False),
                                radialaxis =
                                   dict(showgrid = False, showline = False, showticklabels=False),
                         bgcolor = 'rgba(0,0,0,0)'))
    fig.update_layout(showlegend=False)
    return fig