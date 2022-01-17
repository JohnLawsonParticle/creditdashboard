# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 23:03:43 2022

@author: Guillaume
"""

# -*- coding: utf-8 -*-
import streamlit as st

import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.manifold import TSNE

from funcs import *

st.set_page_config(layout="wide")

session = requests.Session()

ot_value =  0.25470000000000004

df_source = import_df()

df_to_predict = df_source.copy()

predict = predict_df(session,df_to_predict)

df = predict.copy()


default_user_code = sorted(df["SK_ID_CURR"])[0]

st.title("Credit Dashboard ")

varExploration = st.container()
model_container = st.container()
simulation = st.container()

st.sidebar.title("Client Profile")
input_user_code = default_user_code
input_user_code = st.sidebar.selectbox('How would you like to be contacted?',tuple(sorted(df["SK_ID_CURR"])))
index_input_user = get_index_from_SK_ID(df,input_user_code)
display_profile = ['NAME_CONTRACT_TYPE','CODE_GENDER','AMT_CREDIT','AMT_ANNUITY']

st.sidebar.table(df.loc[index_input_user,display_profile].astype(str))


   
with varExploration :
    st.title("Variables Exploration")
    col1,col2 = st.columns([1,3])
    
    col1.subheader("Input Var")
    with col1 : 
        option = st.selectbox('Which variable would you like to explore?',
                              ('DAYS_BIRTH','AMT_CREDIT','AMT_ANNUITY', "CREDIT_INCOME_PERCENT"))
        st.write('You selected:', option)
        df_fault = df.loc[df["TARGET"]=="Faulter",option]
        df_no_fault = df.loc[df["TARGET"]=="Non Faulter",option]
        st.write('Candidate value : ', df.loc[index_input_user,option])
        st.write('Faulters value Mean : ', df.loc[df["TARGET"]=="Faulter",option].mean())
        st.write('Non Faulters value Mean : ', df.loc[df["TARGET"]=="Non Faulter",option].mean())
    
    
    col2.subheader("Distribution")
    
    with col2 : 

        hist_data = [df_fault,df_no_fault]
        group_labels = ['Faulters', 'Non Faulters']
        fig = ff.create_distplot(hist_data, group_labels, show_hist=False)
        fig.add_vrect(x0=df.loc[index_input_user,option], x1= df.loc[index_input_user,option], annotation_text="candidate" )
        fig.update_yaxes(visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
with model_container : 
    st.title("Model results")
    col1,col2 = st.columns(2)
    
    with col1 : 
        st.subheader("Var influence for Client")
        st.write("Predicted Score : ", df.loc[index_input_user,"Score"])
        pred_target = df.loc[index_input_user,"TARGET"]
        st.write(f"Prediction : {pred_target} ") 
        st.plotly_chart(plot_score(df.loc[index_input_user,"Score"]))
                   
    with col2 : 
        st.subheader("Variable Influence Global")
        st.image("data_streamlit/shap_img.png")

        
with simulation : 
    st.title("Simulation")
    col1,col2 = st.columns(2)
    with col1 : 
        st.subheader("Inputs")
        
        input_credit_amt = int(df.loc[index_input_user,"AMT_CREDIT"])
        input_credit_ann = int(df.loc[index_input_user,"AMT_ANNUITY"])
        input_income_percent = float(df.loc[index_input_user,"CREDIT_INCOME_PERCENT"])
        input_credit_term = int(df.loc[index_input_user,"CREDIT_TERM"])
        
    
        new_credit_term = st.slider('Set New Credit Term', 1, 100, input_credit_term)
        st.write("New credit term : ", new_credit_term )    
        new_credit_amt = st.slider('Set New Credit Amount', 0, 5000000, input_credit_amt)
        st.write("New credit amout : ", new_credit_amt )
        new_credit_ann = st.slider('Set New Credit Annuity', 0, 100000, input_credit_ann)
        st.write("New credit annuity : ", new_credit_ann )
        new_income_percent = st.slider('Set New Credit Income Percent', 0.0, 10.0, input_income_percent)
        st.write("New credit income percent : ", new_income_percent )

           
    with col2 : 
        st.subheader("Results")
            

        if (new_credit_term==input_credit_term) and (new_credit_amt == input_credit_amt) and (new_credit_ann == int(input_credit_ann)) and(new_income_percent == float(input_income_percent)):
            
            st.write("Predicted : ", df.loc[index_input_user,"TARGET"])
            st.write("Predicted Score : ", df.loc[index_input_user,"Score"])
            
        else : 
            score, prediction = predict_simu(session, input_user_code,new_credit_amt,new_credit_ann,new_income_percent,new_credit_term)
                        
            st.write("Predicted : ", prediction)
            st.write("Predicted Score : ", score)
            
        st.plotly_chart(plot_neighs(session,df,input_user_code),use_container_width=True)
