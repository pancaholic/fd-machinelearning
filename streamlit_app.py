import streamlit as st
import pandas as pd

st.title('Comparative Model For Fraud Detection')

st.info('Comparative study for a machine learning model to decide which is the best for a certain dataset')

df = pd.read_csv('https://raw.githubusercontent.com/pancaholic/skripsi/refs/heads/master/Finding/Fraud%20Detection%20Dataset.csv?token=GHSAT0AAAAAADCYTOUNIRTCV6SOTDE34HCG2A2YAEQ')

df


