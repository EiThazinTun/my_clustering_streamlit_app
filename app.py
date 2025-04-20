# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 15:54:01 2025

@author: LAB
"""

#app.py
import streamlit as st
import pickle
import matplotlib.pyplot as plt

#Load model
with open('kmeans_model.pkl', 'rb') as f:
    loaded_model  = pickle.load(f)
    
#Set the page config
st.set_page_config(page_title="k-Means Clustering App", layout="centered")

#Set title
st.title(" k-Means Clustering Visualizer")

#Display cluster centers
st.subheader(" Example Data for Visualization")
st.markdown("This demo uses example data(2D) to illustrate clustering results.")

#Load from a saved dataset ot generate synthetic data
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=300, centers=loaded_model.n_clusters, cluster_std=0.60, random_state=0)

#Predictusing the loaded model
y_kmeans=loaded_model.predict(X)

# Predict using the loaded model
y_kmeans = loaded_model.predict(X)

