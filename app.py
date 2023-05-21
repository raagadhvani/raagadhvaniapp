import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import platform
import time
import pathlib
import os

import streamlit as st

## menu
st.set_page_config(page_title = "Raagadhvani")


#title of page
st.title("Raagdhvani 🎵")
                                                                                                                                                   


#hiding menu
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

st.sidebar.success("Menu")
st.write("Welcome to raaga dhvani")