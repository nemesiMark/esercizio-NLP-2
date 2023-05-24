import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
import joblib

# 0) Prima del clone andando in alto a DX sul progetto git, premo fork e la creo
#
# 1)  Come scaricare la cartella creata su GitHub
#           git clone https://github.com/nemesiMark/app.git

# 1-bis) se voglio sincronizzarmi premo su vscode e sotto trova, in IFTS23 premo i tre puntini e clicco "esegui pull"
#
# 2)  Come uplodare file su GitHub
#           git add .
#           git commit -m "nome modifica"
#           git push

# 3)  Come runnare su streamlit
#           steamlit run app0.py


def main():

    st.header("Data Mining")
    #st.markdown("<h1 style='text-align: center; color: black;'>Data Transformation</h1>", unsafe_allow_html=True)
    # -----------------------------------------------------------------------------------------------------------
    uploaded_file = st.file_uploader("Choose your file CSV:")

    if uploaded_file is not None:

        df = pd.DataFrame()

        if uploaded_file.name[-3:] != "csv":

            st.warning("CSV file is required.")

        else:
            df = pd.read_csv(uploaded_file)

            st.dataframe(df)
            
            word = st.text_input("Inserisci la parola: ")

            # Carica il modello dal file .pkl
            pipe = joblib.load('pipeline_classifierTF_IDF.pkl')
            
            if word == "":
                st.warning("Enter the text: ")
            else:
            
                fake = []
                fake.append(word)

                predictions = pipe.predict(fake)
                
                # Cambia il colore del background in base al sentiment
                if predictions[0] == "positive":
                    st.write(
                        f'<div style="background-color: lightgreen; padding: 5px;">'
                        f'Sentiment: {predictions[0].upper()}'
                        '</div>',
                        unsafe_allow_html=True
                    )
                elif predictions[0] == "neutral":
                    st.write(
                        f'<div style="background-color: lightgrey; padding: 5px;">'
                        f'Sentiment: {predictions[0].upper()}'
                        '</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.write(
                        f'<div style="background-color: rgb(255, 100, 100); padding: 5px;">'
                        f'Sentiment: {predictions[0].upper()}'
                        '</div>',
                        unsafe_allow_html=True
                    )
                

if __name__ == "__main__":
    main()
