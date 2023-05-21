import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import platform
import time
import pathlib
import os
#importing required libraries
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import jenkspy
import json
import streamlit as st
#title of page
st.title("Raagdhvani 🎵")
st.header("Classify your song")                                                                                                                                                  

#disabling warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

#hiding menu
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)


from io import StringIO

uploaded_file = st.file_uploader("Choose a file to classify (wav/mp3)")

if uploaded_file is not None:

        ragajson ="""{
            "shankarabharanam": [1.0, 1.125,1.25,1.333,1.5,1.667,1.875,2.00],
            "mayamalagowla":[1.0,1.067,1.25,1.33,1.5,1.6,1.875,2.00]
        }"""
        database = json.loads(ragajson)
        CLUSTER_SIZE=8
        samples, fs = librosa.load(uploaded_file, sr=None)
        #drawing spectogram
        D_highres = librosa.stft(samples, hop_length=100, n_fft=4096)
        S_db_hr = librosa.amplitude_to_db(np.abs(D_highres), ref=np.max)
        plt.figure()
        librosa.display.specshow(S_db_hr, x_axis='time', y_axis='log', sr=fs)
        plt.colorbar()
        plt.xlabel('time')
        plt.ylabel('Frequency (log scale)')
        st.pyplot()




        BAND_PASS_FL = st.number_input('Enter frequency of madhyama sthayi Sa: ')
        if BAND_PASS_FL:
            BAND_PASS_FH= 2*BAND_PASS_FL

            n = len(samples)
            xf = np.linspace(0, int(fs/2), int(n/2))

            # compute fft
            yf = np.fft.fft(samples)




            # bandpass filter
            indices = np.where((xf > BAND_PASS_FL) & (xf < BAND_PASS_FH))
            xf = np.take(xf, indices)[0]
            yf = np.take(yf, indices)[0]



            freqs = np.copy(xf)
            amps = np.copy(yf)

            # split frequencies into clusters
            clusters = jenkspy.jenks_breaks(freqs, n_classes=CLUSTER_SIZE)
            cluster_idx = np.array(
                list(map(lambda bound: (np.where(freqs == bound)[0][0]), clusters))
            )
            #print('Cluster bounding frequencies:', clusters)


            # find peaks
            peakFreqs = []
            for i in range(len(cluster_idx)-1):
                start = cluster_idx[i] if(i == 0) else cluster_idx[i]+1
                stop = cluster_idx[i+1]+1
                peak = freqs[np.where(abs(amps) == max(abs(amps[start:stop])))[0][0]]
                peakFreqs.append(peak)

            #print('Peak Frequencies:', peakFreqs)

            # normalize peaks
            normPeakFreqs = [freq/min(peakFreqs) for freq in peakFreqs]

            # compute euclidean distance
            dists = []
            for entry in database:
                trueRatios = database[entry]
                computedRatios = np.array(normPeakFreqs)
                maxLen = max(len(trueRatios), len(computedRatios))
                trueRatios = np.pad(trueRatios, (0, maxLen - len(trueRatios)))
                computedRatios = np.pad(computedRatios, (0, maxLen - len(computedRatios)))
                dist = np.linalg.norm(computedRatios - trueRatios)
                dists.append(dist)
            #print(dists)


            # predict raaga using index of minDist
            minDist = min(dists)
            minDistIdx = dists.index(minDist)

            st.write("Distance from Shankarabharanam: "+str(dists[0]))
            st.write("Distance from Mayamalagowla: "+str(dists[1]))
            if(minDistIdx==0):
                st.write("Identified Raga: Shankarabharanam")
            else:
                st.write("Identified Raga: Mayamalagowla")