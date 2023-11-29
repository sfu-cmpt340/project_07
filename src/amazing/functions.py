import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft
from scipy.signal import find_peaks
import librosa

columnNames = ['id','date','subjectID','heartrate','state','activity','BMI','age','caffeineLevel','sleepDuration']

def constructTable(dataFilePath):
    table = pd.read_csv(dataFilePath)
    try:
        table.columns = columnNames
        table = table.astype({
            'id': int,
            'date': 'datetime64',
            'subjectID': int,
            'heartrate': str,
            'state': str,
            'activity': str,
            'BMI': float,
            'age': int,
            'caffeineLevel': str,
            'sleepDuration': float
        })
    
    except Exception as error:
        print('error:', error)   
        print("Error creating table - make sure columns in csv are same as above.")
 
    return table

def correlate(x,y):
    corr, pVal = stats.pearsonr(x, y)
    plt.scatter(x, y)
    title = "corr = " + str(corr)
    plt.title(title)
    plt.xlabel('Average Heart Rate (bpm)')
    plt.ylabel('Sleep Duration (hours)')

    if(pVal < 0.05):
        print("Statistically Significant.")
    else:
        print(pVal)

    plt.show()

def getVideoAvgBrightnesses(videoPath):
    vid_source = cv2.VideoCapture(videoPath)
    if vid_source.get(cv2.CAP_PROP_FPS) > 40:
        iterator=2
        numFrames=500
    else:
        iterator=1
        numFrames=250
    
    avg_brightnesses = []
    # Get average brightness value of each frame
    success, img = vid_source.read()
    count = 0
    while (count < numFrames and success):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.resize(gray_img,(100,100))
        avg_brightnesses.append(np.average(gray_img))
        success, img = vid_source.read() 
        count+=iterator
    return avg_brightnesses




def getAudioBPM(audioPath):
    sampling_rate, data = wavfile.read(audioPath)
    # normalized_data = data / np.max(np.abs(data), axis=0)
    mono_data = data[:, 0].ravel()
    normalized_data = mono_data / np.max(np.abs(mono_data), axis=0)
    average_amplitude = np.mean(np.abs(normalized_data))
    print(average_amplitude)
    
    time_axis = np.arange(0, len(data)) / float(sampling_rate)
    
    # plt.figure(figsize=(12, 4))
    # plt.plot(time_axis, normalized_data, color='b')
    # plt.title('Sound Waveform')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.show()

    fft_result = fft(normalized_data)
    frequencies = np.fft.fftfreq(len(fft_result), 1/sampling_rate)
    peaks, _ = find_peaks(np.abs(fft_result))
    bpm_candidates = 60 * frequencies[peaks]
    bpm_candidates = bpm_candidates[bpm_candidates > 0]
    
    print("Number of Peaks:", len(peaks))
    print("Number of BPM candidates:", len(bpm_candidates))

    scores = []

    # todo get scores for each bpm candidate
    
    return 0 # best_bpm

def computeSpectralFlux(y):
    D = librosa.stft(y)
    
    magnitude = np.abs(D)
    
    spectral_flux = np.sum(np.maximum(0, np.diff(magnitude, axis=1)), axis=0)
    
    return spectral_flux, magnitude

def detectOnsets(spectral_flux, threshold=0.5):
    # Find peaks in the spectral flux
    peaks, _ = find_peaks(spectral_flux, height=threshold)
    
    return peaks


def beatTracking(onsets, sr, magnitude_spectrogram):
    agents = []

    for onset in onsets:
        new_agents = []
        
        for agent in agents:
            # Check if the onset fits closely with the last tracked beat and tempo
            if 0.8 < (onset - agent['last_beat']) / agent['tempo'] < 1.2:
                agent['beats'].append(onset)
                agent['last_beat'] = onset
                agent['tempo'] = 60 / np.median(np.diff(agent['beats']))
                agent['score'] += magnitude_spectrogram[:, int(librosa.time_to_frames(onset, sr=sr))]
            else:
                # Spawn a new agent if the onset is in-between
                new_agent = agent.copy()
                new_agent['beats'].append(onset)
                new_agent['last_beat'] = onset
                new_agent['tempo'] = 60 / np.median(np.diff(new_agent['beats']))
                new_agent['score'] += magnitude_spectrogram[:, int(librosa.time_to_frames(onset, sr=sr))]
                new_agents.append(new_agent)

        # Initialize a new agent if there are none or the onset is wildly different
        if not agents or not new_agents:
            new_agent = {
                'beats': [onset],
                'last_beat': onset,
                'tempo': 120,  # Initial tempo (adjust as needed)
                'score': magnitude_spectrogram[:, int(librosa.time_to_frames(onset, sr=sr))]
            }
            new_agents.append(new_agent)

        agents = new_agents

    best_agent = max(agents, key=lambda agent: agent['score'])
    
    return best_agent

def getAudioBPM2(file_path):
    y, sr = librosa.load(file_path)
    
    spectral_flux, magnitude = computeSpectralFlux(y)
    # plt.figure(figsize=(12, 4))
    # plt.title('FFT Spectrum')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Amplitude')
    # plt.plot(np.fft.fftfreq(len(spectral_flux), 1/sr), np.abs(spectral_flux), color='b')
    # plt.show()

    # plt.figure(figsize=(12, 4))
    # plt.title('Spectral Flux')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Spectral Flux')
    # plt.plot(librosa.times_like(spectral_flux, sr=sr), spectral_flux, color='r')
    # plt.show()

    onsets = detectOnsets(spectral_flux)

    onset_times = librosa.frames_to_time(onsets, sr=sr)
    
    best_agent = beatTracking(onset_times, sr, magnitude)
    
    return best_agent['tempo']