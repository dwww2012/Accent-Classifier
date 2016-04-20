import numpy as np
import pandas as pd
from features import mfcc
from features import logfbank
import scipy.io.wavfile as wav
from scipy.io.wavfile import write as wav_write
import librosa
import scikits.samplerate
import os


'''
mfcc(signal, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True)
'''
# read in wav file, get out signal (np array) and sampling rate (int)
def read_in_audio(filename):
    (rate, sig) = wav.read(filename)
    return sig, rate


# read in signal, take absolute value and slice seconds 1-3 from beginning
def get_two_secs(filename):
    sig, rate = read_in_audio(filename)
    abs_sig = np.abs(sig)
    two_secs = abs_sig[rate:3*rate]
    return two_secs

# calculates moving average for a specified window (number of samples)
def take_moving_average(sig, window_width):
    cumsum_vec = np.cumsum(np.insert(sig, 0, 0))
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width])/float(window_width)
    return ma_vec

# read in signal, change sample rate to outrate (samples/sec), use write_wav=True to save wav file to disk
def downsample(filename, outrate=8000, write_wav = False):
    (rate, sig) = wav.read(filename)
    down_sig = librosa.core.resample(sig, rate, outrate, scale=True)
    if not write_wav:
        return down_sig, outrate
    if write_wav:
        wav_write('{}_down_{}.wav'.format(filename, outrate), outrate, down_sig)

# change total number of samps for downsampled file to n_samps by trimming or zero-padding and standardize them
def make_standard_length(filename, n_samps=240000):
    down_sig, rate = downsample(filename)
    normed_sig = librosa.util.fix_length(down_sig, n_samps)
    normed_sig = (normed_sig - np.mean(normed_sig))/np.std(normed_sig))
    return normed_sig

# from a folder containing wav files, normalize each, divide into num_splits-1 chunks and write the resulting np.arrays to a single matrix
def make_split_audio_array(folder, num_splits = 5):
    lst = []
    for filename in os.listdir(folder):
        if filename.endswith('wav'):
            normed_sig = make_standard_length(filename)
            chunk = normed_sig.shape[0]/num_splits
            for i in range(num_splits - 1):
                lst.append(normed_sig[i*chunk:(i+2)*chunk])
    lst = np.array(lst)
    lst = lst.reshape(lst.shape[0], -1)
    return lst

# for input wav file outputs (13, 2999) mfcc np array
def make_normed_mfcc(filename, outrate=8000):
    normed_sig = make_standard_length(filename)
    normed_mfcc_feat = mfcc(normed_sig, outrate)
    normed_mfcc_feat = normed_mfcc_feat.T
    return normed_mfcc_feat

# make mfcc np array from wav file using librosa package
def make_librosa_mfcc(filename):
     y, sr = librosa.load(filename)
     mfcc_feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
     return mfcc_feat

# make mfcc np array from wav file using speech features package
def make_mfcc(filename):
    (rate, sig) = wav.read(filename)
    mfcc_feat = mfcc(sig, rate)
    mfcc_feat = mfcc_feat.T
    return mfcc_feat

# for folder containing wav files, output numpy array of normed mfcc
def make_class_array(folder):
    lst = []
    for filename in os.listdir(folder):
        lst.append(make_normed_mfcc(filename))
    class_array = np.array(lst)
    class_array = np.reshape(class_array, (class_array.shape[0], class_array.shape[2], class_array.shape[1]))
    return class_array

# read in wav file, output (1,13) numpy array of mean mfccs for each of 13 features
def make_mean_mfcc(filename):
    try:
        (rate, sig) = wav.read(filename)
        mfcc_feat = mfcc(sig, rate)
        avg_mfcc = np.mean(mfcc_feat, axis = 0)
        return avg_mfcc
    except:
        pass

# write new csv corresponding to dataframe of given language and gender
def make_df_language_gender(df, language, gender):
    newdf = df.query("native_language == @language").query("sex == @gender")
    newdf.to_csv('df_{}_{}.csv'.format(language, gender))

# write new directories to disk containing the male and female speakers from the most common languages
def make_folders_from_csv():
    top_15_langs = ['english', 'spanish', 'arabic', 'mandarin', 'french', 'german', 'korean', 'russian', 'portuguese', 'dutch', 'turkish', 'italian', 'polish', 'japanese', 'vietnamese']
    for lang in top_15_langs:
        os.makedirs('{}/{}_male'.format(lang, lang))
        os.makedirs('{}/{}_female'.format(lang, lang))

# copy files to the corresponding directories
def copy_files_from_csv():
    top_15_langs = ['english', 'spanish', 'arabic', 'mandarin', 'french', 'german', 'korean', 'russian', 'portuguese', 'dutch', 'turkish', 'italian', 'polish', 'japanese', 'vietnamese']
    for lang in top_15_langs:
        df_male = pd.read_csv('df_{}_male.csv'.format(lang))
        df_female = pd.read_csv('df_{}_female.csv'.format(lang))
        m_list = df_male['filename'].values
        f_list = df_female['filename'].values
        for filename in f_list:
            shutil.copy2('big_langs/{}/{}.wav'.format(lang, filename), 'big_langs/{}/{}_female/{}.wav'.format(lang, lang, filename))

# input folder of wav files, output pandas dataframe of mean mfcc values
def make_mean_mfcc_df(folder):
    norms = []
    for filename in os.listdir(folder):
        (rate, sig) = wav.read(filename)
        mfcc_feat = mfcc(sig, rate)
        mean_mfcc = np.mean(mfcc_feat, axis = 0)
        #mean_mfcc = np.reshape(mean_mfcc, (1,13))
        norms.append(mean_mfcc)
    flat = [a.ravel() for a in norms]
    stacked = np.vstack(flat)
    df = pd.DataFrame(stacked)
    return df
