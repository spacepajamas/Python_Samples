
"""A demo script showing how to DIARIZATION ON WAV USING UIS-RNN."""

import numpy as np
import uisrnn
import librosa
import sys
sys.path.append('ghostvlad')
sys.path.append('visualization')
import toolkits
import model as spkModel
import os
from viewer import PlotDiar
import pickle

# ===========================================
#        Parse the argument
# ===========================================
# import argparse
# parser = argparse.ArgumentParser()
# # set up training configuration.
# parser.add_argument('--gpu', default='', type=str)
# # parser.add_argument('--resume', default=r'ghostvlad/pretrained/weights.h5', type=str)
# parser.add_argument('--resume', default=r'ghostvlad/pretrained/weights_xie.h5', type=str)
# parser.add_argument('--data_path', default='pg_audio', type=str)
# # set up network configuration.
# parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
# parser.add_argument('--ghost_cluster', default=2, type=int)
# parser.add_argument('--vlad_cluster', default=8, type=int)
# parser.add_argument('--bottleneck_dim', default=512, type=int)
# parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
# # set up learning rate, training loss and optimizer.
# parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
# parser.add_argument('--test_type', default='normal', choices=['normal', 'hard', 'extend'], type=str)
#
# global args
# args = parser.parse_args()
#

# Namespace(aggregation_mode='gvlad', bottleneck_dim=512, data_path='pg_audio', ghost_cluster=2, gpu='', loss='softmax', net='resnet34s', resume='ghostvlad/pretrained/weights_xie.h5', test_type='normal', vlad_cluster=8)

from collections import namedtuple
def convert(dictionary):
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)


args_custom_dict = {'aggregation_mode':'gvlad', 'bottleneck_dim':512, 'data_path':'pg_audio', 'ghost_cluster':2, 'gpu':'', 'loss':'softmax', 'net':'resnet34s', 'resume':'ghostvlad/pretrained/weights.h5', 'test_type':'normal', 'vlad_cluster':8}
args = convert(args_custom_dict)

print(args)


SAVED_MODEL_NAME = 'pretrained/saved_model.uisrnn_benchmark'

def append2dict(speakerSlice, spk_period):
    key = list(spk_period.keys())[0]
    value = list(spk_period.values())[0]
    timeDict = {}
    timeDict['start'] = int(value[0]+0.5)
    timeDict['stop'] = int(value[1]+0.5)
    if(key in speakerSlice):
        speakerSlice[key].append(timeDict)
    else:
        speakerSlice[key] = [timeDict]

    return speakerSlice

def arrangeResult(labels, time_spec_rate): # {'1': [{'start':10, 'stop':20}, {'start':30, 'stop':40}], '2': [{'start':90, 'stop':100}]}
    lastLabel = labels[0]
    speakerSlice = {}
    j = 0
    for i,label in enumerate(labels):
        if(label==lastLabel):
            continue
        speakerSlice = append2dict(speakerSlice, {lastLabel: (time_spec_rate*j,time_spec_rate*i)})
        j = i
        lastLabel = label
    speakerSlice = append2dict(speakerSlice, {lastLabel: (time_spec_rate*j,time_spec_rate*(len(labels)))})
    return speakerSlice

def genMap(intervals):  # interval slices to maptable
    slicelen = [sliced[1]-sliced[0] for sliced in intervals.tolist()]
    mapTable = {}  # vad erased time to origin time, only split points
    idx = 0
    for i, sliced in enumerate(intervals.tolist()):
        mapTable[idx] = sliced[0]
        idx += slicelen[i]
    mapTable[sum(slicelen)] = intervals[-1,-1]

    keys = [k for k,_ in mapTable.items()]
    keys.sort()
    return mapTable, keys

def fmtTime(timeInMillisecond):
    millisecond = timeInMillisecond%1000
    minute = timeInMillisecond//1000//60
    second = (timeInMillisecond-minute*60*1000)//1000
    time = '{}:{:02d}.{}'.format(minute, second, millisecond)
    return str(timeInMillisecond/1000)

def load_wav(vid_path, sr):
    wav, _ = librosa.load(vid_path, sr=sr)
    intervals = librosa.effects.split(wav, top_db=20)
    wav_output = []
    for sliced in intervals:
      wav_output.extend(wav[sliced[0]:sliced[1]])
    return np.array(wav_output), (intervals/sr*1000).astype(int)

def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
    linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length) # linear spectrogram
    return linear.T


# 0s        1s        2s                  4s                  6s
# |-------------------|-------------------|-------------------|
# |-------------------|
#           |-------------------|
#                     |-------------------|
#                               |-------------------|
def load_data(path, win_length=400, sr=16000, hop_length=160, n_fft=512,
              embedding_per_second=0.5, overlap_rate=0.5):
    wav, intervals = load_wav(path, sr=sr)
    linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    print('freq, time', freq, ',', time)
    spec_mag = mag_T

    spec_len = sr / hop_length / embedding_per_second
    print('sr', sr, 'hop_length', hop_length, 'embedding_per_second', embedding_per_second)
    spec_hop_len = spec_len #* (1 - overlap_rate)
    print('spec_hop_len', spec_hop_len)

    cur_slide = 0.0
    utterances_spec = []
    k = 0  ##PG
    stop_iter = False
    while (True):  # slide window.
        if (cur_slide + spec_len > time):
            #spec_len = time - cur_slide
            stop_iter = True
            break
        spec_mag = mag_T[:, int(cur_slide + 0.5): int(cur_slide + spec_len + 0.5)]
        k += 1
        print('spec_mag', k, spec_mag.shape)
        print('cur_slide + spec_len', cur_slide + spec_len)
        print('cur_slide , spec_len', cur_slide, ',', spec_len)

        # preprocessing, subtract mean, divided by time-wise var
        mu = np.mean(spec_mag, 0, keepdims=True)
        std = np.std(spec_mag, 0, keepdims=True)
        spec_mag = (spec_mag - mu) / (std + 1e-5)
        utterances_spec.append(spec_mag)

        cur_slide += spec_hop_len

        if stop_iter:
            break

    return utterances_spec, intervals


def main(wav_path, embedding_per_second=1.0,n_classes =5994, overlap_rate=0.5,plot_results = True):

    # gpu configuration
    toolkits.initialize_GPU(args)

    params = {'dim': (257, None, 1),
              'nfft': 512,
              'spec_len': 250,
              'win_length': 400,
              'hop_length':160,
              'n_classes': 5994,
              'sampling_rate': 16000,
              'normalize': True,
              }

    network_eval = spkModel.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                                num_class=params['n_classes'],
                                                mode='eval', args=args)
    network_eval.load_weights(args.resume, by_name=True)
    #

    model_args, _, inference_args = uisrnn.parse_arguments()
    model_args.observation_dim = 512
    uisrnnModel = uisrnn.UISRNN(model_args)
    uisrnnModel.load(SAVED_MODEL_NAME)

    specs, intervals = load_data(wav_path, embedding_per_second=embedding_per_second, overlap_rate=overlap_rate)
    mapTable, keys = genMap(intervals)

    print('intervals',intervals,len(intervals))
    print('mapTable',mapTable, len(mapTable))
    print('keys',keys,len(keys))
    # print('mapTable, keys', mapTable, keys)
    feats = []
    for spec in specs:
        spec = np.expand_dims(np.expand_dims(spec, 0), -1)
        v = network_eval.predict(spec)
        # print('v',v.shape)
        #print('feats', feats.shape)

        feats += [v]

    feats = np.array(feats)[:,0,:].astype(float)  # [splits, embedding dim]
    predicted_label = uisrnnModel.predict(feats, inference_args)
    print(feats.shape)
    print(inference_args)
    print('predicted_label',predicted_label)

    time_spec_rate = 1000*(1.0/embedding_per_second)*(1.0-overlap_rate) # speaker embedding every ?ms
    print('time_spec_rate',time_spec_rate)
    center_duration = int(1000*(1.0/embedding_per_second)//2)
    speakerSlice = arrangeResult(predicted_label, time_spec_rate)
    print('speakerSlice',speakerSlice)
    for spk,timeDicts in speakerSlice.items():    # time map to orgin wav(contains mute)
        print(spk,timeDicts)
        for tid,timeDict in enumerate(timeDicts):
            print(tid, timeDict)
            s = 0
            e = 0
            for i,key in enumerate(keys):
                if(s!=0 and e!=0):
                    break
                if(s==0 and key>timeDict['start']):
                    offset = timeDict['start'] - keys[i-1]
                    print('offset', offset)
                    s = mapTable[keys[i-1]] + offset
                if(e==0 and key>timeDict['stop']):
                    offset = timeDict['stop'] - keys[i-1]
                    e = mapTable[keys[i-1]] + offset

                print('i,s,e')
                print(i,s,e,tid,spk)
            print('>>>>>',i, s, e, tid, spk)
            speakerSlice[spk][tid]['start'] = s
            speakerSlice[spk][tid]['stop'] = e

    speaker_assingments = []
    for spk,timeDicts in speakerSlice.items():
        speaker = str(spk)
        print('========= ' + str(spk) + ' =========')
        for timeDict in timeDicts:
            start = timeDict['start']
            end = timeDict['stop']
            start = fmtTime(start)  # change point moves to the center of the slice
            end = fmtTime(end)
            print(start+' ==> '+end)
            speaker_assingments.append((start,end,speaker,wav_path))

    if plot_results:
        p = PlotDiar(map=speakerSlice, wav=wav_path, gui=True, size=(25, 6))
        p.draw()
        p.plot.show()

    return feats,predicted_label,intervals, speaker_assingments,time_spec_rate


def absoluteFilePaths(directory):
   for dirpath,_,filenames in os.walk(directory):
       for f in filenames:
           yield os.path.abspath(os.path.join(dirpath, f))


if __name__ == '__main__':
    file_idx = 1

    file_name_header = "/Users/pari/University/DIAR_Project/DEMO_DATA/"

    try:
        os.listdir(file_name_header + 'model_data/')
    except:
        os.makedirs(file_name_header + 'model_data/')

    ##path to artificial segment, but could be any .wav file
    path =file_name_header+'artificial_segments/segment_' + str(file_idx) + '_.wav'

    embedding_per_second = 0.5

    diar_data = main(path, embedding_per_second=embedding_per_second,
                     overlap_rate=0.01,
                     plot_results=True)

    ##path to pkl file where all the data is saved in a pickel file
    ## here it's the model_data/ folder
    calc_output_path = file_name_header+ 'model_data/'+str(file_idx)+'_'+str(embedding_per_second)+'.pkl'
    pickle.dump(diar_data, open(calc_output_path, 'wb'))


