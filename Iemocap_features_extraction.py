import numpy as np
import dcase_util
import os
import glob
import librosa
import python_speech_features as ps
import pickle
import pyAudioAnalysis.ShortTermFeatures
########
import pickle
import numpy as np
import gc

def generate_label(emotion):
    label = -1
    if (emotion == 'ang'):
        label = 0
    elif (emotion == 'exc'):
        label = 1
    elif (emotion == 'fru'):
        label = 2
    elif (emotion == 'hap'):
        label = 3
    elif (emotion == 'neu'):
        label = 4
    elif (emotion == 'sad'):
        label = 5
    # elif (emotion == 'N'):
    #     label = 6
    else:
        label = None
    return label


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""

    num_labels = labels_dense.shape[0]

    index_offset = np.arange(num_labels) * num_classes

    labels_one_hot = np.zeros((num_labels, num_classes), dtype=int)

    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot

#########
def extract_max(pitches,magnitudes, shape):
    new_pitches = []
    new_magnitudes = []
    for i in range(0, shape[1]):
        new_pitches.append(np.max(pitches[:,i]))
        new_magnitudes.append(np.max(magnitudes[:,i]))
    return (new_pitches,new_magnitudes)

def smooth(x,window_len=11,window='hanning'):
        if window_len<3:
                return x
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
        s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
        if window == 'flat': #moving average
                w=np.ones(window_len,'d')
        else:
                w=eval('np.'+window+'(window_len)')
        y=np.convolve(w/w.sum(),s,mode='same')
        return y[window_len:-window_len+1]


def set_variables(sample_f,duration,window_time,fmin,fmax,overlap):
    total_samples = sample_f * duration
    #There are sample_f/1000 samples / ms
    #windowsize = number of samples in one window
    window_size = sample_f/1000 * window_time
    hop_length = total_samples / window_size
    #Calculate number of windows needed
    needed_nb_windows = total_samples / (window_size - overlap)
    n_fft = needed_nb_windows * 2.0
    return total_samples, window_size, needed_nb_windows, n_fft, hop_length

def analyse(y,sr,n_fft,hop_length,fmin,fmax):
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, S=None, n_fft= n_fft, hop_length=hop_length, fmin=fmin, fmax=fmax, threshold=0.75)
    shape = np.shape(pitches)
    #nb_samples = total_samples / hop_length
    nb_samples = shape[0]
    #nb_windows = n_fft / 2
    nb_windows = shape[1]
    pitches,magnitudes = extract_max(pitches, magnitudes, shape)

    pitches1 = smooth(pitches,window_len=10)
    pitches2 = smooth(pitches,window_len=20)
    pitches3 = smooth(pitches,window_len=30)
    pitches4 = smooth(pitches,window_len=40)
    pitches1 = np.array(pitches1)[:, np.newaxis]

    return pitches1 #,pitches1,pitches2,pitches3,pitches4,magnitudes
#####
import noisereduce as nr


# Load audio file
audio_data, sampling_rate = librosa.load()
# Noise reduction
noisy_part = audio_data[0:25000]
reduced_noise = nr.reduce_noise(audio_clip=audio_data, noise_clip=noisy_part, verbose=False)
# Visualize
# print("Original audio file:")
# plotAudio(audio_data)
# print("Noise removed audio file:")
# plotAudio(reduced_noise)
# trimmed, index = librosa.effects.trim(reduced_noise, top_db=20, frame_length=512, hop_length=64)
# print(“Trimmed audio file:”)
# plotAudio(trimmed)

def get_melspectrogram_db(file_path, sr=None, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80):
  wav,sr = librosa.load(file_path,sr=sr)
  if wav.shape[0]<5*sr:
    wav=np.pad(wav,int(np.ceil((5*sr-wav.shape[0])/2)),mode='reflect')
  else:
    wav=wav[:5*sr]
  spec=librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft,
              hop_length=hop_length,n_mels=n_mels,fmin=fmin,fmax=fmax)
  spec_db=librosa.power_to_db(spec,top_db=top_db)

  return spec_db,sr


def spec_to_image(spec, eps=1e-6):
  mean = spec.mean()
  std = spec.std()
  spec_norm = (spec - mean) / (std + eps)
  spec_min, spec_max = spec_norm.min(), spec_norm.max()
  spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
  spec_scaled = spec_scaled.astype(np.uint8)
  return spec_scaled

def features_extraction(data, sr):
    # mel_extractor = dcase_util.features.MelExtractor()
    # mel = mel_extractor.extract(y= data)
    # mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
    # chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
    # mel = librosa.feature.melspectrogram(X, sr=sample_rate)
    # contrast = librosa.feature.spectral_contrast(S=stft, sr=sample_rate)
    # tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate)
    trimmed, index = librosa.effects.trim(data, top_db=20, frame_length=512, hop_length=64)
    # print(“Trimmed audio file:”)
    stft = np.abs(librosa.stft(trimmed, n_fft=512, hop_length=256, win_length=512))

    # mfcc_delta_extraction = dcase_util.features.MfccDeltaExtractor
    # mfcc_delta = mfcc_delta_extraction.extract(y=data)
def time_frequency_domain(data,sr):
    trimmed, index = librosa.effects.trim(data, top_db=20, frame_length=512, hop_length=64)
    # print(“Trimmed audio file:”)
    stft = np.abs(librosa.stft(trimmed, n_fft=512, hop_length=256, win_length=512))
    #mel
    mel = librosa.feature.melspectrogram(y=data,sr=sr, n_mels =64)

    hop_length = 512
    frame_length = 2048
    n_fft = 1

    rmse = librosa.feature.rms(data)#(data, frame_length=frame_length, hop_length=hop_length, center=True)
    print(rmse.shape)
    # rmse = rmse[0]
    energy = np.array([
        sum(abs(data[i:i + frame_length] ** 2))
        for i in range(0, len(data), hop_length)
    ])
    print(energy.shape)
    # delta
    delta = librosa.feature.delta(mel)
    #comnbined features
    #chroma
    S = np.abs(librosa.stft(data))
    chroma = librosa.feature.chroma_stft(y=S,sr=sr)

    chroma_cens = librosa.feature.chroma_cens(y=data,sr=sr)
    print(chroma.shape,":",chroma_cens.shape)
    zrc_rate = librosa.feature.zero_crossing_rate(y=data)
    print(zrc_rate.shape)
    er = librosa.feature.rms(y=data)
    print(er.shape)
    centroid = librosa.feature.spectral_centroid(y=data,sr=sr)
    print(centroid.shape)
    mfcc = librosa.feature.mfcc(y=data,sr=sr,n_mfcc=36)
    print(mfcc.shape)
    pitch = analyse(y=data,sr=sr,n_fft=2048,hop_length=512,fmin=150.0,fmax=8000.0)
    print(pitch.shape)
    # features = np.hstack((mfcc_acc.T,mfcc_delta.T, mfcc_static.T,zrc.T,rms.T))
    features = np.hstack((mfcc.T,chroma.T,chroma_cens.T,zrc_rate.T,er.T,pitch,centroid.T))
    print(features.shape)
    print(mel.shape)
    print(delta.shape)
    return features, mel.T, delta.T




def read_IEMOCAP():
    rootdir = '/Volumes/Elements Yulia/landry_pc/IEMOCAP_full_release/'
    # print(os.listdir(rootdir))
    train_features = []
    train_label = []

    valid_features = []
    valid_label = []

    test_features = []
    test_label = []
    for folder in os.listdir(rootdir):
        #find folder that start with S as Session
        if (folder[0] == "S"):
            sub_dir = os.path.join(rootdir, folder, "sentences/wav" )
            emoevl = os.path.join(rootdir, folder, "dialog/EmoEvaluation")
            for session in os.listdir(sub_dir):
                if( session[7] == "i"):
                    emotion_dir = emoevl + "/" + session + ".txt"
                    # emotion file
                    emot_map = {}
                    with open (emotion_dir, "r") as emotion_to_read:
                        while True:
                            line = emotion_to_read.readline()
                            if not line:
                                break
                            if (line[0] == "["):
                                t = line.split()
                                emot_map[t[3]] = t[4]
                                # print(emot_map)
                    file_dir = os.path.join(sub_dir, session, "*.wav")
                    files = glob.glob(file_dir)
                    #####

                    ######
                    for filename in files:
                        wavname = filename.split("/")[-1][:-4]
                        emotion = emot_map[wavname]
                        if emotion in ["ang","exc", "fru", "hap", "neu", "sad"]:
                            if folder in ["Session1","Session3","Session4"]:
                                data, sr =  librosa.load(filename,sr=16000)
                                combine_features, mel, delta = time_frequency_domain(data, sr)
                                # training features extraction
                                train_sample = list()
                                train_sample.append(mel)
                                train_sample.append(delta)
                                train_sample.append(combine_features)
                                #training label
                                train_sample_label = generate_label(emotion)
                                print(emotion,":",train_sample_label)

                                # pack
                                train_features.append(train_sample)
                                train_label.append(train_sample_label)

                            elif folder in ["Session2"] :
                                data, sr = librosa.load(filename, sr=16000)
                                combine_features, mel, delta = time_frequency_domain(data, sr)
                                test_sample = list()
                                test_sample.append(mel)
                                test_sample.append(delta)
                                test_sample.append(combine_features)

                                # training label
                                test_sample_label = generate_label(emotion)
                                print(emotion,":",test_sample_label)

                                # pack
                                test_features.append(test_sample)
                                test_label.append(test_sample_label)
                            else :
                                data, sr = librosa.load(filename, sr=16000)
                                combine_features, mel, delta = time_frequency_domain(data, sr)
                                valid_sample = list()
                                valid_sample.append(mel)
                                valid_sample.append(delta)
                                valid_sample.append(combine_features)

                                # training label
                                valid_sample_label = generate_label(emotion)
                                print(emotion,":",valid_sample_label)

                                # pack
                                valid_features.append(test_sample)
                                valid_label.append(valid_sample_label)

    with open("training_data_emotion", "wb") as f:
        pickle.dump(train_features, f)
        pickle.dump(train_label, f)
    print( "training data done")
    with open("testing_data_emotion", "wb") as f:
        pickle.dump(test_features, f)
        pickle.dump(test_label, f)
    print("testing data done")
    with open("valid_data_emotion", "wb") as f:
        pickle.dump(valid_features, f)
        pickle.dump(valid_label, f)
    print( "valid data done")


class Config(object):
    def __init__(self):
        self.feature_dim = 64  # 特征维度

        self.segmentation_len = 300  # 分段，每一段300帧，不足补0
        self.emotion_list = ['ang', 'exc', 'fru', 'hap', 'neu', 'sad']
        self.label_map_dict = dict(zip(self.emotion_list, range(len(self.emotion_list))))
        self.is_adam = True
        self.num_epoch = 500    # 迭代的次数
        self.batch_size = 32
        self.keep_prob = 0.95
        self.learning_rate = 0.0001
        self.use_gpu = "2"  # -1表示不使用，设置使用哪一块GPU
        self.save_model_dir = r"./model_emo/"

details = Config()
segmentation_len = details.segmentation_len
feature_dim =details.feature_dim
label_len = details.emotion_list
def prepair_data():
    with open('training_data', 'rb') as handle:
        train_feature = pickle.load(handle)
        train_label = pickle.load(handle)

    # 读取训练样本
    print("read the training set features..........")
    train_feature_mel = []
    train_feature_delta1 = []
    train_feature_delta2 = []
    train_labels = []
    for idx, sample_feature in enumerate(train_feature):
        # (time_step, feature_dim)
        mel_spec = sample_feature[0]
        #      mel_spec=np.transpose(mel_spec)
        print(mel_spec.shape)
        delta1 = sample_feature[1]
        delta2 = sample_feature[2]
        print(delta1.shape,":",delta2.shape)
        # 特征矩阵按照每300帧切分为多个小矩阵，不足的补0
        if mel_spec.shape[0] < 10:
            # 小于50帧的跳过
            continue

        remainder = mel_spec.shape[0] % segmentation_len
        if 0 < remainder < 10:
            # 如果末尾不足50帧，丢弃
            mel_spec = mel_spec[:mel_spec.shape[0] - remainder]
            delta1 = delta1[:delta1.shape[0] - remainder]
            delta2 = delta2[:delta2.shape[0] - remainder]
        else:
            # 否则补0
            divide_num = int(np.ceil(mel_spec.shape[0] / segmentation_len))
            pad_num = divide_num * segmentation_len - mel_spec.shape[0]
            mel_spec = np.pad(mel_spec, ((0, pad_num), (0, 0)), "constant", constant_values=0.)
            delta1 = np.pad(delta1, ((0, pad_num), (0, 0)), "constant", constant_values=0.)
            delta2 = np.pad(delta2, ((0, pad_num), (0, 0)), "constant", constant_values=0.)

        for i in range(0, mel_spec.shape[0], segmentation_len):
            # 对于训练样本来说，每300帧就是一个样本
            train_feature_mel.append(mel_spec[i:i + segmentation_len])
            train_feature_delta1.append((delta1[i:i + segmentation_len]))
            train_feature_delta2.append((delta2[i:i + segmentation_len]))
            train_labels.append(train_label[idx])

    # 求训练集的均值和标准差
    print("calculate the mean and std of the training set..........")
    train_total_mel_spec = np.vstack(train_feature_mel)  # (time_step, feature_dim)
    mel_mean = np.mean(train_total_mel_spec, axis=0)
    mel_std = np.std(train_total_mel_spec, axis=0)

    train_total_delta1 = np.vstack(train_feature_delta1)
    delta1_mean = np.mean(train_total_delta1, axis=0)
    delta1_std = np.std(train_total_delta1, axis=0)

    train_total_delta2 = np.vstack(train_feature_delta2)
    delta2_mean = np.mean(train_total_delta2, axis=0)
    delta2_std = np.std(train_total_delta2, axis=0)

    del train_total_mel_spec  , train_total_delta1, train_total_delta2
    gc.collect()

    # 训练集特征标准化
    print("standardized the training set features..........")
    eps = 1e-5
    for index, item in enumerate(train_feature_mel):
        train_feature_mel[index] = (train_feature_mel[index] - mel_mean) / (mel_std + eps)
        train_feature_delta1[index] = (train_feature_delta1[index] - delta1_mean) / (delta1_std + eps)
        train_feature_delta2[index] = (train_feature_delta2[index] - delta2_mean) / (delta2_std + eps)
    train_size = len(train_labels)
    train_feature_mel = np.asarray(train_feature_mel)
    train_feature_delta1 = np.asarray(train_feature_delta1)
    train_feature_delta2 = np.asarray(train_feature_delta2)
    train_labels = np.asarray(train_labels)

    train_feature = np.empty((train_size, 3, segmentation_len, feature_dim), dtype=np.float32)
    train_feature[:, 0, :, :] = train_feature_mel
    train_feature[:, 1, :, :] = train_feature_delta1
    train_feature[:, 2, :, :] = train_feature_delta2
    train_labels = train_labels#dense_to_one_hot(train_labels, len(label_len))

    del train_feature_mel, train_feature_delta1, train_feature_delta2
    gc.collect()

    # 读取测试集样本
    print("read the test set features..........")
    test_feature_mel = []
    test_feature_delta1 = []
    test_feature_delta2 = []
    test_seg_nums = []
    test_seg_labels = []
    test_true_labels = []
    # for index, item in enumerate(test_pkl_paths):
    with open("testing_data", "rb") as f:
        test_feature = pickle.load(f)
        test_label = pickle.load(f)
    for idx, sample_feature in enumerate(test_feature):
        # (time_step, feature_dim)
        mel_spec = sample_feature[0]
        #mel_spec=np.transpose(mel_spec)
        delta1 = sample_feature[1]
        delta2 = sample_feature[2]

        divide_num = int(np.ceil(mel_spec.shape[0] / segmentation_len))
        pad_num = divide_num * segmentation_len - mel_spec.shape[0]
        mel_spec = np.pad(mel_spec, ((0, pad_num), (0, 0)), "constant", constant_values=0.)
        delta1 = np.pad(delta1, ((0, pad_num), (0, 0)), "constant", constant_values=0.)
        delta2 = np.pad(delta2, ((0, pad_num), (0, 0)), "constant", constant_values=0.)
        for i in range(0, mel_spec.shape[0], segmentation_len):
            # 对于训练样本来说，每300帧就是一个样本
            test_feature_mel.append(mel_spec[i:i + segmentation_len])
            test_feature_delta1.append((delta1[i:i + segmentation_len]))
            test_feature_delta2.append((delta2[i:i + segmentation_len]))
            test_seg_labels.append(test_label[idx])
        test_seg_nums.append(divide_num)
        test_true_labels.append(test_label[idx])

    # 测试集特征标准化
    test_size = np.sum(test_seg_nums)
    print("standardized the test set features..........")
    for index, item in enumerate(test_feature_mel):
        test_feature_mel[index] = (test_feature_mel[index] - mel_mean) / (mel_std + eps)
        test_feature_delta1[index] = (test_feature_delta1[index] - delta1_mean) / (delta1_std + eps)
        test_feature_delta2[index] = (test_feature_delta2[index] - delta2_mean) / (delta2_std + eps)

    test_feature_mel = np.asarray(test_feature_mel)
    test_feature_delta1 = np.asarray(test_feature_delta1)
    test_feature_delta2 = np.asarray(test_feature_delta2)
    test_seg_labels = np.asarray(test_seg_labels)
    test_seg_nums = np.asarray(test_seg_nums)

    test_feature = np.empty((test_size, 3, segmentation_len, feature_dim), dtype=np.float32)
    test_feature[:, 0, :, :] = test_feature_mel
    test_feature[:, 1, :, :] = test_feature_delta1
    test_feature[:, 2, :, :] = test_feature_delta2
    test_seg_labels = test_seg_labels#dense_to_one_hot(test_seg_labels, len(label_len))

    del test_feature_mel, test_feature_delta1, test_feature_delta2
    gc.collect()

    # 读取yanzheng集样本
    print("read the test set features..........")
    valid_feature_mel = []
    valid_feature_delta1 = []
    valid_feature_delta2 = []
    valid_seg_nums = []
    valid_seg_labels = []
    valid_true_labels = []
    # for index, item in enumerate(test_pkl_paths):
    with open("valid_data", "rb") as f:
        valid_feature = pickle.load(f)
        valid_label = pickle.load(f)
    for idx, sample_feature in enumerate(valid_feature):
        # (time_step, feature_dim)
        mel_spec = sample_feature[0]
        # mel_spec=np.transpose(mel_spec)
        delta1 = sample_feature[1]
        delta2 = sample_feature[2]

        divide_num = int(np.ceil(mel_spec.shape[0] / segmentation_len))
        pad_num = divide_num * segmentation_len - mel_spec.shape[0]
        mel_spec = np.pad(mel_spec, ((0, pad_num), (0, 0)), "constant", constant_values=0.)
        delta1 = np.pad(delta1, ((0, pad_num), (0, 0)), "constant", constant_values=0.)
        delta2 = np.pad(delta2, ((0, pad_num), (0, 0)), "constant", constant_values=0.)
        for i in range(0, mel_spec.shape[0], segmentation_len):
            # 对于训练样本来说，每300帧就是一个样本
            valid_feature_mel.append(mel_spec[i:i + segmentation_len])
            valid_feature_delta1.append((delta1[i:i + segmentation_len]))
            valid_feature_delta2.append((delta2[i:i + segmentation_len]))
            valid_seg_labels.append(valid_label[idx])
        valid_seg_nums.append(divide_num)
        valid_true_labels.append(valid_label[idx])

    # yanzheng集特征标准化
    valid_size = np.sum(valid_seg_nums)
    print("standardized the test set features..........")
    for index, item in enumerate(valid_feature_mel):
        valid_feature_mel[index] = (valid_feature_mel[index] - mel_mean) / (mel_std + eps)
        valid_feature_delta1[index] = (valid_feature_delta1[index] - delta1_mean) / (delta1_std + eps)
        valid_feature_delta2[index] = (valid_feature_delta2[index] - delta2_mean) / (delta2_std + eps)

    valid_feature_mel = np.asarray(valid_feature_mel)
    valid_feature_delta1 = np.asarray(valid_feature_delta1)
    valid_feature_delta2 = np.asarray(valid_feature_delta2)
    valid_seg_labels = np.asarray(valid_seg_labels)
    valid_seg_nums = np.asarray(valid_seg_nums)

    valid_feature = np.empty((valid_size, 3,  segmentation_len, feature_dim), dtype=np.float32)
    valid_feature[:, 0, :, :] = valid_feature_mel
    valid_feature[:, 1, :, :] = valid_feature_delta1
    valid_feature[:, 2, :, :] = valid_feature_delta2
    valid_seg_labels = valid_seg_labels#dense_to_one_hot(valid_seg_labels,len(label_len))

    del valid_feature_mel, valid_feature_delta1, valid_feature_delta2
    gc.collect()

    print("features are ready..........")
    print("total training samples:{}, true test samples:{}, test segmentations:{}".format(len(train_labels),
                                                                                          len(test_true_labels),
                                                                                          np.sum(test_seg_nums)))

    print("total training samples:{}, true valid samples:{}, valid segmentations:{}".format(len(train_labels),
                                                                                          len(valid_true_labels),
                                                                                          np.sum(valid_seg_nums)))


    print("train:", train_feature.shape, "test:", test_feature.shape, "valid:",valid_feature.shape)


    print("trainlabel:",train_labels.shape, "testlabel:",test_seg_labels.shape, "validlabel:",valid_seg_labels.shape)

    output = 'iemocap_features_cff_nodense' + '.pkl'
    # output = './IEMOCAP'+str(m)+'_'+str(filter_num)+'.pkl'
    f = open(output, 'wb')
    pickle.dump((train_feature, train_labels, test_feature,test_seg_labels, valid_feature,valid_seg_labels, test_seg_nums,valid_seg_nums), f)
    f.close()

if __name__ == "__main__":
    # read_IEMOCAP()
    prepair_data()



