import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv1D, MaxPool1D, Bidirectional, Concatenate, Add, ReLU, GRU, Input,
    Flatten, Dense, Dropout, Embedding, AveragePooling1D, GlobalAveragePooling1D,
    GlobalMaxPooling1D, Activation, Reshape, Multiply, concatenate
)
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, pearsonr
from tensorflow.keras.layers import Input, Conv1D, Concatenate, Add
from tensorflow.keras.layers import Activation, LSTM, GRU, GlobalAveragePooling1D, GlobalMaxPooling1D, Reshape, BatchNormalization
from tensorflow.keras.models import Model
from catboost import CatBoostRegressor
from Bio.SeqUtils import MeltingTemp as mt
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.layers import Input, Conv1D, Concatenate, Add
from tensorflow.keras.layers import Activation, LSTM, GRU, GlobalAveragePooling1D, GlobalMaxPooling1D, Reshape
from tensorflow.keras.layers import Layer

class ChannelAttention(Layer):
    def __init__(self, reduction=3, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.reduction = reduction

    def build(self, input_shape):
        self.in_planes = input_shape[-1]
        self.avg_pool = GlobalAveragePooling1D()
        self.max_pool = GlobalMaxPooling1D()
        self.fc1 = Conv1D(filters=self.in_planes // self.reduction, kernel_size=1, use_bias=False)
        self.relu1 = Activation('relu')
        self.fc2 = Conv1D(filters=self.in_planes, kernel_size=1, use_bias=False)
        self.sigmoid = Activation('sigmoid')

    def call(self, inputs):
        avg_pool = self.avg_pool(inputs)
        avg_pool = Reshape((1, self.in_planes))(avg_pool)
        max_pool = self.max_pool(inputs)
        max_pool = Reshape((1, self.in_planes))(max_pool)

        avg_fc1 = self.fc1(avg_pool)
        avg_relu1 = self.relu1(avg_fc1)
        avg_fc2 = self.fc2(avg_relu1)
        avg_out = avg_fc2

        max_fc1 = self.fc1(max_pool)
        max_relu1 = self.relu1(max_fc1)
        max_fc2 = self.fc2(max_relu1)
        max_out = max_fc2

        out = avg_out + max_out
        out = self.sigmoid(out)
        return out


class SpatialAttention(Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv1 = Conv1D(filters=1, kernel_size=self.kernel_size, padding='same', use_bias=False)
        self.sigmoid = Activation('sigmoid')

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=2, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=2, keepdims=True)
        x = tf.concat([avg_pool, max_pool], axis=2)

        conv1 = self.conv1(x)
        out = self.sigmoid(conv1)
        return out

def cbam_block(inputs, reduction=3, kernel_size=7):
    ca = ChannelAttention(reduction=reduction)(inputs)
    sa = SpatialAttention(kernel_size=kernel_size)(inputs)
    out = inputs * ca
    out = out * sa
    return out

def calculate_melting_temp(sequence):
    melting_temp = mt.Tm_NN(sequence)
    return np.array([melting_temp])

def calculate_gc_content(sequence):
    gc_count = sequence.count('G') + sequence.count('C')
    gc_content = gc_count / len(sequence)
    return gc_content

def load_data_features(train_file, test_file):
    # 读取训练和测试数据文件
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # 提取四个特征列
    train_feature_RNA = train_data[['MFE', 'FETE', 'FMSE', 'ED']].values
    test_feature_RNA = test_data[['MFE', 'FETE', 'FMSE', 'ED']].values

    return train_feature_RNA, test_feature_RNA


def one_hot_encode(seq):
    seq = seq.upper()  # 将序列转换为大写字母
    bases = 'ACGT'
    base_dict = {base: idx for idx, base in enumerate(bases)}
    seq_length = len(seq)
    encoded_seq = np.zeros((seq_length, 4), dtype=int)

    for i, base in enumerate(seq):
        if base in base_dict:
            encoded_seq[i, base_dict[base]] = 1

    return encoded_seq

def one_hot(lines):
    length = 34
    data_n = len(lines)
    seq = np.zeros((data_n, length, 4), dtype=int)
    for l in range(data_n):
        data = lines[l]
        seq_temp = one_hot_encode(data)  # 调用 one_hot_encode 函数
        seq[l, :len(seq_temp)] = seq_temp

    return seq

def properties_code_NCPD(seqs):
    properties_code_dict = {
        'A': [1, 1, 1], 'C': [0, 1, 0], 'G': [1, 0, 0], 'T': [0, 0, 1],
        'a': [1, 1, 1], 'c': [0, 1, 0], 'g': [1, 0, 0], 't': [0, 0, 1]
    }

    properties_code = []
    for seq in seqs:
        seq = str(seq)
        seq_length = len(seq)
        properties_matrix = np.zeros([seq_length, 8], dtype=float)
        density = {base: seq.count(base) / seq_length for base in 'ACGTacgt'}
        m = 0
        for seq_base in seq:
            properties_matrix[m, :4] = one_hot_encode(seq_base)  # 调用 one_hot_encode 函数
            properties_matrix[m, 4:7] = properties_code_dict[seq_base]
            properties_matrix[m, 7] = density[seq_base]
            m += 1
        properties_code.append(properties_matrix)

    properties_code = np.array(properties_code)

    return properties_code

def load_data1(train_file, test_file):

    train_data = pd.read_csv(train_file, header=None, skiprows=1)
    train_seq, train_y = train_data[0], train_data[1]

    test_data = pd.read_csv(test_file, header=None, skiprows=1)
    test_seq, test_y = test_data[0], test_data[1]

    train_seq = properties_code_NCPD(train_seq)
    train_y = train_y.values.reshape(len(train_y), -1).astype('float32')

    test_seq = properties_code_NCPD(test_seq)
    test_y = test_y.values.reshape(len(test_y), -1).astype('float32')

    return train_seq, train_y, test_seq, test_y

def build_model():
    seq_input = Input(shape=(34, 8))  # 修改输入的shape

    lstm_layer = Bidirectional(GRU(24, return_sequences=True))(seq_input)
    lstm_output = Activation('relu')(lstm_layer)

    cnn_Conv1 = Conv1D(filters=30, kernel_size=1, padding='same', activation='relu')(lstm_output)
    cnn_Conv11 = BatchNormalization()(cnn_Conv1)

    cnn_Conv2 = Conv1D(filters=30, kernel_size=2, padding='same', activation='relu')(lstm_output)
    cnn_Conv21 = BatchNormalization()(cnn_Conv2)

    cnn_Conv3 = Conv1D(filters=30, kernel_size=3, padding='same', activation='relu')(lstm_output)
    cnn_Conv31 = BatchNormalization()(cnn_Conv3)

    cnn_Conv4 = Conv1D(filters=30, kernel_size=4, padding='same', activation='relu')(lstm_output)
    cnn_Conv41 = BatchNormalization()(cnn_Conv4)

    cnn_Conv5 = Conv1D(filters=30, kernel_size=5, padding='same', activation='relu')(lstm_output)
    cnn_Conv51 = BatchNormalization()(cnn_Conv5)

    data_t = Concatenate()([cnn_Conv11, cnn_Conv21, cnn_Conv31, cnn_Conv41, cnn_Conv51])
    data_p = MaxPool1D(strides=2, padding='same')(data_t)
    data_d1 = Dropout(0.4)(data_p)
    # data_d1_attention = cbam_block(data_d1)

    # cnn_flatten = Flatten()(data_d1_attention)
    cnn_flatten = Flatten()(data_d1)

    f1 = Dense(200, activation='relu')(cnn_flatten)
    cnn_dropout1 = Dropout(0.5)(f1)
    f2 = Dense(80, activation='relu')(cnn_dropout1)
    cnn_dropout = Dropout(0.5)(f2)

    final_score = Dense(units=1, activation='linear')(cnn_dropout)
    model = tf.keras.Model(inputs=seq_input, outputs=final_score)
    return model

def encode_sequence(sequence):
    encoding = []
    for nucleotide in sequence:
        if nucleotide == "A":
            encoding.append(2)
        elif nucleotide == "C":
            encoding.append(3)
        elif nucleotide == "G":
            encoding.append(4)
        elif nucleotide == "T":
            encoding.append(5)
    encoding = np.array(encoding)
    return encoding


def load_data2(train_file, test_file):
    train_data = pd.read_csv(train_file, header=None, skiprows=1)
    train_seq, train_y = train_data[0], train_data[1]

    test_data = pd.read_csv(test_file, header=None, skiprows=1)
    test_seq, test_y = test_data[0], test_data[1]

    train_seq_encoded = np.array([encode_sequence(seq) for seq in train_seq])
    test_seq_encoded = np.array([encode_sequence(seq) for seq in test_seq])

    # 计算每个序列的熔解温度
    train_seq_mt = np.array([calculate_melting_temp(seq) for seq in train_seq])
    test_seq_mt = np.array([calculate_melting_temp(seq) for seq in test_seq])

    # 计算GC含量
    train_gc = np.array([calculate_gc_content(seq) for seq in train_seq]).reshape(-1, 1)
    test_gc = np.array([calculate_gc_content(seq) for seq in test_seq]).reshape(-1, 1)

    # train_seq_mt = np.column_stack((train_seq_mt, train_gc))
    # test_seq_mt = np.column_stack((test_seq_mt, test_gc))

    # 将标签数据转化为numpy数组并调整形状
    train_y1 = train_y.values.reshape(len(train_y), -1).astype('float32')
    test_y1 = test_y.values.reshape(len(test_y), -1).astype('float32')

    return train_seq_encoded, train_seq_mt, train_gc, train_y1, test_seq_encoded, test_seq_mt, test_gc, test_y1

def load_data_features2(train_file, test_file):
    # 读取训练和测试数据文件
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # 提取四个特征列
    train_feature_RNA = train_data[['MFE', 'FETE', 'FMSE', 'ED']].values
    test_feature_RNA = test_data[['MFE', 'FETE', 'FMSE', 'ED']].values

    return train_feature_RNA, test_feature_RNA

def build_model2():
    input = Input(shape=(34,))
    seq_input = Embedding(16, 50, input_length=34)(input)
    gru_layer1 = Bidirectional(GRU(units=24, return_sequences=True))(seq_input)
    lstm_output = Activation('relu')(gru_layer1)

    cnn_Conv1 = Conv1D(filters=30, kernel_size=1, padding='same', activation='relu')(lstm_output)
    cnn_Conv11 = BatchNormalization()(cnn_Conv1)
    cnn_layer1 = MaxPool1D(pool_size=2)(cnn_Conv11)

    cnn_Conv2 = Conv1D(filters=30, kernel_size=2, padding='same', activation='relu')(lstm_output)
    cnn_Conv21 = BatchNormalization()(cnn_Conv2)
    cnn_layer2 = MaxPool1D(pool_size=2)(cnn_Conv21)

    cnn_Conv3 = Conv1D(filters=30, kernel_size=3, padding='same', activation='relu')(lstm_output)
    cnn_Conv31 = BatchNormalization()(cnn_Conv3)
    cnn_layer3 = MaxPool1D(pool_size=2)(cnn_Conv31)

    cnn_Conv4 = Conv1D(filters=30, kernel_size=4, padding='same', activation='relu')(lstm_output)
    cnn_Conv41 = BatchNormalization()(cnn_Conv4)
    cnn_layer4 = MaxPool1D(pool_size=2)(cnn_Conv41)

    cnn_Conv5 = Conv1D(filters=30, kernel_size=5, padding='same', activation='relu')(lstm_output)
    cnn_Conv51 = BatchNormalization()(cnn_Conv5)
    cnn_layer5 = MaxPool1D(pool_size=2)(cnn_Conv51)

    data_t = Concatenate()([cnn_layer1, cnn_layer2, cnn_layer3, cnn_layer5, cnn_layer4])
    data_d1 = Dropout(0.4)(data_t)
    cnn_flatten = Flatten()(data_d1)

    f1 = Dense(150, activation='relu')(cnn_flatten)
    cnn_dropout2 = Dropout(0.5)(f1)
    f2 = Dense(80, activation='relu')(cnn_dropout2)
    cnn_dropout = Dropout(0.5)(f2)
    final_score = Dense(units=1, activation='linear')(cnn_dropout)

    model = tf.keras.Model(inputs=input, outputs=final_score)
    return model

model2 = Word2Vec.load(r'D:\python project\pythonProject\word2vec1_model.bin')

# Function to encode sequence using Word2Vec model
def encode_sequence1(sequence, model2, window_size=2):
    encoded_sequence = []
    seq_len = len(sequence)
    for i in range(seq_len - window_size + 1):
        chunk = sequence[i:i+window_size]
        if chunk in model2.wv:
            encoded_sequence.append(model2.wv[chunk])
        else:
            encoded_sequence.append(np.zeros(model2.vector_size))
    return np.array(encoded_sequence)

# Function to load data
def load_data3(train_file, test_file):
    train_data = pd.read_csv(train_file, header=None, skiprows=1)
    train_seq, train_y3 = train_data[0], train_data[1]
    train_seq_encoded3 = np.array([encode_sequence1(seq, model2) for seq in train_seq])

    test_data = pd.read_csv(test_file, header=None, skiprows=1)
    test_seq, test_y3 = test_data[0], test_data[1]
    test_seq_encoded3 = np.array([encode_sequence1(seq, model2) for seq in test_seq])

    return train_seq_encoded3, train_y3, test_seq_encoded3, test_y3

def build_model3():

    seq_input = Input(shape=(33, 50))  # Adjusted input shape
    gru_layer = Bidirectional(GRU(24, return_sequences=True))(seq_input)
    lstm_output1 = Activation('relu')(gru_layer)
    output = cbam_block(lstm_output1)

    cnn_Conv1 = Conv1D(filters=30, kernel_size=1, padding='same', activation='relu')(output)
    cnn_Conv11 = BatchNormalization()(cnn_Conv1)

    cnn_Conv2 = Conv1D(filters=30, kernel_size=2, padding='same', activation='relu')(output)
    cnn_Conv21 = BatchNormalization()(cnn_Conv2)

    cnn_Conv3 = Conv1D(filters=30, kernel_size=3, padding='same', activation='relu')(output)
    cnn_Conv31 = BatchNormalization()(cnn_Conv3)

    cnn_Conv4 = Conv1D(filters=30, kernel_size=4, padding='same', activation='relu')(output)
    cnn_Conv41 = BatchNormalization()(cnn_Conv4)

    cnn_Conv5 = Conv1D(filters=30, kernel_size=5, padding='same', activation='relu')(output)
    cnn_Conv51 = BatchNormalization()(cnn_Conv5)

    data_t = Concatenate()([cnn_Conv11, cnn_Conv21, cnn_Conv31, cnn_Conv41, cnn_Conv51])
    data_p = MaxPool1D(strides=2, padding='same')(data_t)
    data_d1 = Dropout(0.4)(data_p)

    # data_d1_attention = cbam_block(data_d1)
    cnn_flatten = Flatten()(data_d1)

    f1 = Dense(180, activation='relu')(cnn_flatten)
    drop1 = Dropout(0.5)(f1)
    f2 = Dense(80, activation='relu')(drop1)
    cnn_dropout = Dropout(0.5)(f2)

    final_score = Dense(units=1, activation='linear')(cnn_dropout)

    model1 = tf.keras.Model(inputs=seq_input, outputs=final_score)
    return model1


def load_data_features_from_txt(train_file, test_file):

    # 读取训练和测试数据，跳过第一行标题，忽略第一列序列号
    train_data = pd.read_csv(train_file, sep="\t", skiprows=0, usecols=lambda x: x not in ['序列号'])
    test_data = pd.read_csv(test_file, sep="\t", skiprows=0, usecols=lambda x: x not in ['序列号'])

    # 提取特定的特征列
    train_feature_DNA = train_data[['Average_ProT', 'Average_HelT', 'Average_MGW', 'Average_Roll']].values
    test_feature_DNA = test_data[['Average_ProT', 'Average_HelT', 'Average_MGW', 'Average_Roll']].values

    return train_feature_DNA, test_feature_DNA

# 示例调用
train_file = "D:\python project\相关实验-34\加入DNA二级结构\HT1-1(20nt)形状.txt"
test_file = "D:\python project\相关实验-34\加入DNA二级结构\HT3(23nt)形状.txt"
train_feature_DNA, test_feature_DNA = load_data_features_from_txt(train_file, test_file)

# 文件路径
train_file_RNA = r'D:\python project\相关实验-34\5个生物特征\HT1-1.csv'
# test_file_RNA = r'D:\python project\相关实验-34\5个生物特征\HT1-2(RNA).csv'
test_file_RNA = r'D:\python project\相关实验-34\A最好\HT3(RNA).csv'
train_file = r"D:\python project\相关实验-34\训练集(34nt).csv"
test_file = r"D:\python project\相关实验-34\A最好\HT3.csv"
# test_file = r"D:\python project\相关实验-34\HT1-2(RNA).csv"

train_seq, train_y, test_seq, test_y = load_data1(train_file, test_file)
train_feature_RNA, test_feature_RNA = load_data_features2(train_file_RNA, test_file_RNA)
train_seq_encoded, train_seq_mt, train_gc, train_y1, test_seq_encoded, test_seq_mt, test_gc, test_y1 = load_data2(train_file, test_file)
train_seq_encoded3, train_y3, test_seq_encoded3, test_y3 = load_data3(train_file, test_file)

# 构建模型并加载权重
base_model = build_model()
try:
    base_model.load_weights(r'D:\python project\相关实验-34\5个生物特征\目前效果最好的模型和权重\Bi-GRU+CNN+全连接3.keras')
except ValueError as e:
    print(f"Error loading weights: {e}")
    # Exit or handle the error appropriately
    exit(1)

# 获取倒数第二层的输出作为新模型的输出
new_output = base_model.layers[-2].output
new_model = Model(inputs=base_model.input, outputs=new_output)

# 获取CNN模型的特征表示
X_train_cnn_features = new_model.predict(train_seq)
X_test_cnn_features = new_model.predict(test_seq)

train_combined_features = np.concatenate((train_feature_RNA, train_feature_DNA, train_seq_mt, train_gc, X_train_cnn_features), axis=1)
test_combined_features = np.concatenate((test_feature_RNA, test_feature_DNA, test_seq_mt, test_gc, X_test_cnn_features), axis=1)
# 这里加不加gc含量效果差不多
# 将目标变量转换成一维数组
train_y = train_y.ravel()
test_y = test_y.ravel()

# 构建随机森林回归模型，并应用给定的参数
rf_model = RandomForestRegressor(n_estimators=180,
                                  max_depth=10,
                                  min_samples_leaf=4,
                                  min_samples_split=10,
                                  random_state=42)

# 使用训练数据拟合随机森林模型
rf_model.fit(train_combined_features, train_y)
# 在测试集上进行预测
prediction1 = rf_model.predict(test_combined_features)

# 加载基础模型的权重
base_model1 = build_model2()
try:
    base_model1.load_weights(r'D:\python project\相关实验-34\再训练模型\单个嵌入编码(50)+双CNN+全连接(150).keras')
except ValueError as e:
    print(f"Error loading weights: {e}")
    exit(1)

# 获取倒数第二层的输出作为新模型的输出
new_output = base_model1.layers[-2].output
new_model1 = Model(inputs=base_model1.input, outputs=new_output)

# 获取CNN模型的特征表示
X_train_cnn_features1 = new_model1.predict(train_seq_encoded)
X_test_cnn_features1 = new_model1.predict(test_seq_encoded)

# 结合生物特征和CNN特征
train_combined_features1 = np.concatenate((train_feature_RNA, train_feature_DNA, train_seq_mt, train_gc, X_train_cnn_features1), axis=1)
test_combined_features1 = np.concatenate((test_feature_RNA, test_feature_DNA, test_seq_mt, test_gc, X_test_cnn_features1), axis=1)

# 将目标变量转换成一维数组
train_y = train_y.ravel()
test_y = test_y.ravel()

# 构建随机森林回归模型，并应用给定的参数
rf_model1 = RandomForestRegressor(n_estimators=150,
                                  max_depth=8,
                                  min_samples_leaf=4,
                                  min_samples_split=10,
                                  random_state=42)

# 使用训练数据拟合随机森林模型
rf_model1.fit(train_combined_features1, train_y)
# 在测试集上进行预测
prediction2 = rf_model1.predict(test_combined_features1)

# 加载基础模型的权重
base_model2 = build_model3()
try:
    base_model2.load_weights(r'D:\python project\相关实验-34\A最好\词向量编码.keras')
except ValueError as e:
    print(f"Error loading weights: {e}")
    exit(1)

# 获取倒数第二层的输出作为新模型的输出
new_output = base_model2.layers[-2].output
new_model2 = Model(inputs=base_model2.input, outputs=new_output)

# 获取CNN模型的特征表示
X_train_cnn_features2 = new_model2.predict(train_seq_encoded3)
X_test_cnn_features2 = new_model2.predict(test_seq_encoded3)

# 结合生物特征和CNN特征
train_combined_features2 = np.concatenate((train_feature_RNA, train_feature_DNA, train_seq_mt, train_gc, X_train_cnn_features2), axis=1)
test_combined_features2 = np.concatenate((test_feature_RNA, test_feature_DNA, test_seq_mt, test_gc, X_test_cnn_features2), axis=1)

# 将目标变量转换成一维数组
train_y = train_y.ravel()
test_y = test_y.ravel()

# 使用最佳参数初始化 CatBoostRegressor 模型
best_params = {'depth': 6, 'iterations': 150, 'learning_rate': 0.05, 'loss_function': 'RMSE'}
best_catboost_model = CatBoostRegressor(**best_params, verbose=0)
# 这里很神奇，就是200的时候不如150的效果好，但是合并起来是200的好一点

# 在选择的重要特征上训练 CatBoost 模型
best_catboost_model.fit(train_combined_features2, train_y)
# 在测试集上进行预测
prediction3 = best_catboost_model.predict(test_combined_features2)

weights = np.array([0.3, 0.4, 0.3])
# 计算加权平均预测结果
final_predictions = (weights[0] * prediction1 +
                     weights[1] * prediction2 +
                     weights[2] * prediction3)

# 计算评估指标：Spearman 相关系数、Pearson 相关系数和均方差
spearman_corr, _ = spearmanr(final_predictions, test_y)
pearson_corr, _ = pearsonr(final_predictions.ravel(), test_y.ravel())
mse = mean_squared_error(final_predictions.ravel(), test_y.ravel())

print(f"Spearman Correlation: {spearman_corr}")
print(f"Pearson Correlation: {pearson_corr}")
print(f"Mean Squared Error: {mse}")






# 0.3, 0.4, 0.3
# Spearman Correlation: 0.7849965554508781
# Pearson Correlation: 0.7836144285794145
# Mean Squared Error: 389.58564776141975  HT2


# HT1-2:
# 加R：
# Spearman Correlation: 0.7964205703825867
# Pearson Correlation: 0.8028620556371436
# Mean Squared Error: 372.01881679725466

# 加T:
# Spearman Correlation: 0.7838789276759931
# Pearson Correlation: 0.7893422520684158
# Mean Squared Error: 394.549503028378

# 加G:
# Spearman Correlation: 0.7841567231661452
# Pearson Correlation: 0.7893166423151751
# Mean Squared Error: 394.5904881915611

#HT2:
# 加R：
# Spearman Correlation: 0.7836176904847392
# Pearson Correlation: 0.78237541610011
# Mean Squared Error: 392.73793556481144

# 加T:
# Spearman Correlation: 0.7644828996616055
# Pearson Correlation: 0.7659579292304194
# Mean Squared Error: 413.23536071277323

# 加G:
# Spearman Correlation: 0.7648196134287822
# Pearson Correlation: 0.7661808231655745
# Mean Squared Error: 412.80518480759474

# 加D:
# Spearman Correlation: 0.7648292459214798
# Pearson Correlation: 0.7662217675342395
# Mean Squared Error: 412.91269271203737