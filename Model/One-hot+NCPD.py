import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import spearmanr, pearsonr
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Conv1D, MaxPool1D, AveragePooling1D, Flatten, Activation, Dense, Dropout, Bidirectional, Concatenate, Add, BatchNormalization, ReLU, GRU
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.optimizers import Adam

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

def calculate_gc_content(sequence):
    gc_count = sequence.count('G') + sequence.count('C')
    gc_content = gc_count / len(sequence)
    return gc_content

def load_data(train_file, test_file):

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

    cnn_flatten = Flatten()(data_d1)

    f1 = Dense(200, activation='relu')(cnn_flatten)
    drop1 = Dropout(0.5)(f1)
    f2 = Dense(80, activation='relu')(drop1)
    drop2 = Dropout(0.5)(f2)

    final_score = Dense(units=1, activation='linear')(drop2)

    model = tf.keras.Model(inputs=seq_input, outputs=final_score)
    return model

def main():
    train_file = r"D:\python project\相关实验-34\训练集(34nt).csv"
    test_file = r"D:\python project\相关实验-34\测试集(34nt).csv"
    train_seq, train_y, test_seq, test_y = load_data(train_file, test_file)

    # 划分训练集和验证集
    train_seq, val_seq, train_y, val_y = train_test_split(train_seq, train_y, test_size=0.2, random_state=42)

    model = build_model()
    # 调整学习率为 0.0001
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mse')

    best_model_callback = ModelCheckpoint('NCPD+One-hot编码+6.18.keras', save_best_only=True, monitor='val_loss', mode='min')

    # best_model_callback = ModelCheckpoint(
       #  'One-hot编码.keras',
       # save_best_only=False,  # 改为保存每个 epoch 的模型
       #  monitor='val_loss',  # 继续监控验证损失（可选）
       #  mode='min'  # 使用最小验证损失的模式（可选）
    # )

    batch_size = 32
    epochs = 32
    all_history = []

    for epoch in range(epochs):
        history = model.fit(train_seq, train_y, validation_data=(val_seq, val_y), batch_size=batch_size, epochs=1,
                            callbacks=[best_model_callback], shuffle=True)
        all_history.append(history.history)

        train_correlation = spearmanr(model.predict(train_seq).flatten(), train_y.flatten())[0]
        print("Epoch {}/{} - 训练Spearman相关系数：{:.4f}".format(epoch + 1, epochs, train_correlation))

        train_loss = history.history['loss'][0]
        print("Epoch {}/{} - 训练损失：{:.4f} ".format(epoch + 1, epochs, train_loss))

    predictions = model.predict(test_seq)
    test_correlation = spearmanr(predictions.flatten(), test_y.flatten())[0]
    print("测试Spearman相关系数：{:.4f}".format(test_correlation))
    test_correlation_pearson = pearsonr(predictions.flatten(), test_y.flatten())[0]
    print("测试Pearson相关系数：{:.4f}".format(test_correlation_pearson))
    # 计算均方差损失
    mse = mean_squared_error(test_y.flatten(), predictions.flatten())
    print("测试均方差损失：{:.4f}".format(mse))

    # 绘制训练损失
    loss = [h['loss'] for h in all_history]  # 获取所有epoch的训练损失值
    val_loss = [h['val_loss'] for h in all_history]  # 获取所有epoch的验证损失值
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')  # 添加验证损失曲线
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('NCPD+One-hot编码+6.18.png')
    plt.show()

if __name__ == "__main__":
    main()

# 对应权重是Bi-GRU+CNN+全连接1.keras：测试Spearman相关系数：0.7811
# 测试Pearson相关系数：0.7834
# 测试均方差损失：411.9646，  epoch=24，batchsize=24

# 对应权重是Bi-GRU+CNN+全连接3.keras：测试Spearman相关系数：0.7805
# 测试Pearson相关系数：0.7819
# 测试均方差损失：407.1051，神经元是200，80，0.5.32，32,目前这个是效果好的

# 测试Spearman相关系数：0.7800
# 测试Pearson相关系数：0.7827
# 测试均方差损失：405.6367
# 28的时候效果好像没提升
