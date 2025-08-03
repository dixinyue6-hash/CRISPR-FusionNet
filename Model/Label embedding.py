import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import spearmanr, pearsonr
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, Dropout, Bidirectional, Concatenate, Reshape, Add, Multiply, GlobalAveragePooling1D, GlobalMaxPooling1D, Activation
from tensorflow.keras.layers import Input, Conv1D, Concatenate, Add
from tensorflow.keras.layers import Activation, BatchNormalization, GRU, GlobalAveragePooling1D, GlobalMaxPooling1D, Reshape
from tensorflow.keras.layers import Layer
from tensorflow.keras.optimizers import Adam

# 测试Spearman相关系数：0.7731
# 测试Pearson相关系数：0.7736
# 测试均方差损失：445.8038，对应权重——单个嵌入编码+GRU(24)+CNN.keras
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
# 这里对应的数据也可以改，参考crispr-ont和crnn中编码嵌入的形式
def load_data1(train_file, test_file):

    train_data = pd.read_csv(train_file, header=None, skiprows=1)
    train_seq, train_y = train_data[0], train_data[1]

    test_data = pd.read_csv(test_file, header=None, skiprows=1)
    test_seq, test_y = test_data[0], test_data[1]

    # 对训练集和测试集的序列进行编码
    train_seq_encoded = np.array([encode_sequence(seq) for seq in train_seq])
    test_seq_encoded = np.array([encode_sequence(seq) for seq in test_seq])

    train_y = train_y.values.reshape(len(train_y), -1).astype('float32')
    test_y = test_y.values.reshape(len(test_y), -1).astype('float32')

    return train_seq_encoded, train_y, test_seq_encoded, test_y

def build_model():
    input = Input(shape=(34,))
    seq_input = Embedding(16, 50, input_length=34)(input)
    gru_layer1 = Bidirectional(GRU(units=24, return_sequences=True))(seq_input)

    lstm_output = Activation('relu')(gru_layer1)
    # lstm_output = Concatenate(name='concat')([seq_input, lstm_output])

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

    data_p = MaxPooling1D(strides=2, padding='same')(data_t)
    data_d1 = Dropout(0.4)(data_p)
    cnn_flatten = Flatten()(data_d1)

    f1 = Dense(200, activation='relu')(cnn_flatten)
    cnn_dropout2 = Dropout(0.5)(f1)
    f2 = Dense(80, activation='relu')(cnn_dropout2)
    cnn_dropout = Dropout(0.5)(f2)
    final_score = Dense(units=1, activation='linear')(cnn_dropout)

    model = tf.keras.Model(inputs=input, outputs=final_score)
    return model
def main():
    train_file = r"D:\python project\相关实验-34\训练集(34nt).csv"
    test_file = r"D:\python project\相关实验-34\测试集(34nt).csv"
    train_seq_encoded, train_y, test_seq_encoded, test_y = load_data1(train_file, test_file)

    model = build_model()
    # 调整学习率为 0.0001
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mse')

    # 定义回调函数
    best_model_callback = ModelCheckpoint('单个嵌入编码(50)+GRU+CNN.keras', save_best_only=True,
                                          monitor='val_loss', mode='min')

    batch_size = 32
    epochs = 32
    all_history = []

    # 训练最终模型
    for epoch in range(epochs):
        history = model.fit([train_seq_encoded], train_y, validation_split=0.2, batch_size=batch_size,
                            epochs=1,
                            callbacks=[best_model_callback], shuffle=True)
        all_history.append(history.history)

        train_correlation = spearmanr(model.predict([train_seq_encoded]).flatten(), train_y.flatten())[
            0]
        print("Epoch {}/{} - 训练Spearman相关系数：{:.4f}".format(epoch + 1, epochs, train_correlation))

        train_loss = history.history['loss'][0]
        print("Epoch {}/{} - 训练损失：{:.4f} ".format(epoch + 1, epochs, train_loss))

    # 评估模型性能
    predictions = model.predict([test_seq_encoded])
    test_correlation = spearmanr(predictions.flatten(), test_y.flatten())[0]
    print("测试Spearman相关系数：{:.4f}".format(test_correlation))
    test_correlation_pearson = pearsonr(predictions.flatten(), test_y.flatten())[0]
    print("测试Pearson相关系数：{:.4f}".format(test_correlation_pearson))
    mse = mean_squared_error(test_y.flatten(), predictions.flatten())
    print("测试均方差损失：{:.4f}".format(mse))

    # 绘制训练损失
    loss = [h['loss'] for h in all_history]
    val_loss = [h['val_loss'] for h in all_history]
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('单个嵌入编码(50)+GRU+CNN.png')
    plt.show()


if __name__ == "__main__":
    main()

# 测试Spearman相关系数：0.7771
# 测试Pearson相关系数：0.7777
# 测试均方差损失：417.2022, 对应权重——单个嵌入编码(50)+GRU+CNN，GRU是24个，

# 测试Spearman相关系数：0.7714
# 测试Pearson相关系数：0.7710
# 测试均方差损失：430.5753, 对应权重——单个嵌入编码(50)+GRU(16)+CNN

# 测试Spearman相关系数：0.7750
# 测试Pearson相关系数：0.7757
# 测试均方差损失：417.7373， 对于权重——单个嵌入编码(50)+GRU+CNN跳连.keras，跳连效果也还行


# 测试Spearman相关系数：0.7783
# 测试Pearson相关系数：0.7802
# 测试均方差损失：413.1569， 对应权重——单个嵌入编码(50)+GRU+CNN+全连接.keras

# 测试Spearman相关系数：0.7810
# 测试Pearson相关系数：0.7779
# 测试均方差损失：429.1914，神经元为160，对应权重——单个嵌入编码(50)+GRU+CNN+全连接1.keras


# 测试Spearman相关系数：0.7791
# 测试Pearson相关系数：0.7800
# 测试均方差损失：414.6720,对应权重——单个嵌入编码(50)+GRU+CNN+全连接2.keras.这个有点忘了是150，0.5吗

# 测试Spearman相关系数：0.7820
# 测试Pearson相关系数：0.7841
# 测试均方差损失：409.6300，对应权重——单个嵌入编码(50)+全连接+改学习率.keras.24，32，150，0.4

# 神经元240，0.5，0.5，epoch=23，32