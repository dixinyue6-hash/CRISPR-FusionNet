import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Conv1D, MaxPool1D, Flatten, Dense, Dropout, Bidirectional, Concatenate, Add
from tensorflow.keras.layers import BatchNormalization, ReLU, Activation, LSTM, GRU, GlobalAveragePooling1D, GlobalMaxPooling1D, Reshape, Multiply
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from tensorflow.keras.layers import Layer
from tensorflow.keras.optimizers import Adam

# Load Word2Vec model
model = Word2Vec.load(r'D:\python project\相关实验-34\词向量编码\3-mer\word2vec_cbow(3-mer)_model.bin')
# Function to encode sequence using Word2Vec model
def encode_sequence(sequence, model, window_size=3):
    encoded_sequence = []
    seq_len = len(sequence)
    for i in range(seq_len - window_size + 1):
        chunk = sequence[i:i+window_size]
        if chunk in model.wv:
            encoded_sequence.append(model.wv[chunk])
        else:
            encoded_sequence.append(np.zeros(model.vector_size))
    return np.array(encoded_sequence)

# Function to load data
def load_data(train_file, test_file):
    train_data = pd.read_csv(train_file, header=None, skiprows=1)
    train_seq, train_y = train_data[0], train_data[1]
    train_seq_encoded = np.array([encode_sequence(seq, model) for seq in train_seq])

    test_data = pd.read_csv(test_file, header=None, skiprows=1)
    test_seq, test_y = test_data[0], test_data[1]
    test_seq_encoded = np.array([encode_sequence(seq, model) for seq in test_seq])

    return train_seq_encoded, train_y, test_seq_encoded, test_y

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
   #  out = (out1 + out2)/2
    return out

def build_model():

    seq_input = Input(shape=(32, 50))  # Adjusted input shape
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

# Main function
def main():
    train_file = r"D:\python project\pythonProject\训练集(34nt).csv"
    test_file = r"D:\python project\pythonProject\测试集(34nt).csv"
    train_seq_encoded, train_y, test_seq_encoded, test_y = load_data(train_file, test_file)

    # Split train into train and validation
    train_seq, val_seq, train_y, val_y = train_test_split(train_seq_encoded, train_y, test_size=0.2, random_state=42)

    model1 = build_model()
    # 调整学习率为 0.0001
    optimizer = Adam(learning_rate=0.0001)
    model1.compile(optimizer=optimizer, loss='mse')

    best_model_callback = ModelCheckpoint('词向量编码11.keras', save_best_only=True,
                                          monitor='val_loss', mode='min')
    batch_size = 32
    epochs = 32
    all_history = []

    for epoch in range(epochs):
        history = model1.fit(train_seq, train_y, validation_data=(val_seq, val_y), batch_size=batch_size, epochs=1,
                             callbacks=[best_model_callback], shuffle=True)
        all_history.append(history.history)

        train_correlation = spearmanr(model1.predict(train_seq).flatten(), train_y.to_numpy().flatten())[0]

        print(f"Epoch {epoch + 1}/{epochs} - 训练Spearman相关系数：{train_correlation:.4f}")

        train_loss = history.history['loss'][0]
        print(f"Epoch {epoch + 1}/{epochs} - 训练损失：{train_loss:.4f}")

    predictions = model1.predict(test_seq_encoded)
    test_correlation = spearmanr(predictions.flatten(), test_y.to_numpy().flatten())[0]
    print(f"测试Spearman相关系数：{test_correlation:.4f}")
    test_correlation_pearson = pearsonr(predictions.flatten(), test_y.to_numpy().flatten())[0]
    print(f"测试Pearson相关系数：{test_correlation_pearson:.4f}")

    mse = mean_squared_error(test_y.values.flatten(), predictions.flatten())
    print(f"测试均方差损失：{mse:.4f}")

    loss = [h['loss'] for h in all_history]
    val_loss = [h['val_loss'] for h in all_history]
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('词向量编码11.png')
    plt.show()

if __name__ == "__main__":
    main()


# 测试Spearman相关系数：0.7804
# 测试Pearson相关系数：0.7828
# 测试均方差损失：409.7023，对应权重——词向量+CBAM(180).keras，就这个超过了，这个效果最好，但是权重没有了


# 测试Spearman相关系数：0.7817
# 测试Pearson相关系数：0.7823
# 测试均方差损失：413.4997，这个对应的catboost的结果是：
# Best Parameters: {'depth': 6, 'iterations': 150, 'learning_rate': 0.05, 'loss_function': 'RMSE'}
# Mean Squared Error: 386.3622837884072
# Spearman Correlation Coefficient: 0.7885281936633016
# 测试Pearson相关系数：0.7945

# 测试Spearman相关系数：0.7805
# 测试Pearson相关系数：0.7828
# 测试均方差损失：405.9534  对应权重——词向量+CBAM(180).keras。这个是原本效果最好的那个，但是权重找不到了

# 测试Spearman相关系数：0.7793
# 测试Pearson相关系数：0.7808
# 测试均方差损失：409.6926， 对应权重——词向量+CBAM(180)1.keras，这里对应的效果不好

# s:0.7779
# p:0.7823
# Best Parameters: {'depth': 6, 'iterations': 150, 'learning_rate': 0.05, 'loss_function': 'RMSE'}
# Mean Squared Error: 384.3828616138896
# Spearman Correlation Coefficient: 0.7882337903075862
# 测试Pearson相关系数：0.7955

# 测试Spearman相关系数：0.7806
# 测试Pearson相关系数：0.7823
# 测试均方差损失：410.0733
# Best Parameters: {'depth': 8, 'iterations': 150, 'learning_rate': 0.05, 'loss_function': 'RMSE'}
# Mean Squared Error: 382.6210891711483
# Spearman Correlation Coefficient: 0.7897095461822022
# 测试Pearson相关系数：0.7966

# Best Parameters: {'depth': 8, 'iterations': 100, 'learning_rate': 0.05, 'loss_function': 'RMSE'}
# Mean Squared Error: 380.5909740250957
# Spearman Correlation Coefficient: 0.7905033015980131
# 测试Pearson相关系数：0.7978
