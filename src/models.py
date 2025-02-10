import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Embedding, Conv1D, MaxPooling1D, Flatten, Dense, 
                                     LSTM, Bidirectional, Dropout)

def build_cnn_model(embedding_matrix, vocab_size, max_len, embedding_dim):
    """
    构建使用给定嵌入矩阵的 CNN 模型。
    可以传入由 skip-gram 或 fasttext 构造的 embedding_matrix 来分别得到不同的 CNN 模型。
    """
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size,
                        output_dim=embedding_dim,
                        weights=[embedding_matrix],
                        input_length=max_len,
                        trainable=False))
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_lstm_model(embedding_matrix, vocab_size, max_len, embedding_dim):
    """
    构建使用给定嵌入矩阵的 LSTM 模型。
    可以传入由 skip-gram 或 fasttext 构造的 embedding_matrix 来分别得到不同的 LSTM 模型。
    """
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size,
                        output_dim=embedding_dim,
                        weights=[embedding_matrix],
                        input_length=max_len,
                        trainable=False))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

"""   
class BERT(nn.Module):

    output_dim=1
    
    def __init__(self, output_dim):
        super(BERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, output_dim)

    def forward(self, x):
        _, x = self.bert(x)
        x = self.fc(x)
        return x
"""