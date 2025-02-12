import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import BertModel

class CNN(nn.Module):
    def __init__(self, embedding_matrix, vocab_size, max_len, embedding_dim):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32), freeze=False)
        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * ((max_len - 5 + 1) // 2), 128)  
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
class LSTM(nn.Module):
    """
    vocab_size=5000, embed_dim=100, hidden_dim=128, output_dim=1, num_layers=2
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, output_dim): 
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        x = self.dropout(hidden[-1])
        x = self.fc(x)
        return x
    
# def build_lstm_model(embedding_matrix, vocab_size, max_len, embedding_dim):
#     """
#     构建使用给定嵌入矩阵的 LSTM 模型。
#     可以传入由 skip-gram 或 fasttext 构造的 embedding_matrix 来分别得到不同的 LSTM 模型。
#     """
#     model = Sequential()
#     model.add(Embedding(input_dim=vocab_size,
#                         output_dim=embedding_dim,
#                         weights=[embedding_matrix],
#                         input_length=max_len,
#                         trainable=False))
#     model.add(Bidirectional(LSTM(128, return_sequences=True)))
#     model.add(Bidirectional(LSTM(64)))
#     model.add(Dense(128, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return model

class BERT(nn.Module):
    """
    output_dim=1
    """
    def __init__(self, output_dim):
        super(BERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, output_dim)

    def forward(self, x):
        _, x = self.bert(x)
        x = self.fc(x)
        return x
    
# def build_cnn_model(embedding_matrix, vocab_size, max_len, embedding_dim):
#     """
#     构建使用给定嵌入矩阵的 CNN 模型。
#     可以传入由 skip-gram 或 fasttext 构造的 embedding_matrix 来分别得到不同的 CNN 模型。
#     """
#     model = Sequential()
#     model.add(Embedding(input_dim=vocab_size,
#                         output_dim=embedding_dim,
#                         weights=[embedding_matrix],
#                         input_length=max_len,
#                         trainable=False))
#     model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
#     model.add(MaxPooling1D(pool_size=2))
#     model.add(Flatten())
#     model.add(Dense(128, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return model




if __name__=='__main__':

    ...