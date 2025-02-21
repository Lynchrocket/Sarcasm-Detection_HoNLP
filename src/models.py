import torch
import torch.nn as nn
import torch.nn.functional as F
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
        # x = self.sigmoid(x)# 建议这里也换成和LSTM一样的操作，结果预计会更好看
        return x

class LSTM(nn.Module):
    def __init__(self, embedding_matrix, vocab_size, embed_dim, hidden_dim=64, num_layers=1): 
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32), freeze=False)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        x = self.dropout(hidden[-1])
        x = self.fc(x)
        # x = self.sigmoid(x): Sigmoid() 可能导致梯度消失或收敛到单一类别,Sigmoid() 会把输出压缩到 0~1，如果模型训练不充分，可能会全部趋向于 0（即 Non-Sarcasm）
        #在这里去掉sigmod(),在prediction里面加东西
        return x
    
class Bidirectional_LSTM(nn.Module):
    def __init__(self, embedding_matrix, vocab_size, embed_dim, hidden_dim=128, num_layers=2): 
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32), freeze=False)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        x = self.dropout(hidden[-1])
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

class BERT(nn.Module):
    def __init__(self, output_dim=1):
        super(BERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, x = self.bert(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
