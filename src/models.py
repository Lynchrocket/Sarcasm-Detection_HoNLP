import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class CNN(nn.Module):
    """
    vocab_size=5000, embed_dim=100, num_filters=100, kernel_sizes=[3,4,5], output_dim=1
    """
    def __init__(self, vocab_size, embed_dim, num_filters, kernel_sizes, output_dim):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=(k, embed_dim)) for k in kernel_sizes
        ])
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, output_dim)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1) # [batch, embed_dim, seq_len]
        x = [F.relu(conv(x)).max(dim=2)[0] for conv in self.convs]
        x = torch.cat(x, dim=1)
        x = self.fc(x)
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