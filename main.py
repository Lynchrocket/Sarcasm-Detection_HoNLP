import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from tqdm import tqdm

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.manifold import TSNE

import src.utils as utils
from src.data_processing import get_clean_data
from src.models import CNN

class TextDataset(Dataset):
    def __init__(self, texts, labels, max_len, word2int):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        self.word2int = word2int
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        tokens = self.texts[idx].split(' ')[:self.max_len]
        tokens = list(map(lambda x: self.word2int.get(x, 0), tokens))
        tokens += [0] * (self.max_len - len(tokens))  # Padding
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# Data preprocessing
train_tweets, train_labels, test_tweets, test_labels = get_clean_data(
    "./data/datasets/ghosh/train_sample.txt",
    "./data/datasets/ghosh/test_sample.txt",
    save=True
)

train_tokens = train_tweets.apply(lambda x: x.split())
test_tokens = test_tweets.apply(lambda x: x.split())

word2vec, word2int = utils.get_word2vec_model(train_tokens)

max_len = utils.get_max_len(train_tweets)
batch_size = 4
train_dataset = TextDataset(train_tweets, train_labels, max_len, word2int)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TextDataset(test_tweets, test_labels, max_len, word2int)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

embedding_dim = word2vec.vector_size
vocab_size = len(word2int) + 1
embedding_matrix = utils.get_word2vec_embeddings(word2vec, word2int, embedding_dim)

num_epochs = 5
epochloop = tqdm(range(num_epochs), position=0, desc='Training', leave=True)

cnn_classifier = CNN(embedding_matrix, vocab_size, max_len, embedding_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(cnn_classifier.parameters(), lr=0.001)

cnn_classifier.train()
for epoch in epochloop:
    for idx, (batch_X, batch_y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs = cnn_classifier(batch_X)
        loss = criterion(outputs.squeeze(), batch_y.float())
        loss.backward()
        optimizer.step()

        if(idx % 100 == 0):
                print(f"Epoch {epoch}, Step {idx}, Loss: {loss.item()}")

# test loop
cnn_classifier.eval()

# metrics
test_loss = 0
test_acc = 0

all_target = []
all_predicted = []

testloop = tqdm(test_dataloader, leave=True, desc='Inference')
with torch.no_grad():
    for feature, target in testloop:
        out = cnn_classifier(feature)

        predicted = []
        out_probs = []

        predicted = torch.tensor([1 if i == True else 0 for i in out > 0.5])
        loss = criterion(out.squeeze(), target.float())
        
        equals = predicted == target

        acc = torch.mean(equals.type(torch.FloatTensor))
        test_acc += acc.item()

        test_loss += loss.item()

        all_target.extend(target.cpu().numpy())
        all_predicted.extend(predicted.cpu().numpy())

    print(f'Accuracy: {test_acc/len(test_dataloader):.4f}, Loss: {test_loss/len(test_dataloader):.4f}')


print(classification_report(all_predicted, all_target))