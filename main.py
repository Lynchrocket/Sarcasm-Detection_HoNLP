import argparse
import os
import random
import numpy as np
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from sklearn.metrics import classification_report

import src.utils as utils
from src.data_processing import get_clean_data
from src.models import CNN, LSTM

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

parser = argparse.ArgumentParser()
parser.add_argument("--model",default='cnn',type=str,help="The kind of model")
parser.add_argument("--vectorize",default='skipgram',type=str,help="The kind of vectorization")
parser.add_argument("--epochs",default=10,type=int,help="number of epochs")
parser.add_argument("--bs",default=32,type=int,help="batch size")
parser.add_argument("--save",default=True,type=bool,help="whether to save")
args = parser.parse_args()

def machine_learning(train_file, test_file):
    train_tweets, train_labels, test_tweets, test_labels = get_clean_data(train_file, test_file)

    if os.path.exists('./model/tfidf.pkl'):
        vectorizer = joblib.load('./model/tfidf.pkl')
        train_x = vectorizer.transform(train_tweets).toarray()
    else:
        vectorizer = TfidfVectorizer(max_features=1000)
        train_x = vectorizer.fit_transform(train_tweets).toarray()
        joblib.dump(vectorizer, './model/tfidf.pkl')
    train_y = train_labels
    test_x = vectorizer.transform(test_tweets).toarray()
    test_y = test_labels

    model = SVC(kernel="linear")
    print('SVM is fitting')
    model.fit(train_x, train_y)
    
    print('SVM is testing')
    pred_y = model.predict(test_x)
    print(classification_report(test_y, pred_y))
    if args.save:
        joblib.dump(model, f"./model/model_{args.model}.pkl")
        # model = joblib.load("...")

def deep_learning(train_file, test_file):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'{device} is using.')

    train_tweets, train_labels, test_tweets, test_labels = get_clean_data(train_file, test_file)

    train_tokens = train_tweets.apply(lambda x: x.split())
    test_tokens = test_tweets.apply(lambda x: x.split())

    word2vec, word2int = utils.get_word2vec_model(train_tokens, model_type=args.vectorize)

    max_len = utils.get_max_len(train_tweets)
    batch_size = args.bs
    train_dataset = TextDataset(train_tweets, train_labels, max_len, word2int)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TextDataset(test_tweets, test_labels, max_len, word2int)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    embedding_dim = word2vec.vector_size
    vocab_size = len(word2int) + 1
    embedding_matrix = utils.get_word2vec_embeddings(word2vec, word2int, embedding_dim)

    num_epochs = args.epochs
    epochloop = tqdm(range(num_epochs), position=0, desc='Training', leave=True)

    if args.model == 'cnn':
        model = CNN(embedding_matrix, vocab_size, max_len, embedding_dim)
    else:
        model = LSTM(embedding_matrix, vocab_size, embedding_dim)
    model = model.to(device)
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': []
    }

    model.train()
    for epoch in epochloop:
        train_loss = 0
        train_acc = 0
        for idx, (batch_X, batch_y) in enumerate(train_dataloader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            train_loss = criterion(outputs.squeeze(), batch_y.float())

            # predicted = (outputs > 0.5).float()
            predicted = torch.sigmoid(outputs) > 0.5
            acc = (predicted == batch_y).float().mean()
            train_acc += acc.item()

            train_loss.backward()
            
            if args.model == 'lstm':
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            optimizer.step()

            if idx % 100 == 0:
                print(f"Epoch {epoch}, Step {idx}, Loss: {train_loss.item()}")

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss.detach().numpy() / len(train_dataloader))
        history['train_acc'].append(train_acc / len(train_dataloader))

    history_df = pd.DataFrame(history).set_index('epoch')
    history_df.to_csv(f'./history/train_history_{args.model}_{args.vectorize}_{random.random()}.csv')

    # test loop
    model.eval()

    # metrics
    test_loss = 0
    test_acc = 0

    all_target = []
    all_predicted = []

    testloop = tqdm(test_dataloader, leave=True, desc='Inference')
    with torch.no_grad():
        for feature, target in testloop:
            feature, target = feature.to(device), target.to(device)

            out = model(feature)
            loss = criterion(out.squeeze(), target.float())

            # predicted = (out > 0.5).float()
            predicted = (torch.sigmoid(out) > 0.5).float()
            
            equals = predicted == target

            acc = torch.mean(equals.type(torch.FloatTensor))
            test_acc += acc.item()

            test_loss += loss.item()

            all_target.extend(target.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())

        print(f'Accuracy: {test_acc/len(test_dataloader):.4f}, Loss: {test_loss/len(test_dataloader):.4f}')


    print(classification_report(all_target, all_predicted))

    if args.save:
        torch.save(model, f"./model/model_{args.model}_{args.vectorize}_{num_epochs}.pth")
        # model = torch.load("...")

if __name__ == '__main__':
    train_file = "./data/datasets/ghosh/train.txt"
    test_file = "./data/datasets/ghosh/test.txt"
    if args.model == 'svm':
        machine_learning(train_file, test_file)
    else:
        deep_learning(train_file, test_file)