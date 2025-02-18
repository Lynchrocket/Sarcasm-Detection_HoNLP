import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

import src.utils as utils
from src.data_processing import get_clean_data

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

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Sarcasm", "Sarcasm"], 
                yticklabels=["Non-Sarcasm", "Sarcasm"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.savefig(f'./visualization/full_dataset/{title}.png')

def show_classification_report(y_true, y_pred, title):
    report = classification_report(y_true, y_pred, target_names=["Non-Sarcastic", "Sarcastic"], output_dict=True)
    print(report)
    df = pd.DataFrame(report).transpose()
    df.to_csv(f'./visualization/full_dataset/{title}.csv', index=True)

def predict(train_file, test_file, model_name, model_file_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'{device} is using.')

    train_tweets, train_labels, test_tweets, test_labels = get_clean_data(train_file, test_file)

    train_tokens = train_tweets.apply(lambda x: x.split())
    test_tokens = test_tweets.apply(lambda x: x.split())

    word2vec, word2int = utils.get_word2vec_model(train_tokens, model_type=model_name.split('-')[1].lower())

    max_len = utils.get_max_len(train_tweets)
    batch_size = 32
    test_dataset = TextDataset(test_tweets, test_labels, max_len, word2int)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = torch.load(model_file_path, weights_only=False)
    model = model.to(device)
    criterion = nn.BCELoss()

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

            predicted = []

            predicted = (out > 0.5).float()
            loss = criterion(out.squeeze(), target.float())
            
            equals = predicted == target

            acc = torch.mean(equals.type(torch.FloatTensor))
            test_acc += acc.item()

            test_loss += loss.item()

            all_target.extend(target.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())

        print(f'Accuracy: {test_acc/len(test_dataloader):.4f}, Loss: {test_loss/len(test_dataloader):.4f}')

    plot_confusion_matrix(all_target, all_predicted, title=f"Confusion Matrix - {model_name}")
    print(f"\nClassification Report for {model_name}:")
    show_classification_report(all_target, all_predicted, title=f"{model_name}-Report")


if __name__ == '__main__':
    train_file = "./data/datasets/ghosh/train.txt"
    test_file = "./data/datasets/ghosh/test.txt"
    model_paths = {
        "CNN-SkipGram": "./model/model_cnn_skipgram_10.pth",
        "LSTM-SkipGram": "./model/model_lstm_skipgram_10.pth",
        "CNN-FastText": "./model/model_cnn_fasttext_10.pth",
        "LSTM-FastText": "./model/model_lstm_fasttext_10.pth"
    }
    for model_name, model_file in model_paths.items():
        predict(train_file, test_file, model_name, model_file)