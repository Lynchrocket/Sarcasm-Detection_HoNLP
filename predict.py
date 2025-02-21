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
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss() #这样，模型的输出可以是 logits（未经过 sigmoid 归一化的值），而不是概率值。


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

            # predicted = (out > 0.5).float(）
            predicted = (torch.sigmoid(out) > 0.5).float()
            
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

# CNN可能存在过拟合，如果cnn的损失函数下降并趋于稳定，那么不存在过拟合。train_accuracy = 1,LSTM可能一定存在过拟合。如果还有问题：
#如果修改后的代码还有问题，可以尝试以下方法：
#双向LSTM：model = LSTM(embedding_matrix, vocab_size, embedding_dim, bidirectional=True, hidden_dim=256)
#增强隐藏单元数：model = LSTM(embedding_matrix, vocab_size, embedding_dim, hidden_dim=256, num_layers=2)

# 1.检查词向量，确保 SkipGram/FastText 真的学到了有意义的词表示。
#可能的原因：
#模型训练是否收敛

#你的 LSTM 训练过程中可能存在 梯度消失，导致模型学习不到有效特征，从而输出趋向于某一类别（Non-Sarcasm）。
#如何检查？
#观察训练 loss 是否下降，如果一直很高，说明模型没有学到有用的信息。
#观察训练集和测试集的准确率，如果训练集准确率很低，说明模型未收敛。
#二分类任务中，LSTM 可能有偏置

#你的 LSTM 只有一个 sigmoid() 输出层，而 sigmoid() 倾向于输出 靠近 0 或 1 的概率值。当模型训练不充分时，它可能默认学习到多数类（Non-Sarcasm），从而预测所有样本为 Non-Sarcasm。
#词向量质量问题

#CNN 和 LSTM 都使用相同的 SkipGram 和 FastText，但 CNN 依赖的是 局部特征（n-gram filters），而 LSTM 依赖 上下文顺序，如果词向量没有很好地捕捉上下文信息，LSTM 可能比 CNN 更难训练。
#如何检查？
#你可以随机采样一些词，看看它们的 word2vec 词向量是否合理（比如，"sarcasm" 和 "joke" 应该有较近的词向量）。
#你可以对 SkipGram 和 FastText 训练的词向量进行可视化（t-SNE 降维到 2D），看看是否形成合理的簇。
#损失函数可能不合适

#你使用的是 BCELoss()，但 LSTM 的输出是单个 sigmoid() 结果，并且 target 是 long 类型，而 BCELoss() 期望 target 是 float 类型。

#LSTM 可能存在过拟合或欠拟合
#你的 LSTM 可能层数过多，导致难以训练。
#你使用的是 hidden_dim=128 和 num_layers=2，可以尝试：
#降低 hidden_dim=64，减少模型复杂度
#只用 num_layers=1
