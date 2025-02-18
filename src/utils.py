import os
import numpy as np
import pandas as pd
from gensim.models import Word2Vec, FastText

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
        text = text.split('\n')
    return text

def load_csv_data(filename, header=None, sep='\t+'):
    data = pd.read_csv(filename, sep=sep, header='infer' if header is None else None, engine='python', index_col=None)
    if header is not None:
        data.columns = header
    return data

def save_file(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(data))

def get_word2vec_model(sentences, model_type='skipgram'):
    if model_type == 'skipgram':
        if os.path.exists(f"{parent_dir}/model/skipgram.model"):
            word2vec = Word2Vec.load(f"{parent_dir}/model/skipgram.model")
        else:
            word2vec = Word2Vec(sentences=sentences, vector_size=100, window=5, sg=1, min_count=1, workers=4)
            word2vec.save(f"{parent_dir}/model/skipgram.model")
    else: # fasttext
        if os.path.exists(f"{parent_dir}/model/fasttext.model"):
            word2vec = FastText.load(f"{parent_dir}/model/fasttext.model")
        else:
            word2vec = FastText(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)
            word2vec.save(f"{parent_dir}/model/fasttext.model")

    word2int = { word : id for word, id in word2vec.wv.key_to_index.items() }
    return word2vec, word2int

def get_word2vec_embeddings(word2vec, word2int,embedding_size):
    embedding_matrix = np.zeros((len(word2int), embedding_size))
    for word, idx in word2int.items():
        embedding_matrix[idx] = word2vec.wv[word] if word in word2vec.wv else np.random.normal(0, 1, embedding_size)
    return embedding_matrix

def get_max_len(tweets):
    max_tweet_len = len(max(tweets, key=len).split())
    return max_tweet_len

def shuffle_words(tweets):
    shuffled = []
    for tweet in tweets:
        words = [word for word in tweet.split()]
        np.random.shuffle(words)
        shuffled.append(' '.join(words))
    return shuffled