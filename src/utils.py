import os
import numpy as np
import pandas as pd

def load_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
        text = text.split('\n')
    return text

def load_csv_data(filename, header=None, sep='\t+'):
    data = pd.read_csv(filename, sep=sep, header='infer' if header is None else None, engine='python')
    data.columns = header
    return data

def save_file(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(data))