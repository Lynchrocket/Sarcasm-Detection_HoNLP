#!/bin/bash

python3 main.py --model svm 
python3 main.py --model cnn --vectorize skipgram
python3 main.py --model cnn --vectorize fasttext
python3 main.py --model lstm --vectorize skipgram
python3 main.py --model lstm --vectorize fasttext