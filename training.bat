@REM python main.py --model svm 
@REM python main.py --model cnn --vectorize skipgram
@REM python main.py --model cnn --vectorize fasttext
python main.py --model lstm --vectorize skipgram
python main.py --model lstm --vectorize fasttext