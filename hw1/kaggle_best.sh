python preprocess.py train.csv train.np 10 1
python preprocess.py test_X.csv test.np 10 0

python regression_kaggle_best.py train.np test.np ./kaggle_best.csv 1
