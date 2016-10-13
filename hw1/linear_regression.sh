export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
python preprocess.py train.csv train.np 10 1
python preprocess.py test_X.csv test.np 10 0

python regression.py train.np test.np linear_regression.csv 1
