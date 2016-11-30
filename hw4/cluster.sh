data_dir=$1
output_file=$2

python2.7 preprocessing.py $data_dir/title_StackOverflow.txt ./clear_title.txt
python2.7 clustering_word2vec_cosine_kmeans.py --predict $output_file --index $data_dir/check_index.csv
