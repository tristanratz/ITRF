nohup python -u ./train_reranker.py  >> ../data/process_output/training_reranker.log 2>&1 &
echo $! > ../data/process_output/save_pid_reranker_training.txt