nohup accelerate launch --config_file ./accelerate_config3.yaml ./train_llm.py  >> ../data/process_output/training_llm_3.log 2>&1 &
echo $! > ../data/process_output/save_pid_llm_training.txt