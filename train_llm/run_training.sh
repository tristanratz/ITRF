nohup accelerate launch --config_file ./accelerate_config.yaml ./train_llm.py  >> ../data/process_output/training_llm_qlora.log 2>&1 &
echo $! > ../data/process_output/save_pid_llm_training.txt