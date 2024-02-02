nohup python3 -u load_cc.py --process 0 >> ../data/process_output/cc_0.log 2>&1 &
echo $! > ../data/process_output/save_pid_0.txt

nohup python3 -u load_cc.py --process 1 >> ../data/process_output/cc_1.log 2>&1 &
echo $! > ../data/process_output/save_pid_1.txt

nohup python3 -u load_cc.py --process 2 >> ../data/process_output/cc_2.log 2>&1 &
echo $! > ../data/process_output/save_pid_2.txt

nohup python3 -u load_cc.py --process 3 >> ../data/process_output/cc_3.log 2>&1 &
echo $! > ../data/process_output/save_pid_3.txt

# --cdoc 1074128 --cchunk 10659804