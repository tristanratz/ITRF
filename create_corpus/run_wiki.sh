nohup python3 -u load_wiki.py --process 0 --console True >> ../data/process_output/wiki_0.log 2>&1 &
echo $! > ../data/process_output/save_pid_0.txt

nohup python3 -u load_wiki.py --process 1 --console True >> ../data/process_output/wiki_1.log 2>&1 &
echo $! > ../data/process_output/save_pid_1.txt

nohup python3 -u load_wiki.py --process 2 --console True >> ../data/process_output/wiki_2.log 2>&1 &
echo $! > ../data/process_output/save_pid_2.txt

nohup python3 -u load_wiki.py --process 3 --console True >> ../data/process_output/wiki_3.log 2>&1 &
echo $! > ../data/process_output/save_pid_3.txt
