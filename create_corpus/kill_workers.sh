# Kill all processes with PID in the file
kill -9 $(cat ../data/process_output/save_pid_0.txt) $(cat ../data/process_output/save_pid_1.txt) $(cat ../data/process_output/save_pid_2.txt) $(cat ../data/process_output/save_pid_3.txt)