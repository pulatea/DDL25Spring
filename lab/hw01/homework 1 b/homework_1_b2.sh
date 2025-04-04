#!/bin/bash
cd "$(dirname "$0")" || return  # move to script's directory
START_TIME=$SECONDS  # start timer

for ((i=0; i<6; i=i+1))  # 3 ranks x 2 pipelines
do
    touch "out_b2_$i.txt"  # create log file for each rank
    (sleep 1; python -u "homework_1_b2.py" $i > "out_b2_$i.txt") &  # run in background
done

wait  # wait until all processes finish
echo "Elapsed time (s): $((SECONDS - START_TIME))"  # print total time