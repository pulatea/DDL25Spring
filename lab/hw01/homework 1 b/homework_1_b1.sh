#!/bin/bash
cd "$(dirname "$0")" || return  # move to script's directory
START_TIME=$SECONDS  # start timer

for ((i=0; i<3; i=i+1))  # iterate over 3 ranks 0, 1, 2
do
    touch "out_b1_$i.txt"  # create log file for each rank
    (sleep 1; python -u "homework_1_b1.py" $i > "out_b1_$i.txt") &  # run in background
done

wait  # wait until all processes finish
echo "Elapsed time (s): $((SECONDS - START_TIME))"  # print total time