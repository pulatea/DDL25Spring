#!/bin/bash
cd "$(dirname "$0")" || return

for ((i=0; i<3; i=i+1))
do
    touch "out$i.txt"
    (sleep 1; python -u "intro_DP_GA.py" $i>"out$i.txt") &
done
