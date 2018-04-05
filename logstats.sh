#!/bin/bash

echo  "Last 20 epoch ave scores:"
cat $1 | awk '$1=="Episode" {SUM=SUM+$6;COUNT=COUNT+1} $1=="Training"&& SUM>0 {printf "%f\n",SUM/COUNT;SUM=0;COUNT=0}' | tail -20

echo -n "Number of parameter updates:  "
grep Train $1 | awk '{print $7}'  | st | tail -1 | awk '{print $4}'

echo -n "Initial loss: "
grep loss log1.out  | head -1 | tr -d '()' | awk '{print $NF}'

echo -n "Current loss: "
grep loss $1  | tail -1 | tr -d '()' | awk '{print $NF}'