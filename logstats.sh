#!/bin/bash

head log1.out  | grep Arguments..Names | tr '()' '  ' | tr -d , | tr ' ' '\n' | egrep -v Namesp | egrep -v '^$'
echo ""

echo  "Last 20 epoch ave scores:"
cat $1 | awk '$1=="Episode" {SUM=SUM+$6;COUNT=COUNT+1} $1=="Training"&& SUM>0 {printf "%f\n",SUM/COUNT;SUM=0;COUNT=0}' | tail -20

echo -n "Number of parameter updates:  "
grep Train $1 | awk '{print $7}'  | st | tail -1 | awk '{print $4}'

echo -n "Total steps (=4 frames) executed: "
grep Episode $1 | awk '{print $7}' | tr -d '(' | awk '{SUM=SUM+$1;COUNT=COUNT+1} END{printf "%'"'"'d\n",  SUM/4}'

echo -n "Initial loss: "
grep loss: $1  | head -1 | tr -d '()' | awk '{print $11}'

echo -n "Current loss: "
grep loss: $1  | tail -1 | tr -d '()' | awk '{print $11}'
