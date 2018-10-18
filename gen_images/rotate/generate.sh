#!/bin/bash

for input in `ls x1*.png`
do
	for angle in `seq 3 3 15`
	do
		output=$(echo $input | sed "s/.png/_rot_$angle.png/")
		echo $angle $input $output
		convert -rotate $angle $input $output
	done
done
