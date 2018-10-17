#!/bin/bash
echo "script"

for a in `ls *.conf`
do
	patchy2d $a -s 100000 -e 20 --pmod 100 --mod 10 --new_width 20
done
