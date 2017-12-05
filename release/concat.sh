#! /bin/sh
#
# concat.sh
# Copyright (C) 2017 zq <zq@mclab>
#
# Distributed under terms of the MIT license.
#


test_fn="/home/zq/DATASETS/diff_vending/test_20171117.txt"
RES_DIR="result"

while IFS= read -r line
do
    echo " | Processing $line"
    convert $RES_DIR/"$line"_render_a.jpg $RES_DIR/"$line"_render_b.jpg +append $RES_DIR/"$line"_concat.jpg
    /bin/rm  $RES_DIR/"$line"_render_a.jpg $RES_DIR/"$line"_render_b.jpg
done < "$test_fn"
