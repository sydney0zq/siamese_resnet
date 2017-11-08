#! /bin/sh
#
# init.sh
# Copyright (C) 2017 zq <zq@mclab>
#
# Distributed under terms of the MIT license.
#

IM_DIR="./test/"          # Must with /
IMA_FN=./index/testa.txt
IMB_FN=./index/testb.txt

### Generate image text filename
echo " | Generate testing set txt file to cache dir..."
TA_FN="/tmp/testa.$$"
TB_FN="/tmp/testb.$$"
find $IM_DIR -name "*.jpg" | while read jpgfn
do
    fn=$(basename $jpgfn)
    imkey=${fn%%_*}
    echo "$imkey"_a >> $TA_FN
    echo "$imkey"_b >> $TB_FN
done

sort $TA_FN | uniq > $IMA_FN
sort $TB_FN | uniq > $IMB_FN

/bin/rm $TA_FN $TB_FN


