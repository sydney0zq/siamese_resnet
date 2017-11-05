#! /bin/sh
#
# init.sh
# Copyright (C) 2017 zq <zq@mclab>
#
# Distributed under terms of the MIT license.
#


# Init script for generate necessary files

CACHE_DIR="./cache"
RES_DIR="../result"

IM_DIR="../data/test/"          # Must with /
IMA_FN="./cache/testa.txt"
IMB_FN="./cache/testb.txt"

[ -d "$CACHE_DIR" ] || mkdir -p $CACHE_DIR

### Copy detection result to cache dir
echo " | Copy detection results to cache dir..."
cp $RES_DIR/det_a.txt $RES_DIR/det_b.txt ./cache/

### Generate image text filename
echo " | Generate testing set txt file to cache dir..."
TA_FN="/tmp/testa.txt"
TB_FN="/tmp/testb.txt"
find $IM_DIR -name "*.jpg" | while read jpgfn
do
    fn=$(basename $jpgfn)
    imkey=${fn%%_*}
    echo "$imkey"_a >> $TA_FN
    echo "$imkey"_b >> $TB_FN
done
sort $TA_FN | uniq >> $IMA_FN
sort $TB_FN | uniq >> $IMB_FN

python3 compute_mAP.py



