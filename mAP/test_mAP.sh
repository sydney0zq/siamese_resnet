#! /bin/sh
#
# init.sh
# Copyright (C) 2017 zq <zq@mclab>
#
# Distributed under terms of the MIT license.
#

CACHE_DIR="./cache/test"
RES_DIR="../result"

IM_DIR="../data/test/"          # Must with /
IMA_FN=$CACHE_DIR/testa.txt
IMB_FN=$CACHE_DIR/testb.txt
DETA_FN=$CACHE_DIR/det_a.txt
DETB_FN=$CACHE_DIR/det_b.txt

[ -d "$CACHE_DIR" ] || mkdir -p $CACHE_DIR

### Copy detection result to cache dir
echo " | Copy detection results to cache dir..."
cp $RES_DIR/det_a.txt $DETA_FN
cp $RES_DIR/det_b.txt $DETB_FN

### Generate image text filename
echo " | Generate testing set txt file to cache dir..."
TA_FN="/tmp/testa.$$"
TB_FN="/tmp/testb.$$"
echo "" > $TA_FN
echo "" > $TB_FN
find $IM_DIR -name "*.jpg" | while read jpgfn
do
    fn=$(basename $jpgfn)
    imkey=${fn%%_*}
    echo "$imkey"_a >> $TA_FN
    echo "$imkey"_b >> $TB_FN
done

sort $TA_FN | uniq > $IMA_FN
sort $TB_FN | uniq > $IMB_FN

python3 compute_mAP.py  $DETA_FN $DETB_FN \
                        "/home/zq/diff_resnet/data/test/{}.xml" \
                        $IMA_FN $IMB_FN



