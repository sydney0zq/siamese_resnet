#! /bin/sh
#
# init.sh
# Copyright (C) 2017 zq <zq@mclab>
#
# Distributed under terms of the MIT license.
#

CACHE_DIR="./cache/test"
RES_DIR="../result"

IM_INDEX_DIR="../data/index"
IMA_FN=$CACHE_DIR/testa.txt
IMB_FN=$CACHE_DIR/testb.txt
DETA_FN=$CACHE_DIR/det_a.txt
DETB_FN=$CACHE_DIR/det_b.txt

[ -d "$CACHE_DIR" ] || mkdir -p $CACHE_DIR

### Copy detection result to cache dir
echo " | Copy detection results and image index to cache dir..."
cp $RES_DIR/det_a.txt $DETA_FN
cp $RES_DIR/det_b.txt $DETB_FN
cp $IM_INDEX_DIR/testa.txt $IMA_FN
cp $IM_INDEX_DIR/testb.txt $IMB_FN

### Generate image text filename
python3 compute_mAP.py  $DETA_FN $DETB_FN \
                        "../data/test/{}.xml" \
                        $IMA_FN $IMB_FN



