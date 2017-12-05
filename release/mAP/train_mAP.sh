#! /bin/sh
# Init script for generate necessary files

CACHE_DIR="./cache/train"
RES_DIR="../result"

IM_DIR="../data/train/"          # Must with /
IMA_FN=$CACHE_DIR/traina.txt
IMB_FN=$CACHE_DIR/trainb.txt
DETA_FN=$CACHE_DIR/det_a.txt
DETB_FN=$CACHE_DIR/det_b.txt

[ -d "$CACHE_DIR" ] || mkdir -p $CACHE_DIR

### Copy detection result to cache dir
echo " | Copy detection results to cache dir..."
cp $RES_DIR/det_a.txt $DETA_FN
cp $RES_DIR/det_b.txt $DETB_FN

### Generate image text filename
echo " | Generate testing set txt file to cache dir..."
TA_FN="/tmp/traina.txt"
TB_FN="/tmp/trainb.txt"
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

python3 compute_mAP.py  $DETA_FN $DETB_FN \
                        "../data/train/{}.xml" \
                        $IMA_FN $IMB_FN



