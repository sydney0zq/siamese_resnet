#! /bin/sh
#
# cal_mAP.sh
# Copyright (C) 2017 zq <zq@mclab>
#
# Distributed under terms of the MIT license.
#


[[ "$1" == "train" ]] && cd ./mAP && bash train_mAP.sh
[[ "$1" == "test" ]] && cd ./mAP && bash test_mAP.sh
[[ "$1" == "" ]] && cd ./mAP && bash test_mAP.sh


