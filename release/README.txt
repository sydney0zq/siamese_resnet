README.txt

:Author: zq
:Email: theodoruszq@gmail.com
:Date: 2017-12-05 11:05


### PACKAGE DEP ###
1. pytorch
2. gflags

======================================================================

### 数据部分 ###

1. 在 data 文件夹，分为 train 和 test 两个，其中的文件名的格式必须按照./demotrain_tree.txt 和 test_tree.txt 的生成
2. 你必须运行bash genTrainTxt.sh; bash genTestTxt.sh 来获取文件索引，这在之后计算mAP的时候会用到

NOTE: 如果你新加入了数据，应该先使用`rename_new_data.py`进行预处理修改文件名，注意不要重复文件名


### 训练模型 ###

1. 训练模型的控制主程序在 train.py，里面的 parse 函数定义了各个参数的作用，一般常用的调用方法有
    python3 train.py --batch_size 8 --model simplesub
    # model 可以选择为 sub，simple，base 等等，具体可以看./model 文件夹，经过测试 simplesub 应该是比较好的

2. 你可以修改一些超参数来重新训练模型


### 测试模型 ###

1. 主要是 eval.py 和 eval_wo_gd.py
    eval.py 用于有 ground truth 的代码，可以在 result 中渲染出 ground truth 和 prediction 的结果
    eval_wo_gd.py 是没有 ground truth 的代码，只画出 prediction

    你可以在 eval.py 的 parse 查看有哪些参数，可以指定预测出的文件名，文件夹等等，一般使用方法
    python3 eval.py --model sub --model_fn sub_20171102.pth.tar --render 1
    # 注意你必须指定 model 的类型和 model_fn 的名字，不然代码不知道要测试哪个模型
    # render 为 1 的时候，会生成渲染图片，如果为 0，只有 prediction 的 txt 文件

2. 之后你可以使用 score_mAP.sh 来查看预测结果的好坏，你必须使用eval.py来生成./result/det_a.txt ./result/det_b.txt 这两个预测文件
    在测试集上mAP一般能达到0.85以上。

