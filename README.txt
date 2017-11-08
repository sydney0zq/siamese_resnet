README.txt

:Author: zq
:Email: zq@mclab
:Date: 2017-10-27 20:35





   


































=======2017-10-28 23:06(Saturday)=======
这两天做了差值的图像进行训练，和没有做差值的图像进行训练。结果都很差，网络并没有学到什么东西。

我不知道是网络出了问题还是我代码中有一些问题。我目前感觉我的代码是没有问题的。因为基本都可视化了。

=======2017-11-05 20:30(Sunday)=======
这几天做了mAP的效果。但是AP是非常低不知道为什么，在训练集上只有0.17左右，并且在测试集上只有0.9。这是非常糟糕的结果，还需要进一步找到问题所在。





### ISSUES ###
1. 做差分后，不应该再做normalize了（pytorch库提供的），首先ToTensor已经将其归一化的，第二，归一化没有什么意义
2. L1 loss可以应对比较稀疏的输入；而L2 Loss就是尽量保证网络权重不为0
3. 检测后得到的结果要保持一致，本项目一直保持midx, midy, w, h









=======2017-10-27 20:35(Friday)=======
Some details about the ground truth label and predict result format:

1.
    Load pair label by dataloader:
        midx, midy, w, h
    Detection result:
        midx, midy, w, h
2.
    imkey should be an integer
3.
    Model should be not pretrained
