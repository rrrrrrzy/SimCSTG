# SimCSTG

一种交通时空图序列预测的对比学习框架

训练（使用GWN在PEMS08上）：

```
python train.py
```

使用MTGNN在PEMS04上训练：

```
python train.py --dataset d4 --model_name MTGNN
```

训练后进行测试，将对应训练目录下的BEST_MODEL.pth文件复制到trained_model目录下对应的数据集目录中，运行：

```
python test.py --dataset d8 --model_file BEST_MODEL
```

