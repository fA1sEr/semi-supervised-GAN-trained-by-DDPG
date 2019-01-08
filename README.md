# semi supervised gan trained by DDPG
## Prerequisites
- Python 2.7 or Python 3.3+
- [Tensorflow 1.0.0](https://github.com/tensorflow/tensorflow/tree/r1.0)
- [SciPy](http://www.scipy.org/install.html)
- [NumPy](http://www.numpy.org/)

## Usage
将数据集处理成hdf5格式：
```bash
$ python preprocess.py
```

Train models:
```bash
$ python trainer.py
```

Test models with saved checkpoints:
```bash
$ python evaler.py --checkpoint ckpt_dir
```
The *ckpt_dir* should be like: ```train_dir/default-MNIST_lr_0.0001_update_G5_D1-20170101-194957/model-1001```


可视化：
```bash
$ tensorboard --logdir=train_dir
```
然后用浏览器访问 "服务器地址:6006"