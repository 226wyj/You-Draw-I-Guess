# You-Draw-I-Guess

An image classification system based on CIFAR-10.



项目依赖：

* pytorch
* PIL
* tqdm
* argparse
* flask
* Bootstrap 4.6.0



项目目录：

* data文件夹用于存放CIFAR-10数据





训练：

```python
python ./main.py --do_train --vgg --epoch=200
```



测试：

```python
python ./main.py --do_eval --vgg
```



预测：

```python
python ./main.py --do_predict --vgg
```

