import sys,os
import PIL
from tqdm import tqdm
from torchvision.datasets import MNIST

if __name__ == '__main__':
    root='mnist_data'
    if not os.path.exists(root):
        os.makedirs(root)

    train= MNIST(root='mnist_data',train=True, download=True)
    test = MNIST(root='mnist_data',train=False, download=True)
    with tqdm(total=len(train), ncols=150) as pro_bar:
        for idx, (X, y) in enumerate(train):
            f = 'mnist_data/train' + "/" +str(train[idx][1])+'/'+ "training_" + str(idx) + ".jpg" # 文件路径
            train[idx][0].save(f)
            pro_bar.update(n=1)
    with tqdm(total=len(test), ncols=150) as pro_bar:
            for idx, (X, y) in enumerate(test):
                f = 'mnist_data/test' + "/" +str(test[idx][1])+'/'+ "test_" + str(idx) + ".jpg"  # 文件路径
                test[idx][0].save(f)
                pro_bar.update(n=1)

