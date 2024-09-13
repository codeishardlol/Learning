import torch
from torch import nn
from neural_network import  nnetwork
from torchvision import datasets
from torchvision import  transforms
if __name__=='__main__':
    transforms=transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    testdatasets=datasets.ImageFolder(root='./mnist_data/test',transform=transforms)
    model=nnetwork()
    model.load_state_dict(torch.load('mnist.pth'))

    correct=0
    wrong=0
    for idx,(data,label) in enumerate(testdatasets):
        output=model(data)
        predict=output.argmax(1).item()
        # print(predict.shape(),label.shape())
        if predict==label:
            correct+=1
        else:
            wrong+=1
    testdatasetslen=len(testdatasets)
    #correctlen=len(CORRECT)
    acc=correct/testdatasetslen
    print(f'accuracy={correct}/{testdatasetslen}={acc:.4f}')


