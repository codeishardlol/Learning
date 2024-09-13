import torch
import time
import torch.nn as nn
from torch import optim
from neural_network import nnetwork
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

if __name__=='__main__':
    starttime=time.time()
    transforms = transforms.Compose(
        [transforms.Grayscale(num_output_channels=1),
         transforms.ToTensor()]
    )

    train_datasets=datasets.ImageFolder(root='./mnist_data/train',transform=transforms)
    test_datasets=datasets.ImageFolder(root='./mnist_data/test',transform=transforms)
    #print("traindata_length:",len(train_datasets),'\n',"testdata_length:",len(test_datasets))
    train_loader=DataLoader(train_datasets,batch_size=64,shuffle=True)
    #print("train_loader length:" len(train_loader)
    model=nnetwork()
    #model2=try_nnetwork()
    model1=nn.Linear(784,32)
    optimizer=torch.optim.Adam(model.parameters(),lr=0.005)
    criterion=nn.CrossEntropyLoss()

    for epoch in range(50):
         for batch_idx,(data,label) in enumerate(train_loader):
            #         if batch_idx==3:
            #             break
            #         print(data.shape,label.shape,label)
            output=model(data)
            loss=criterion(output,label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if batch_idx %100==0:
                print(f'Epoch:{epoch+1},Batch:{batch_idx/len(train_loader)},Loss:{loss.item():.4f}')
    torch.save(model.state_dict(),'mnist.pth')
    endtime=time.time()
    print(f'process finished! time in {endtime-starttime} sec')
    