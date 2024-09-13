# 线性模型

> 1. 鞍点，是所有样本求loss后对权重的梯度为0，SGD取随机的样本以计算梯度，min-ibatch，用部分数据的loss计算

# 反向传播

> 1. 反向传播时，对上一层求偏导过程，激活函数也是作为链式法则一起求吗
>
> 2. 如何通过反向传播、学习率，调整各个层的权重值
>
> 3. backward在torch里的实现原理不清楚，导致无法理解什么时候在哪个位置该出现backward调用，如下代码：
>
>    ```python
>    import torch
>    x_data = [1.0, 2.0, 3.0]
>    y_data = [2.0, 4.0, 6.0]
>    w = torch.tensor([1.0]) # w的初值为1.0
>    w.requires_grad = True # 需要计算梯度
>    def forward(x):
>        return x*w  # w是一个Tensor
>    def loss(x, y):
>        y_pred = forward(x)
>        return (y_pred - y)**2r
>    print("predict (before training)", 4, forward(4).item())
>    for epoch in range(100):
>        for x, y in zip(x_data, y_data):
>            l =loss(x,y) # l是一个张量，tensor主要是在建立计算图 forward, compute the loss
>            l.backward() #  backward,compute grad for Tensor whose requires_grad set to True
>            print('\tgrad:', x, y, w.grad.item())
>            w.data = w.data - 0.01 * w.grad.data   # 权重更新时，注意grad也是一个tensor
>            w.grad.data.zero_() # after update, remember set the grad to zero
>        print('progress:', epoch, l.item()) # 取出loss使用l.item，不要直接使用l（l是tensor会构建计算图）
>    print("predict (after training)", 4, forward(4).item())
>    ```
>
>    为啥backward出现在那个位置呢
>
> 4. 张量的grad是对所有的layer都进行计算，那么最后的权重是只计算了一开始的权重w吗
>
> 5. 
# 前向传播


