# 线性模型

> 1. 鞍点，是所有样本求loss后对权重的梯度为0，SGD取随机的样本以计算梯度，min-ibatch，用部分数据的loss计算
> 1. 取数值时，记得用data（），loss等记得用item（），否则会累积计算图

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

> 转置的意义何在？
> $$
> y=w^tx+b还是y=xw+b
> $$



# 线性回归和logistic回归

> 分类和线性回归是不同的，例如mnist中图片呈现出哪个数字，7和9的相似度高于7和8

# 多分类和数据载入

> 没有什么问题
>
> CrossEntropyLoss和logsoftmax+NLLLoss的联系
>
> 模型复杂起来后，使用带冲量的优化器函数，momenta
>
> ReLU公式：
> $$
>  output=max(0,W^tX+B)
> $$
> 



# 卷积神经网络

> RGB-3通道 灰度-单通道，简而言之，灰度图即黑白图，损失信息用黑白图处理信息
>
> ~~3通道图为啥可以有多个卷积核，因为这些卷积核将分别与R、G、B卷积~~
>
> 3通道图：上面理解错了，卷积核通道数只能为3，分别与每个通道运算，但是可以有m个卷积核，重复上述操作
>
> 输出的维度w、h是原通道图卷积后得到的东西，例如3*3卷积核维度，那么输出维度就减2
>
> padding=填充，用于特殊需求比如输出维度一样
>
> stride有效降低图像高宽度
>
> maxpooling没懂，池化层可以改变线性关系，实现复杂特征
>

# CNN和RNN

> ​	RNN处理序列数据（时序数据、自然语言）
>
> Seq—len是序列的长度，比如输入为1个样本，3天的数据，每天的数据有4个维度，隐藏层有2个维度
>
> rnn上一层能传到下一层，代价是这个过程无法并行
>
> one-hot是？缺点：维度太高，硬编码（一一对应），过于稀疏，因此引入词嵌入
>
> 
>
> 

# 一些零碎的补充

> pytorch中，例如batch_size=10是指将1000的样本每份100个，和我理解中不同
>
> 写python代码要减少代码冗余，所以结构有相似之处，可以构建函数或者类
>
> b,c,w,h b为batch，c为通道，dim=1，即按照第一个维度进行处理
>
>  增量式开发，一步步复杂网络，保证开发正确

# 继续学习深度学习

> 1、理论上，读一些书《深度学习……》花书？
>
> 2、阅读pytorch文档
>
> 3、复现现有工作（工作：~~下载好并跑通~~，那是会配环境，要读懂代码（看网络结构技巧，），尝试自己去写）
>
> 4、选特点领域，读论文，想创新点，扩充视野
>
> 
>
> 



