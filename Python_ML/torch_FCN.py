# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'MachineLearning/Python_ML'))
	print(os.getcwd())
except:
	pass

#%%
import torch
import torchfcn
import matplotlib.pyplot as plt
import numpy as np
import os

# 参数初始化
def get_parameters(model, bias=False):
    import torch.nn as nn
    modules_skipped = (
        nn.ReLU,
        nn.MaxPool2d,
        nn.Dropout2d,
        nn.Sequential,
        torchfcn.models.FCN32s,
        torchfcn.models.FCN16s,
        torchfcn.models.FCN8s,
    ) 
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.ConvTranspose2d):
            # weight is frozen because it is just a bilinear upsampling
            if bias:
                assert m.bias is None
        elif isinstance(m, modules_skipped):
            continue
        else:
            raise ValueError('Unexpected module: %s' % str(m))


#%%
# 创建自己的数据集类
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        super(MyDataset, self).__init__()
        # X, Y 类初始化传入数据
        # X 维度 （图片张数，3个RGB通道，图片长，图片宽）
        # Y 维度 同上
        
        # *****************下面自己写**************************
        self.data = []      # 创建一个空列表
        self.length = len(X)    # 数据的个数（图片的张数），从X的维度中获得
        for i in range(self.length):
            self.data.append((X[i], Y[i])) # 将（X_i，y_i）加入data列表，即一组（数据-标签）对加入列表
        
    def __getitem__(self, i):
        return self.data[i]       # 返回第i个（数据-标签）组
    
    def __len__(self):
        return self.length         # 返回数据的个数 
        #******************上面自己写**************************


#%%
def load_data():
    
    #*****************下面自己写**************************
    datapath = '/home/tongxueqing/data/datasets/img-train/' # img-train 文件夹的路径
    labelpath = '/home/tongxueqing/data/datasets/img-label/' # img-label 文件夹的路径
    #******************上面自己写**************************
    
    length = len(os.listdir(datapath)) # 获取图片数量
    X = []
    Y = []
    for i in range(1, length + 1):
        
        #*****************下面自己写**************************
        data_name = 'train_%d.bmp' % i # img-train中第i图片名，e.g. train_1.bmp
        label_name = 'train_%d_anno.bmp' % i # label图片名
        #******************上面自己写**************************
        
        data_img = plt.imread(datapath + data_name)
        label_img = plt.imread(labelpath + label_name)
        
        #*****************下面自己写**************************
        X.append(data_img) # 将data_img加入X中
        Y.append(label_img) # 将label_img加入Y中

        #*****************上面自己写**************************
        
    #*****************下面自己写**************************
    X = np.array(X)
    Y = np.array(Y)
    X = X.transpose([0, 3, 1, 2]) # 转置X，将维度变为（图片张数，3个RGB通道，图片长，图片宽）
    Y = Y.transpose([0, 3, 1, 2]) # 转置Y，将维度变为（图片张数，3个RGB通道，图片长，图片宽）
    #******************上面自己写**************************
    
    train = MyDataset(X, Y)
    train_loader = torch.utils.data.DataLoader(train, shuffle = True)
    return train_loader


#%%
# 自己看example中的默认值
def main(lr = None, momentum = None, weight_decay = None, max_iteration = None):
    # 导入数据
    train_loader = load_data()
    
    # 下面一行自己写
    model = torchfcn.models.FCN32s(n_class = 2) # 创建网络模型，输入类别数，即label图像中颜色的种类
    model = model.cuda() # 调用显卡

    # 设置优化方法
    optim = torch.optim.SGD(
            [
                {'params': get_parameters(model, bias = False)},
                {'params': get_parameters(model, bias = True),
                 'lr': lr * 2, 'weight_decay': 0},
            ],
            lr = lr,
            momentum = momentum,
            weight_decay = weight_decay)
    
    # 设置训练器
    trainer = torchfcn.Trainer(
        cuda = True,
        # *****************自己写*************
        model = model,         # 网络模型
        optimizer = optim,     # 优化方法
        train_loader = train_loader,# 数据加载器
        # *************自己写*****************
        val_loader = train_loader,
        out = '/home/tongxueqing/tong/out.log',
        max_iter = max_iteration
    )
    
    # 开始训练
    trainer.epoch = 0
    trainer.iteration = 0
    trainer.train()
    
if __name__ == '__main__':
    main(lr = 1e-10, momentum = 0.99, max_iteration = 100000, weight_decay = 5e-4)


