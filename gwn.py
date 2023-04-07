import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()
        # super是用来调用父类函数的函数

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        # einsum 将x: (n,c,v,l) 和 A(v,w) 按照v的维度相乘，得到一个(n,c,w,l)的结果
        return x.contiguous()
        # 将x在内存的存储形式于其语义形式变得一致，可能是为了后续操作

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)
        # 这里用一个1x1的卷积核来做mlp（线性层）

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    # 一个GCN
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        '''
        c_in：in的channels num
        c_out：out的channels num
        dropout：比率
        support_len：暂时不知道
        order：可能是graph diffusion中的K
        '''
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        # support 可能是graph diffusion 当中的transition matrix P 以及自适应邻接矩阵A
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        # 先将(P^k)X连接起来，再进行一个线性变换，等价于对(P^k)XW求和
        h = F.dropout(h, self.dropout, training=self.training)
        # 以一定比率dropout防止过拟合
        return h


class gwnet(nn.Module):
    # Graph WaveNet
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, 
                gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,
                out_dim=12,residual_channels=32,dilation_channels=32,
                skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2):
        '''
        device: 用哪张显卡
        num_nodes: 节点个数
        dropout: dropout
        supports: 图结构提供的adj信息
        gcn_bool: 是否添加图卷积层
        addadpadj: 是否添加自适应邻接矩阵
        aptinit: 如果选择随机初始化adpt adj，就是supports[0]， 否则是None
        '''
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks  # 有多少block GWN块
        self.layers = layers  # 每个block有多少层GWN Layer
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        # 初始化一些模块，因为每个Layer都包含下列模块，且有多个Block，所以使用ModuleList来存储
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        # 进入GWN blocks之前会经过一个Linear Layer，这里用start_conv表示
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            # 如果使用图卷积层并且 使用adpt adj，
            if aptinit is None:
                # aptinit == None 即 ramdonadj == True
                if supports is None:
                    self.supports = []
                # 随机初始化A=E1 E2的E
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len +=1
            else:
                # aptinit ！= None 即 ramdonadj == False 即 adjinit = supports[0]
                if supports is None:
                    self.supports = []
                # 这种情况下使用supports[0]的奇异分解（特征分解）来生成初始的E1，E2
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1




        for b in range(blocks):
            '''
            这里的bolck和layer的理解：
            1、每个block里面有2个layer，这两个layer的dilation分别是1和2
            2、总共有4个block，即8个layers，与论文中dilation的次序为1、2、1、2、1、2、1、2一致
            3、每经过一个Layer，若想要输出维度为1，则receptive field += additional_scope
            4、additional_scope的规律：当dilation为1，即常规卷积，则每次卷积运算过后dim减少kernel-1
                当dilation为2，每次卷积dim减少 (kernel_size-1)*(dilation)
            '''
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # # dilated convolutions
                # self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                #                                    out_channels=dilation_channels,
                #                                    kernel_size=(1,kernel_size),dilation=new_dilation))

                # self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                #                                  out_channels=dilation_channels,
                #                                  kernel_size=(1, kernel_size), dilation=new_dilation))

                # # 1x1 convolution for residual connection
                # self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                #                                      out_channels=residual_channels,
                #                                      kernel_size=(1, 1)))

                # # 1x1 convolution for skip connection
                # self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                #                                  out_channels=skip_channels,
                #                                  kernel_size=(1, 1)))
                
                
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                    out_channels=dilation_channels,
                                                    kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                
                
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))

        self.receptive_field = receptive_field
        
        # regression head
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)
        
        # projection head
        self.project = nn.Sequential(
            nn.Linear(skip_channels, residual_channels),
            nn.BatchNorm1d(residual_channels),
            nn.ReLU(),
            nn.Linear(residual_channels, residual_channels)
        )


    def forward(self, input):
        input = input.transpose(1, 3)
        in_len = input.size(3)
        # print('input_shape: ', input.size())
        # print('receptive_filed: ', self.receptive_field)
        # print('in_len: ', in_len)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0)) # 加padding
        else:
            x = input
        x = self.start_conv(x) # 刚进入时的Linear变换
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            # 如果使用GCN and 有自适应邻接矩阵 and 有support
            # 计算一次新的adpadj，使用两个nodevec相乘，再relu变为正值，softmax映射到[0, 1]
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip


            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x,self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]

            x = self.bn[i](x)

        # regression head
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        # skip.shape = (B, D, N, 1)
        # x.shape = (B, 12, N, 1)
        
        # projection head
        rep = torch.squeeze(skip)
        rep = torch.sum(rep, dim=2)
        rep = self.project(rep)
        # rep.shape = (B, D)
        
        # print('encoding shape: ', skip.shape)
        # print('regression shape = ', x.shape)
        # print('projection shape = ', rep.shape)
        return x, rep
