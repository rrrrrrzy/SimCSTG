import util
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from scipy.fftpack import dct, idct
import pysdtw


T = torch.Tensor
def pairwise_cos(x: T, y: T) -> T:
    x_norm = x.norm(dim = 2).unsqueeze(-1)
    y_norm = y.norm(dim = 2).unsqueeze(-2)
    dist = 1 - torch.einsum('abc, acd->abd', x, torch.transpose(y, 1, 2)) / torch.einsum('abc, acd->abd', x_norm, y_norm)
    return torch.clamp(dist, 0.0, 2)

def cos_dtw_filter_negative(input_, thres):    
    fun = pairwise_cos
    sdtw = pysdtw.SoftDTW(gamma=0.0000001, dist_func=fun, use_cuda=True)
    
    xt = torch.Tensor(input_)[..., 0]
    x1 = xt.repeat(xt.shape[0], 1, 1)
    x2 = xt.repeat(1, xt.shape[0], 1).reshape(-1, xt.shape[1], xt.shape[2])
    dtw = sdtw(x1, x2)  / xt.shape[1]
    dtw = dtw.reshape(xt.shape[0], xt.shape[0])
    m = torch.ge(dtw, thres)
    m.fill_diagonal_(True)
    
    return m


class trainer():
    def __init__(self, device, model, model_name, adj_m, scaler, method, 
                 fn_t, tempe, lam, lrate):
        self.device = device
        self.model = model
        self.model.to(device)
        for p in self.model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.uniform_(p)
        self.scaler = scaler
        self.loss = util.masked_mae
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate)
        self.lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)
        self.nor_adj = util.asym_adj(adj_m)
        self.method = method
        self.model_name = model_name
        self.fn_t = fn_t
        self.tempe = tempe
        self.lam = lam

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        bs = input.shape[0]
        frame = input.shape[1]
        num_node = input.shape[2]
        

        output, rep = self.model(input)
        predict = self.scaler.inverse_transform(output)
        s_loss = self.loss(predict, real_val, 0.0)
        mape = util.masked_mape(predict, real_val, 0.0).item()
        rmse = util.masked_rmse(predict, real_val, 0.0).item()
        
        # 不进行对比
        if self.method == 'pure':
            s_loss.backward()
            self.optimizer.step()
            return s_loss.item(), mape, rmse, 0, 0, 0, 0, 0
        
        elif self.method == 'graph':
            diff = 0
            
            # data arg
            input_ = input.detach().clone()
            _, aug_rep = self.model(input_)
            norm1 = rep.norm(dim=1) # rep.shape = (B, D)
            norm2 = aug_rep.norm(dim=1) 
            sim_matrix = torch.mm(rep, torch.transpose(aug_rep, 0, 1)) / torch.mm(norm1.view(-1, 1), norm2.view(1, -1))
            sim_matrix = torch.exp(sim_matrix / self.tempe)
            diag = bs
            pos_sim = sim_matrix[range(diag), range(diag)]
            avg_neg = diag - 1
            
            # filter negative
            m = cos_dtw_filter_negative(input, self.fn_t)
            s = torch.sum(m, dim=1) - 1
            avg_neg = torch.mean(s * 1.0).cpu().item()
            sim_matrix = sim_matrix * m
            max_id = torch.argmax(sim_matrix, dim=1)
            labels = torch.arange(diag).to(self.device)
            corr_num = torch.sum(max_id==labels).item()
            avg_acc = corr_num / diag
            u_loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            u_loss = torch.mean(-torch.log(u_loss))

            # loss
            loss = s_loss + self.lam * u_loss
            loss.backward()
            self.optimizer.step()
            return loss.item(), mape, rmse, s_loss.item(), u_loss.item(), avg_neg, avg_acc, diff
    
    def eval(self, input, real_val):
        self.model.eval()
        with torch.no_grad():
            output, _ = self.model(input)
            predict = self.scaler.inverse_transform(output)
            loss = self.loss(predict, real_val, 0.0)
            mape = util.masked_mape(predict, real_val, 0.0).item()
            rmse = util.masked_rmse(predict, real_val, 0.0).item()
            return loss.item(), mape, rmse
