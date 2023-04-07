import os
import time
import util
import random
import torch
import argparse
import numpy as np
from gwn import gwnet
from MTGNN import gtnet
from engine import trainer

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='device name')
parser.add_argument('--dataset', type=str, default='d8', help='dataset name')
parser.add_argument('--model_name', type=str, default='GWN', help='model name: GWN or MTGNN')
parser.add_argument('--in_dim', type=int, default=1, help='input dimension')
parser.add_argument('--out_dim', type=int, default=1, help='output dimension')
parser.add_argument('--rnn_units', type=int, default=64, help='hidden dimension') # 哪里需要用到rnn
parser.add_argument('--num_layers', type=int, default=2, help='number of layer') # 什么的layers
parser.add_argument('--cheb_k', type=int, default=2, help='cheb order') # 什么
parser.add_argument('--horizon', type=int, default=12, help='sequence length')
parser.add_argument('--method', type=str, default='graph', help='two choices: pure, graph') # 原模型或者graph对比学习
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--model_file',type=str, default='', help='model_file')
args = parser.parse_args()
print(args) 



# prepare data
if args.dataset == 'd4':
    adj_data = 'data/adj_mx_04.pkl'
    input_data = 'data/PEMS-04'
    num_nodes = 307
    embed_dim = 10
elif args.dataset == 'd8':
    adj_data = 'data/adj_mx_08.pkl'
    input_data = 'data/PEMS-08'
    num_nodes = 170
    embed_dim = 2

device = torch.device(args.device)
_, _, adj_m = util.load_pickle(adj_data)
dataloader = util.load_dataset(input_data, args.batch_size, args.batch_size, args.batch_size)
scaler = dataloader['scaler']   

# 选择模型
if args.model_name == 'GWN':
    model = gwnet(device, num_nodes, dropout=0, in_dim=args.in_dim+1,
            out_dim=args.horizon, residual_channels=32,dilation_channels=32,
            skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2)
    
elif args.model_name == 'MTGNN':
    predefined_A = torch.tensor(adj_m)-torch.eye(num_nodes)
    predefined_A = predefined_A.to(device)
    model = gtnet(True, True, 2, num_nodes,
                  device, predefined_A=predefined_A,
                  dropout=0, subgraph_size=20,
                  node_dim=40,
                  dilation_exponential=1,
                  conv_channels=32, residual_channels=32,
                  skip_channels=64, end_channels= 128,
                  seq_length=12, in_dim=2, out_dim=12,
                  layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True)
    
model.to(device)
nparam = sum([p.nelement() for p in model.parameters()])
print(f'Model: {args.model_name}; total parameters: {nparam}')



model_file = args.model_file

if args.dataset == 'd8':
    model.load_state_dict(torch.load('trained_model/pem08/' + model_file + '.pth'))
elif args.dataset == 'd4':
    model.load_state_dict(torch.load('trained_model/pem04/' + model_file + '.pth'))
else:
    assert args.model_name in ['d4', 'd8'], 'Please specify the dataset: d4 or d8'

# test
outputs = []
for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
    testx = torch.Tensor(x).to(device)
    model.eval()
    with torch.no_grad():
        output, _ = model(testx)
        output = output.transpose(1,3)
        outputs.append(torch.squeeze(output, dim=1))
realy = torch.Tensor(dataloader['y_test']).to(device)
realy = realy.transpose(1,3)[:,0,:,:]
preds = torch.cat(outputs, dim=0)
preds = preds[:realy.size(0),...]


# output
test_loss = []
test_mape = []
test_rmse = []
res = []

if args.dataset == 'd8':
    f = open('results/pem08/' + model_file + '.csv', 'w')
elif args.dataset == 'd4':
    f = open('results/pem04/' + model_file + '.csv', 'w')
    
for k in range(args.horizon):
    pred = scaler.inverse_transform(preds[:,:,k])
    real = realy[:,:,k]
    metrics = util.metric(pred, real)
    log = '{:d},{:.4f},{:.4f},{:.4f}'
    print(log.format(k + 1, metrics[0], metrics[2], metrics[1]), file = f)
    test_loss.append(metrics[0])
    test_mape.append(metrics[1])
    test_rmse.append(metrics[2])
    if k in [2, 5, 11]:
        res += [metrics[0], metrics[2], metrics[1]]
mtest_loss = np.mean(test_loss)
mtest_mape = np.mean(test_mape)
mtest_rmse = np.mean(test_rmse)

log = 'Average Test MAE,{:.4f},{:.4f},{:.4f}'
print(log.format(mtest_loss, mtest_rmse, mtest_mape), file = f)
f.close()
