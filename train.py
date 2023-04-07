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
parser.add_argument('--rnn_units', type=int, default=64, help='hidden dimension')
parser.add_argument('--num_layers', type=int, default=2, help='number of layer')
parser.add_argument('--cheb_k', type=int, default=2, help='cheb order')
parser.add_argument('--horizon', type=int, default=12, help='sequence length')
parser.add_argument('--method', type=str, default='graph', help='two choices: pure, graph')
parser.add_argument('--fn_t', type=float, default=0.01, help='filter negatives threshold')
parser.add_argument('--tempe', type=float, default=0.1, help='temperature parameter')
parser.add_argument('--lam', type=float, default=0.1, help='loss lambda')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--lrate', type=float, default=0.003, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('--epochs', type=int, default=250, help='epochs')

args = parser.parse_args()
print(args) 


# sanity check
assert args.method in ['pure', 'graph'], 'Please specify the type of methods'
assert args.model_name in ['MTGNN', 'GWN'], 'Please specify the type of model'


# data
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
save = f'save_{args.model_name}_{args.dataset}_{args.method}'
if not os.path.exists(save):
    os.makedirs(save)
save += '/'
    
device = torch.device(args.device)
_, _, adj_m = util.load_pickle(adj_data)
dataloader = util.load_dataset(input_data, args.batch_size, args.batch_size, args.batch_size)
scaler = dataloader['scaler'] 


if args.model_name == 'GWN':
    model = gwnet(device, num_nodes, dropout=args.dropout, in_dim=args.in_dim+1,
            out_dim=args.horizon, residual_channels=32,dilation_channels=32,
            skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2)
    
elif args.model_name == 'MTGNN':
    predefined_A = torch.tensor(adj_m)-torch.eye(num_nodes)
    predefined_A = predefined_A.to(device)
    
    model = gtnet(True, True, 2, num_nodes,
                  device, predefined_A=predefined_A,
                  dropout=args.dropout, subgraph_size=20,
                  node_dim=40,
                  dilation_exponential=1,
                  conv_channels=32, residual_channels=32,
                  skip_channels=64, end_channels= 128,
                  seq_length=12, in_dim=2, out_dim=12,
                  layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True)

nparam = sum([p.nelement() for p in model.parameters()])
print(f'Model: {args.model_name}; total parameters: {nparam}')

engine = trainer(device, model, args.model_name, adj_m, scaler, args.method, args.fn_t, args.tempe, args.lam, args.lrate)

his_loss =[]
train_time = []
val_time = []
min_loss = float('inf')
for i in range(1, args.epochs + 1):
    # train
    train_loss = []
    train_mape = []
    train_rmse = []
    train_sloss = []
    train_uloss = []
    train_neg = []
    train_acc = []
    input_diff = []
    t1 = time.time()
    dataloader['train_loader'].shuffle()
    for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
        trainx = torch.Tensor(x).to(device)
        trainy = torch.Tensor(y).to(device)    
               
        metrics = engine.train(trainx, trainy[:,:,:,:1])
        # trainx.shape = (batchsize, sequence length, node number, 2) 2:[time of day(start), time of day(end)]

        train_loss.append(metrics[0])
        train_mape.append(metrics[1])
        train_rmse.append(metrics[2])
        train_sloss.append(metrics[3])
        train_uloss.append(metrics[4])
        train_neg.append(metrics[5])
        train_acc.append(metrics[6])
        input_diff.append(metrics[7])

         
    engine.lr_scheduler.step()   
    t2 = time.time()
    train_time.append(t2-t1)

    # validation
    valid_loss = []
    valid_mape = []
    valid_rmse = []
    s1 = time.time()
    for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testy = torch.Tensor(y).to(device)
        metrics = engine.eval(testx, testy[:,:,:,:1])

        valid_loss.append(metrics[0])
        valid_mape.append(metrics[1])
        valid_rmse.append(metrics[2])
    s2 = time.time()
    val_time.append(s2-s1)

    # train stats
    mtrain_loss = np.mean(train_loss)
    mtrain_mape = np.mean(train_mape)
    mtrain_rmse = np.mean(train_rmse)
    mtrain_sloss = np.mean(train_sloss)
    mtrain_uloss = np.mean(train_uloss)
    mtrain_neg = np.mean(train_neg)
    mtrain_acc = np.mean(train_acc)
    minput_diff = np.mean(input_diff)

    mvalid_loss = np.mean(valid_loss)
    mvalid_mape = np.mean(valid_mape)
    mvalid_rmse = np.mean(valid_rmse)
    his_loss.append(mvalid_loss)
    
    log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train SupLoss: {:.4f}, Train UnsupLoss: {:.4f}, Train UnsupAcc: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}, Train Neg: {:.4f}, Input Diff: {:.4f}, Train Time: {:.4f}/epoch, Valid Time: {:.4f}/epoch'
    print(log.format(i, mtrain_loss, mtrain_sloss, mtrain_uloss, mtrain_acc, mtrain_rmse, mtrain_mape, mvalid_loss, mvalid_rmse, mvalid_mape, mtrain_neg, minput_diff, (t2 - t1), (s2 - s1)))

    if min_loss > mvalid_loss:
        torch.save(engine.model.state_dict(), save + 'epoch_' + str(i) + '_' + str(round(mvalid_loss, 2)) + '.pth')
        min_loss = mvalid_loss


bestid = np.argmin(his_loss)
engine.model.load_state_dict(torch.load(save + 'epoch_' + str(bestid + 1) + '_' + str(round(his_loss[bestid], 2)) + '.pth'))
torch.save(engine.model.state_dict(), save + f'BEST_MODEL.pth')
his_loss = np.array(his_loss)
np.savetxt(save + 'history_loss.csv', his_loss, delimiter=',')
log = 'Best Valid MAE: {:.4f}'
print(log.format(round(his_loss[bestid], 4)))

valid_loss = []
valid_mape = []
valid_rmse = []
for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
    testx = torch.Tensor(x).to(device)
    testy = torch.Tensor(y).to(device)
    metrics = engine.eval(testx, testy[:,:,:,:1])

    valid_loss.append(metrics[0])
    valid_mape.append(metrics[1])
    valid_rmse.append(metrics[2])
mvalid_loss = np.mean(valid_loss)
mvalid_mape = np.mean(valid_mape)
mvalid_rmse = np.mean(valid_rmse)
log = 'Recheck Valid MAE: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}'
print(log.format(np.mean(mvalid_loss), np.mean(mvalid_rmse), np.mean(mvalid_mape)))

# test
outputs = []
for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
    testx = torch.Tensor(x).to(device)
    engine.model.eval()
    with torch.no_grad():
        output, _ = engine.model(testx)
        output = output.transpose(1,3)
        outputs.append(torch.squeeze(output, dim=1))
realy = torch.Tensor(dataloader['y_test']).to(device)
realy = realy.transpose(1,3)[:,0,:,:]
preds = torch.cat(outputs, dim=0)
preds = preds[:realy.size(0),...]

test_loss = []
test_mape = []
test_rmse = []
res = []
for k in range(args.horizon):
    pred = scaler.inverse_transform(preds[:,:,k])
    real = realy[:,:,k]
    metrics = util.metric(pred, real)
    log = 'Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
    print(log.format(k + 1, metrics[0], metrics[2], metrics[1]))
    test_loss.append(metrics[0])
    test_mape.append(metrics[1])
    test_rmse.append(metrics[2])
    if k in [2, 5, 11]:
        res += [metrics[0], metrics[2], metrics[1]]
mtest_loss = np.mean(test_loss)
mtest_mape = np.mean(test_mape)
mtest_rmse = np.mean(test_rmse)

log = 'Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
print(log.format(mtest_loss, mtest_rmse, mtest_mape))
res += [mtest_loss, mtest_rmse, mtest_mape]
res = [round(r, 4) for r in res]
print(res)




