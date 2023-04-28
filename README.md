# SimCSTG: Simple Contrastive Learning Framework for Spatio-temporal Graph Model

My undergraduate dissertation

train (with GWN onPEMS08):

```
python train.py
```

train (with MTGNN on PEMS04):

```
python train.py --dataset d4 --model_name MTGNN
```

test

```
python test.py --dataset d8 --model_file BEST_MODEL
```

