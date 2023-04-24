import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm
import numpy as np
from os.path import join
from os import remove
import h5py
from math import ceil

## 算cache时用whole_training_data_loader,取的是descData里预生成的npy格式描述符，训练时用 training_data_loader，取的时cache写入的h5文件,然后定位出positive和negatives的位置，再去npy里取数据

def train(opt, model, encoder_dim, device, dataset, criterion, optimizer, train_set, whole_train_set, whole_training_data_loader, epoch, writer):  #criterion 准则
    epoch_loss = 0
    startIter = 1 # keep track of batch iter across subsets for logging

    if opt.cacheRefreshRate > 0:
        subsetN = ceil(len(train_set) / opt.cacheRefreshRate)
        #TODO randomise the arange before splitting?
        subsetIdx = np.array_split(np.arange(len(train_set)), subsetN)
    else:
        subsetN = 1
        subsetIdx = [np.arange(len(train_set))]  #编号[0,1,2,3...]
        #print("train_set len:" + str(len(train_set))) #15000

    nBatches = (len(train_set) + opt.batchSize - 1) // opt.batchSize  #batchSize defualt 16

    for subIter in range(subsetN):  #subIter:0
        print('====> Building Cache')
        model.eval()
        with h5py.File(train_set.cache, mode='w') as h5:   #每个epoch新建这个cache，然后在epoch结束后删掉，如果异常退出将不删
            pool_size = encoder_dim
            #print("pool_size:" +  str(pool_size)
            if opt.pooling.lower() == 'seqnet':
                pool_size = opt.outDims  #4096
            h5feat = h5.create_dataset("features", [len(whole_train_set), pool_size], dtype=np.float32)
            #print("whole_training_data_loader:" + str(len(whole_training_data_loader))) #1250
            with torch.no_grad():  #猜测：将整个数据集每个图片（是预生成的描述符）用当前seqnet计算描述符，存至h5feat
                for iteration, (input, indices) in tqdm(enumerate(whole_training_data_loader, 1),total=len(whole_training_data_loader)-1, leave=False):
                    #print("input:" +  str(np.shape(input))) #[24, 10, 4096]
                    #print(indices) # 每次取24个，依次后推直至29999 [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
                    image_encoding = (input).float().to(device)
                    seq_encoding = model.pool(image_encoding)
                    h5feat[indices.detach().numpy(), :] = seq_encoding.detach().cpu().numpy()
                    del input, image_encoding, seq_encoding
        #print("subsetIdx")
        #print(subsetIdx) #[array([    0,     1,     2, ..., 14997, 14998, 14999])]
        #print(subsetIdx[subIter]) #    0     1     2 ... 14997 14998 14999]
        sub_train_set = Subset(dataset=train_set, indices=subsetIdx[subIter])  #subsetN 为1，这里取的是所有数据
        #print("sub_train_set:" + str(np.shape(sub_train_set)))

        training_data_loader = DataLoader(dataset=sub_train_set, num_workers=opt.threads, 
                    batch_size=opt.batchSize, shuffle=True, 
                    collate_fn=dataset.collate_fn, pin_memory=not opt.nocuda)

        print('Allocated:', torch.cuda.memory_allocated())
        print('Cached:', torch.cuda.memory_reserved())

        model.train()
        for iteration, (query, positives, negatives, 
                negCounts, indices) in tqdm(enumerate(training_data_loader, startIter),total=len(training_data_loader),leave=False):
            loss = 0
            if query is None:
                continue # in case we get an empty batch
            B = query.shape[0]
            print("query: " + str(query.shape)) #torch.Size([16, 10, 4096])
            print("positives: " + str(positives.shape)) #torch.Size([16, 10, 4096])
            print("negatives: " + str(negatives.shape))#torch.Size([160, 10, 4096])
            print("negCounts: " + str(negCounts))  #tensor([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
            nNeg = torch.sum(negCounts)

            input = torch.cat([query,positives,negatives]).float()
            input = input.to(device)
            seq_encoding = model.pool(input)


            seqQ, seqP, seqN = torch.split(seq_encoding, [B, B, nNeg]) #[B, B, nNeg] [16,16,160]

            optimizer.zero_grad()
            
            # calculate loss for each Query, Positive, Negative triplet
            # due to potential difference in number of negatives have to 
            # do it per query, per negative
            for i, negCount in enumerate(negCounts):
                for n in range(negCount):
                    negIx = (torch.sum(negCounts[:i]) + n).item()
                    #print("negIx" + str(negIx))  #0到159
                    loss += criterion(seqQ[i:i+1], seqP[i:i+1], seqN[negIx:negIx+1]) #每一个训练数据对应一个正样本，10个负样本

            loss /= nNeg.float().to(device) # normalise by actual number of negatives
            loss.backward()
            optimizer.step()
            del input, seq_encoding, seqQ, seqP, seqN
            del query, positives, negatives

            batch_loss = loss.item()
            epoch_loss += batch_loss

            if iteration % 50 == 0 or nBatches <= 10:
                print("==> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, 
                    nBatches, batch_loss), flush=True)
                writer.add_scalar('Train/Loss', batch_loss, 
                        ((epoch-1) * nBatches) + iteration)
                writer.add_scalar('Train/nNeg', nNeg, 
                        ((epoch-1) * nBatches) + iteration)
                print('Allocated:', torch.cuda.memory_allocated())
                print('Cached:', torch.cuda.memory_cached())

        startIter += len(training_data_loader)
        del training_data_loader, loss
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        remove(train_set.cache) # delete HDF5 cache

    avg_loss = epoch_loss / nBatches

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, avg_loss), 
            flush=True)
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch)

