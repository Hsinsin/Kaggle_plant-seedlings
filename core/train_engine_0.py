import os
import time
import torch
import shutil
import logging
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')

from tools.meter import AverageValueMeter

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth'):
    filepath = '_'.join((str(state['epoch']),filename))
    filepath = os.path.join(save_dir, filepath)
    make_dir(save_dir)
    torch.save(state, filepath)
    if is_best:
        shutil.copy(filepath, os.path.join(save_dir, 'model_best.pth'))

class Train_Engine(object):
    def __init__(self, net):
        self.net = net
        self.loss = AverageValueMeter()
        self.acc = AverageValueMeter()

    def fit(self, train_data, test_data, optimizer, criterion, lr_scheduler, epochs=100, print_interval=100,eval_step=1, save_step=10, save_dir='checkpoint'):
        best_test_acc = 0.0
        losses = []
        accs = []
        nums = []

        with open('acc.txt','w') as f:
            for epoch in range(0, epochs):
                self.loss.reset()
                self.acc.reset()
                self.net.train()

                lr = lr_scheduler.update(epoch)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                logging.info('Epoch: %d learning rate update to %.3e' %(epoch,lr))

                tic = time.time()
                btic = time.time()
                for i, data in enumerate(train_data):
                    imgs, labels = data
                    labels = labels.cuda()
                    scores = self.net(imgs)
                    loss = criterion(scores, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    self.loss.add(loss.item())
                    acc = (scores.max(1)[1] == labels.long()).float().mean()
                    self.acc.add(acc.item())

                    if print_interval and (i+1)%print_interval == 0:
                        loss_mean = self.loss.value()[0]
                        acc_mean = self.acc.value()[0]
                        logging.info('Epoch: %d \tBatch: %d \tSpeed: %f samples/sec \tloss = %f \t'
                                     'acc = %f' % (epoch, i + 1, train_data.batch_size * print_interval / (time.time() - btic),
                                                   loss_mean, acc_mean))
                                                
                        btic =time.time()
                
                loss_mean = self.loss.value()[0]
                acc_mean = self.acc.value()[0]
                losses.append(loss_mean)
                accs.append(acc_mean)
                nums.append(epoch)
                throughput = int(train_data.batch_size * len(train_data) / (time.time() - tic))

                logging.info('Epoch: %d \ttraining: loss = %f \tacc = %f' % (epoch, loss_mean, acc_mean))
                logging.info('Epoch: %d \tspeed: %d \tsamples/sec\ttime cost: %f' % (epoch, throughput, time.time() - tic))
                f.write('Epoch: %d \ttraining: loss = %f \tacc = %f' % (epoch, loss_mean, acc_mean))
                f.write('\n')
                f.flush()

                is_best = False
                if test_data is not None and eval_step and (epoch+1) % eval_step == 0:
                    test_acc = self.test_func(test_data)
                    is_best = test_acc > best_test_acc
                    if is_best:
                        best_test_acc = test_acc
                
                state_dict = self.net.module.state_dict()
                if not (epoch+1) % save_step:
                    save_checkpoint({
                        'state_dict': state_dict,
                        'epoch': epoch + 1
                        },is_best = is_best, save_dir=save_dir)
            
            print('---Training Finished---\n')
            plt.figure(figsize = (10, 8),dpi = 80)
            plt.subplot(121)
            plt.plot(nums, losses)
            plt.xlabel("num")
            plt.ylabel("loss") 
            plt.title('loss')
            plt.subplot(122)
            plt.plot(nums, accs)
            plt.xlabel("num")
            plt.ylabel("accuracy") 
            plt.title('accuracy')
            plt.savefig('results.png')
    
    def test_func(self, test_data):
        num_correct = 0
        num_imgs = 0
        self.net.eval()
        for data in test_data:
            imgs, labels = data
            labels = labels.cuda()
            scores = self.net(imgs)
            num_correct += (scores.max(1)[1] == labels).float().sum().item()
            num_imgs += imgs.shape[0]
        
        return num_correct / num_imgs