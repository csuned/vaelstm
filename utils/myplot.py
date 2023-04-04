import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import wandb

ROOT_DIR = '../'
RES_DIR = ROOT_DIR + 'results'

def plt_compare_series(series_list, title, xlabel=None, ylabel=None, label_list=None):
    for i in range(series_list[0].shape[1]):
        fig, ax = plt.subplots(1,1,figsize=(6,3))
        for idx, s in enumerate(series_list):
            if label_list==None:
                ax.plot(s[:,i,:,:].reshape((s.shape[0]*s.shape[2],)))
            else:
                ax.plot(s[:,i,:,:].reshape((s.shape[0]*s.shape[2],)), label=label_list[idx])
        if label_list!=None:
            ax.legend()
        wandb.log(
                   {
                    f"comparison_{title}_{i}": wandb.Image(fig), 
                   }, commit=False
            )
        plt.savefig(RES_DIR+f'/{title}_{i}.jpg')
        plt.close()
    return
    