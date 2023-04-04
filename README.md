# vaelstm
This is a Torch version of "https://github.com/lin-shuyu/VAE-LSTM-for-anomaly-detection"

*WANDB IS READY*

Before you run:
1. Make sure you have wandb installed
2. Please login to wandb, with 'wandb login'

By default, you will see the results on 'MyVAELSTM' project.

*HOW TO RUN*

cd scripts/
python3 run_vae_lstm.py

*THIS IS A MULTIVARIATE VERSION*

This version is based on random input with *two attributes*, the model is converging at training set.

Please fine tune the parameter with your dataset, as well as try different VAEs (KLD is used by default)

Ping me when you spot something :)

--
chuanhao.sun@ed.ac.uk