import argparse
import os
import torch
from exp_anomaly_detection import Exp_Anomaly_Detection
import random
from utils.print_args import print_args
from test_tube import HyperOptArgumentParser
import time
import numpy as np

if __name__ == '__main__':
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='MiTSformer')

    
    # basic config
    parser.add_argument('--task_name', type=str, default='anomaly_detection',
                        help='task name, options:[long_term_forecast, extrinsic_regression, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='MITS',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')
    
    # data loader
    parser.add_argument('--data', type=str, default='SMD', help='dataset type')
    parser.add_argument('--data_name', type=str, default='SMD', help='dataname')
    parser.add_argument('--T', type=int, default=100, help='seq_len')
    parser.add_argument('--seq_len', type=int, default=100, help='seq_len')
    # parser.add_argument('--root_path', type=str, default='./dataset/classification/SelfRegulationSCP1', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='./dataset/extrinsic_regression/AppliancesEnergy', help='data file')
    parser.add_argument('--dis_proportion',type=float,default=0.5,help = 'discrete_proportion')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    
    # model define
    
    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')
    
    parser.add_argument('--anomaly_metric', type=str, default='all',
                        help='anomaly_metric task, options:[all,con_only]')
    
    
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--block_layers', type=int, default=1, help='num of encoder layers')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--dis_embed', type=bool, default=False)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--embed', type=str, default='timeF',
                            help='time features encoding, options:[timeF, fixed, learned]')
    
    
    # optimization
    parser.add_argument('--lradj_flag', type=bool, default=True, help='adjust learning rate')
    parser.add_argument('--learning_rate', type=float,default=0.001, help='optimizer learning rate')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    
    
    parser.add_argument('--smooth_loss_w', type=float, default=0.5)
    parser.add_argument('--type_loss_w', type=float, default=1.0)
    parser.add_argument('--dis_rec_loss_w', type=float, default=1.0)
    parser.add_argument('--con_rec_loss_w', type=float, default=1.0)
    
    
    
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    
    
    #  Tune
    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    
    # first obtain the data statistics and typeflag and fix it!
    
    
    if args.data_name =='SMD':
        p=38
        dis_dim=19
    if args.data_name =='MSL':
        p=55
        dis_dim=27
    if args.data_name =='SMAP':
        p=25
        dis_dim=12
    if args.data_name =='SWAT':
        p=51
        dis_dim=25
    
    if args.data_name =='PSM':
        p=25
        dis_dim=12
    
    
    
    
    typeflag = []
    for i in range(dis_dim):
        typeflag.append(0)
    for i in range(dis_dim, p):
        typeflag.append(1)

    #  if randomly sampled discrete variables:
    random.shuffle(typeflag)

    
    
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    args.typeflag = typeflag
    args.p = p
    args.dis_dim = dis_dim
    args.con_dim = p-dis_dim

    args.root_path = './dataset/anomaly_detection/' + args.data_name

    print('Args in experiment:')
    print_args(args)

    Exp = Exp_Anomaly_Detection

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_sl{}_dm{}_ln{}_lradj{}_lr{}_sm_loss_w{}_norratio{}'.format(
                args.task_name,
                args.model,
                args.data_name,
                args.seq_len,
                args.d_model,
                args.block_layers,
                args.lradj_flag,
                args.learning_rate,
                args.smooth_loss_w,
                args.anomaly_ratio)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            print('Type Flag:')
            print(args.typeflag)
            model, LOSS = exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            accuracy, precision, recall, f_score = exp.test(setting)
            with torch.cuda.device('cuda:' + str(args.gpu)):
                torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_sl{}_dm{}_ln{}_lradj{}_lr{}_sm_loss_w{}_norratio{}'.format(
            args.task_name,
            args.model,
            args.data_name,
            args.seq_len,
            args.d_model,
            args.block_layers,
            args.lradj_flag,
            args.learning_rate,
            args.smooth_loss_w,
            args.anomaly_ratio)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        accuracy, precision, recall, f_score = exp.test(setting, test=1)
        with torch.cuda.device('cuda:' + str(args.gpu)):
            torch.cuda.empty_cache()

    res = {
        
        't_now': time.asctime(),
        'data': args.data_name,
        'p': args.p,
        'typeflag': args.typeflag,
        'd_model': args.d_model,
        'block_layers': args.block_layers,
        'anomaly_ratio':args.anomaly_ratio,
        'lr_adj': args.lradj_flag,
        'learning_rate': args.learning_rate,
        'smooth_loss_w': args.smooth_loss_w,
        'Accuracy': accuracy,
        'Precision':precision,
        'Recall':recall,
        'F1-score': f_score
    }

    expd_path = './Results_anomaly_detection_new/' + args.data_name + '/' + args.model + '/'

    if not os.path.exists(expd_path):
        os.makedirs(expd_path)

    expd_pathh = expd_path + '/result_ours.txt'
    with open(expd_pathh, 'a') as f:
        f.write('\n')
        for k, v in res.items():
            f.write(k + ':' + str(v) + '   ')
        f.write('\n')

    
    torch.cuda.empty_cache()
