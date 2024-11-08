import argparse
import os
import torch
from exp_imputation import Exp_Imputation
import random
from utils.print_args import print_args
import time
import numpy as np

if __name__ == '__main__':
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)


    parser = argparse.ArgumentParser(description='MiTSformer')

    # basic config
    parser.add_argument('--task_name', type=str, default='imputation',
                        help='task name, options:[long_term_forecast, extrinsic_regression, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='MITS',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, default='ETT_small', help='dataset type')
    parser.add_argument('--data_name', type=str, default='ETTh2', help='dataname')
    parser.add_argument('--T', type=int, default=96, help='seq_len')
    parser.add_argument('--seq_len', type=int, default=96, help='seq_len')
    parser.add_argument('--label_len', type=int, default=96, help='label_len')
    parser.add_argument('--pred_len', type=int, default=96, help='pred_len')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--dis_proportion',type=float,default=0.5,help = 'discrete_proportion')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # model define


    # imputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')
    parser.add_argument('--imputation_visual_flag', type=bool, default=False, help='visual plot')



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
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
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
    parser.add_argument('--gpu', type=int, default=3, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')


    #  Tune
    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    # first obtain the data statistics and typeflag and fix it!



    p = 7
    dis_dim = 3
    if args.data_name =='Electricity':
        p=321
        dis_dim=160
        data_path = 'electricity.csv'
        data = 'custom'
        root_path = './dataset/imputation/electricity'

    if args.data_name =='Weather':
        p=21
        dis_dim=10
        data_path = 'weather.csv'
        data = 'custom'
        root_path = './dataset/imputation/weather'

    if args.data_name =='ETTh1':
        data_path = 'ETTh1.csv'
        data = 'ETTh1'
        root_path = './dataset/imputation/ETT-small'

    if args.data_name =='ETTh2':
        data_path = 'ETTh2.csv'
        data = 'ETTh2'
        root_path = './dataset/imputation/ETT-small'



    if args.data_name =='ETTm1':
        data_path = 'ETTm1.csv'
        data = 'ETTm1'
        root_path = './dataset/imputation/ETT-small'

    if args.data_name =='ETTm2':
        data_path = 'ETTm2.csv'
        data = 'ETTm2'
        root_path = './dataset/imputation/ETT-small'





    typeflag = []
    for i in range(dis_dim):
        typeflag.append(0)
    for i in range(dis_dim, p):
        typeflag.append(1)

    #  if randomly sampled discrete variables
    random.shuffle(typeflag)

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]




    args.typeflag = typeflag

    args.root_path = root_path
    args.data_path = data_path
    args.data = data
    args.dis_dim = dis_dim
    args.p = p
    args.con_dim = p - dis_dim

    print('Args in experiment:')
    print_args(args)

    Exp = Exp_Imputation

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments

            setting = '{}_{}_{}_sl{}_mr{}_dm{}_ln{}_lradj{}_lr{}_sm_loss_w{}'.format(
                args.task_name,
                args.model,
                args.data_name,
                args.seq_len,
                args.mask_rate,
                args.d_model,
                args.block_layers,
                args.lradj_flag,
                args.learning_rate,
                args.smooth_loss_w)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            print('Type Flag:')
            print(args.typeflag)
            model,LOSS = exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            mae, mse, rmse, mape, mspe = exp.test(setting)

            with torch.cuda.device('cuda:' + str(args.gpu)):
                torch.cuda.empty_cache()

    else:
        ii = 0
        setting = '{}_{}_{}_sl{}_mr{}_dm{}_ln{}_lradj{}_lr{}_sm_loss_w{}'.format(
            args.task_name,
            args.model,
            args.data_name,
            args.seq_len,
            args.mask_rate,
            args.d_model,
            args.block_layers,
            args.lradj_flag,
            args.learning_rate,
            args.smooth_loss_w)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        ae, mse, rmse, mape, mspe = exp.test(setting, test=1)

        with torch.cuda.device('cuda:' + str(args.gpu)):
            torch.cuda.empty_cache()



    res = {

        't_now': time.asctime(),
        'data': args.data_name,
        # 'K': args.num_classes,
        'p': args.p,
        'mask_rate':args.mask_rate,
        'typeflag':args.typeflag,
        'd_model':args.d_model,
        'block_layers':args.block_layers,
        'lr_adj':args.lradj_flag,
        'learning_rate': args.learning_rate,
        'smooth_loss_w':args.smooth_loss_w,
        'MAE': mae,
        'MSE':mse
    }

    expd_path = './Results_imputation_new/' + args.data_name + '/'+ str(args.mask_rate*1000) + '/' + args.model + '/'

    if not os.path.exists(expd_path):
        os.makedirs(expd_path)

    expd_pathh = expd_path + '/result_ours.txt'
    with open(expd_pathh, 'a') as f:
        f.write('\n')
        for k, v in res.items():
            f.write(k + ':' + str(v) + '   ')
        f.write('\n')

    torch.cuda.empty_cache()