import os
import torch
import model
import numpy as np

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'MITS': model
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def Norm_Discretization(self,X,typeflag,return_flag='S'):
        '''

        :param X: input tensor B*T*D
        :param typeflag: indicate discrete (0)  or continuous (1)
        :param return_flag: return mixed variables (M) or separate (S)
        :return: MiTS
        '''

        X_mix = torch.zeros_like(X).to(X.device)
        for i in range(X_mix.shape[0]):
            for j in range(X_mix.shape[-1]):
                if typeflag[j] == 0:
                    instance = X[i, :, j]
                    instance_norm = (instance - torch.min(instance)) / (torch.max(instance) - torch.min(instance))
                    X_mix[i, :, j] = instance_norm
                    X_mix[i, :, j] = X_mix[i, :, j] > 0.5
                else:
                    X_mix[i, :, j]=X[i, :, j]


        dis_ind = []
        con_ind = []
        for i in range(len(typeflag)):
            if typeflag[i] == 0:

                dis_ind.append(i)
            else:
                con_ind.append(i)

        X_dis = X_mix[:, :, dis_ind]
        X_con = X_mix[:, :, con_ind]

        if return_flag == 'M':
            return X_mix, X_mix
        if return_flag =='S':
            return X_dis, X_con


    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
