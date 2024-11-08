from data_provider.data_factory import data_provider
from exp_basic import Exp_Basic
from tools import EarlyStopping, adjust_learning_rate,visual
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import time
import torch.multiprocessing
from utils.metrics import metric
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')

torch.multiprocessing.set_sharing_strategy('file_system')

class Exp_Imputation(Exp_Basic):
    def __init__(self, args):
        super(Exp_Imputation, self).__init__(args)

    def _build_model(self):



        # Model ini
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def Norm_Discretization(self, X, typeflag, return_flag='S'):
        '''

        :param X: input tensor B*D*T
        :param typeflag: indicate discrete (0)  or continuous (1)
        :param return_flag: return mixed variables (M) or seperate (S)
        :return: MiTS
        '''

        X_mix = torch.zeros_like(X).to(X.device)

        for i in range(X_mix.shape[0]):
            for j in range(X_mix.shape[-2]):
                if typeflag[j] == 0:
                    instance = X[i, j, :]
                    # ensure numerical stability
                    instance_norm = (instance - torch.min(instance)) / (
                                torch.max(instance) - torch.min(instance) + 1e-5)
                    X_mix[i, j, :] = instance_norm
                    X_mix[i, j, :] = X_mix[i, j, :] > 0.5
                else:
                    X_mix[i, j, :] = X[i, j, :]

        dis_ind = []
        con_ind = []
        for i in range(len(typeflag)):
            if typeflag[i] == 0:

                dis_ind.append(i)
            else:
                con_ind.append(i)

        X_dis = X_mix[:, dis_ind, :]
        X_con = X_mix[:, con_ind, :]

        if return_flag == 'M':
            return X_mix, X_mix
        if return_flag == 'S':
            return X_dis, X_con

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        con_rec_criterion = nn.MSELoss()
        dis_rec_criterion = nn.CrossEntropyLoss()
        type_criterion = nn.CrossEntropyLoss()
        task_criterion = nn.MSELoss()
        return con_rec_criterion, dis_rec_criterion, type_criterion, task_criterion

    def vali(self, vali_data, vali_loader, task_criterion):
        total_loss = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):

                batch_x = batch_x.float().to(self.device).transpose(-1, -2)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # random mask only for continuous variables
                B = batch_x.shape[0]
                mask = torch.rand((B, self.args.con_dim, self.args.T)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained


                batch_x_dis, batch_x_con_orig = self.Norm_Discretization(batch_x, typeflag=self.args.typeflag,
                                                                         return_flag='S')

                batch_x_con = batch_x_con_orig.masked_fill(mask == 0, 0)


                x_dis_pe = None

                x_con_imput_result = self.model(batch_x_dis, batch_x_con,x_dis_pe, mask=mask)



                loss = task_criterion(x_con_imput_result[mask == 0], batch_x_con_orig[mask == 0])


                total_loss.append(loss.detach().cpu().numpy())

        total_loss = np.average(total_loss)

        self.model.train()

        return total_loss

    def train(self, setting):



        smooth_arr = torch.zeros((self.args.T - 1, self.args.T))
        for i in range(self.args.T - 1):
            smooth_arr[i, i] = -1
            smooth_arr[i, i + 1] = 1

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        con_rec_criterion, dis_rec_criterion, type_criterion, task_criterion = self._select_criterion()

        train_loss_list = []
        con_recon_loss_list = []
        dis_recon_loss_list = []
        dis_recon_acc_list = []
        type_loss_list = []
        smooth_loss_list = []

        task_loss_list = []




        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss_l = []
            con_recon_loss_l = []
            dis_recon_loss_l = []
            dis_recon_acc_l = []
            type_loss_l = []
            smooth_loss_l = []

            task_loss_l = []


            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                #  B*T*D -> B*D*T !
                batch_x = batch_x.float().to(self.device).transpose(-1, -2)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # random mask only for continuous variables
                B = batch_x.shape[0]

                mask = torch.rand((B, self.args.con_dim, self.args.T)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                # inp = batch_x.masked_fill(mask == 0, 0)

                batch_x_dis, batch_x_con_orig = self.Norm_Discretization(batch_x, typeflag=self.args.typeflag,return_flag='S')
                batch_x_dis_vari = (batch_x_dis[:, :, 1:] - batch_x_dis[:, :, :-1]).detach()


                batch_x_con = batch_x_con_orig.masked_fill(mask == 0, 0)


                x_dis_pe = None

                x_lcv, x_lcv_embed, x_con_embed, means_con, stdev_con = self.model.DATA_EMBED(batch_x_dis, x_dis_pe,
                                                                                              batch_x_con,mask=mask)


                pred_domain, true_domain = self.model.variable_discriminate(x_lcv_embed, x_con_embed)

                x_lcv_embed, x_con_embed, dis_att, con_att, dis_con_attn, con_dis_attn = self.model.Encoding(
                    x_lcv_embed,
                    x_con_embed)
                x_dis_rec, x_con_rec = self.model.SSL_RECON(x_lcv_embed, x_con_embed, means_con, stdev_con)

                x_con_imput_result = self.model.Imputation(x_con_embed, means_con, stdev_con)



                type_loss = type_criterion(pred_domain, true_domain.long()) / (self.args.p)
                dis_rec_loss = dis_rec_criterion(x_dis_rec.reshape(-1, 2),
                                                 batch_x_dis.reshape(-1, 1).squeeze(-1).long()) / (self.args.dis_dim)

                #  reconstruction only for unmasked ones
                con_rec_loss = con_rec_criterion(x_con_rec[mask == 1], batch_x_con_orig[mask == 1]) / (self.args.con_dim)



                #  B*D*(T-1)
                smooth_mat = torch.einsum('xl,bln->xbn', smooth_arr.to(batch_x.device),
                                          x_lcv.transpose(-1, -2)).transpose(0, 1).transpose(1, 2)
                smooth_loss = (smooth_mat * smooth_mat * abs(batch_x_dis_vari)).mean()

                task_loss = task_criterion(x_con_imput_result[mask == 0],batch_x_con_orig[mask == 0] )


                dis_rec_acc = batch_x_dis.clone().eq(
                    x_dis_rec.clone().argmax(dim=-1)).float().mean().detach().cpu().numpy() * 100

                loss = task_loss + (self.args.type_loss_w * type_loss + self.args.dis_rec_loss_w * dis_rec_loss \
                       + self.args.con_rec_loss_w * con_rec_loss + self.args.smooth_loss_w * smooth_loss)/4


                train_loss_l.append(loss.item())
                con_recon_loss_l.append(con_rec_loss.item())
                dis_recon_loss_l.append(dis_rec_loss.item())
                dis_recon_acc_l.append(dis_rec_acc)
                type_loss_l.append(type_loss.item())
                smooth_loss_l.append(smooth_loss.item())
                task_loss_l.append(task_loss.item())


                if (i + 1) % 2 == 0:
                    # print("\titers: {0}, epoch: {1} | loss_all: {2:.7f} |dis_rec_loss: {2:.7f} |con_rec_loss: {2:.7f} ".format(i + 1,epoch + 1, loss.item(),dis_rec_loss.item(),con_rec_loss.item()))
                    print(
                        "iters: {0}, epoch: {1} | loss_all: {2:.7f} ".format(
                            i + 1, epoch + 1, loss.item()))
                    print('Con_reconstruction_loss = %f' % con_rec_loss.item())
                    print('Dis_reconstruction_loss = %f' % dis_rec_loss.item())
                    print('Dis_reconstruction_acc = %f' % dis_rec_acc)
                    print('Smooth_loss = %f' % smooth_loss.item())
                    print('Imputation_loss = %f' % task_loss.item())
                    # print('task_loss = %f' % task_loss.item())


                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss_l = np.average(train_loss_l)
            con_recon_loss_l = np.average(con_recon_loss_l)
            dis_recon_loss_l = np.average(dis_recon_loss_l)
            dis_recon_acc_l = np.average(dis_recon_acc_l)
            type_loss_l = np.average(type_loss_l)
            smooth_loss_l = np.average(smooth_loss_l)
            task_loss_l = np.average(task_loss_l)



            vali_loss = self.vali(vali_data, vali_loader, task_criterion)
            test_loss = self.vali(test_data, test_loader, task_criterion)



            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.5f} Vali Loss: {3:.5f} Test Loss: {4:.5f} "
                    .format(epoch + 1, train_steps, train_loss_l, vali_loss,test_loss))

            early_stopping(vali_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            train_loss_list.append(train_loss_l)
            con_recon_loss_list.append(con_recon_loss_l)
            dis_recon_loss_list.append(dis_recon_loss_l)
            dis_recon_acc_list.append(dis_recon_acc_l)
            type_loss_list.append(type_loss_l)
            smooth_loss_list.append(smooth_loss_l)
            task_loss_list.append(task_loss_l)


            if self.args.lradj_flag:

                    #  learning rate update
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model, (
            train_loss_list, con_recon_loss_list, dis_recon_loss_list, dis_recon_acc_list, type_loss_list,
            smooth_loss_list, task_loss_list)



    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        masks = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device).transpose(-1, -2)
                batch_x_mark = batch_x_mark.float().to(self.device)


                B = batch_x.shape[0]
                mask = torch.rand((B, self.args.con_dim, self.args.T)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                # inp = batch_x.masked_fill(mask == 0, 0)


                batch_x_dis, batch_x_con_orig = self.Norm_Discretization(batch_x, typeflag=self.args.typeflag,
                                                                         return_flag='S')

                batch_x_con = batch_x_con_orig.masked_fill(mask == 0, 0)


                x_dis_pe = None

                x_con_imput_result = self.model(batch_x_dis, batch_x_con, x_dis_pe, mask=mask)
                # eval

                #  B*D*T -> B*T*D
                outputs = x_con_imput_result.transpose(-1,-2)
                outputs = outputs.detach().cpu().numpy()
                pred = outputs
                true = batch_x_con_orig.transpose(-1,-2).detach().cpu().numpy()
                mask = mask.transpose(-1,-2)

                preds.append(pred)
                trues.append(true)
                masks.append(mask.detach().cpu())

                if i % 20 == 0:
                    filled = true[0, :, -1].copy()
                    filled = filled * mask[0, :, -1].detach().cpu().numpy() + \
                             pred[0, :, -1] * (1 - mask[0, :, -1].detach().cpu().numpy())

                    if self.args.imputation_visual_flag:
                        visual(true[0, :, -1], filled, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        masks = np.concatenate(masks, 0)

        print('test shape:', preds.shape, trues.shape)



        mae, mse, rmse, mape, mspe = metric(preds[masks == 0], trues[masks == 0])
        print('mse:{}, mae:{}'.format(mse, mae))


        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return mae, mse, rmse, mape, mspe
