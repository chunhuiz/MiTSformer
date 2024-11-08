from data_provider.data_factory import data_provider
from exp_basic import Exp_Basic
from tools import EarlyStopping, adjust_learning_rate,visual,cal_accuracy
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

class Exp_Long_Term_Forecasting(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecasting, self).__init__(args)

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
        :param return_flag: return mixed variables (M) or separate (S)
        :return: MiTS
        '''

        X_mix = torch.zeros_like(X).to(X.device)
        # print(X_mix.shape)
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
        task_dis_criterion = nn.CrossEntropyLoss()
        task_con_criterion = nn.MSELoss()
        return con_rec_criterion, dis_rec_criterion, type_criterion, task_dis_criterion,task_con_criterion

    def vali(self, vali_data, vali_loader, task_dis_criterion,task_con_criterion):
        total_loss = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device).transpose(-1, -2)
                batch_x_mark = batch_x_mark.float().to(self.device).transpose(-1, -2)

                batch_y = batch_y.float()[:, -self.args.pred_len:, :].to(self.device).transpose(-1, -2)

                batch_x_dis, batch_x_con = self.Norm_Discretization(batch_x, typeflag=self.args.typeflag,
                                                                    return_flag='S')
                batch_y_dis, batch_y_con = self.Norm_Discretization(batch_y, typeflag=self.args.typeflag,
                                                                    return_flag='S')




                x_dis_pe = None


                x_dis_forecast, x_con_forecast = self.model(batch_x_dis,batch_x_con,x_dis_pe=x_dis_pe,x_mark_enc = batch_x_mark)

                dis_fore_loss = task_dis_criterion(x_dis_forecast.reshape(-1, 2),
                                                   batch_y_dis.reshape(-1, 1).squeeze(-1).long()) / (self.args.dis_dim)
                con_fore_loss = task_con_criterion(x_con_forecast, batch_y_con) / (self.args.con_dim)



                loss = self.args.dis_rec_loss_w * dis_fore_loss + self.args.con_rec_loss_w * con_fore_loss

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

        con_rec_criterion, dis_rec_criterion, type_criterion, task_dis_criterion,task_con_criterion = self._select_criterion()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        train_loss_list = []
        con_recon_loss_list = []
        dis_recon_loss_list = []
        dis_recon_acc_list = []
        type_loss_list = []
        smooth_loss_list = []

        dis_forecast_loss_list = []
        con_forecast_loss_list = []
        dis_forecast_acc_list = []




        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss_l = []
            con_recon_loss_l = []
            dis_recon_loss_l = []
            dis_recon_acc_l = []
            type_loss_l = []
            smooth_loss_l = []
            dis_forecast_loss_l = []
            con_forecast_loss_l = []
            dis_forecast_acc_l = []


            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                #  B*T*D -> B*D*T !
                batch_x = batch_x.float().to(self.device).transpose(-1, -2)
                batch_x_mark = batch_x_mark.float().to(self.device).transpose(-1, -2)

                batch_y = batch_y.float()[:, -self.args.pred_len:, :].to(self.device).transpose(-1, -2)


                batch_x_dis, batch_x_con = self.Norm_Discretization(batch_x, typeflag=self.args.typeflag,return_flag='S')
                batch_y_dis, batch_y_con = self.Norm_Discretization(batch_y, typeflag=self.args.typeflag,
                                                                    return_flag='S')
                batch_x_dis_vari = (batch_x_dis[:, :, 1:] - batch_x_dis[:, :, :-1]).detach()


                x_dis_pe = None


                x_lcv, x_lcv_embed, x_con_embed, means_con, stdev_con = self.model.DATA_EMBED(batch_x_dis, x_dis_pe,
                                                                                              batch_x_con,x_mark_enc=batch_x_mark)


                pred_domain, true_domain = self.model.variable_discriminate(x_lcv_embed, x_con_embed)

                x_lcv_embed, x_con_embed, dis_att, con_att, dis_con_attn, con_dis_attn = self.model.Encoding(
                    x_lcv_embed,
                    x_con_embed)

                x_dis_rec, x_con_rec = self.model.SSL_RECON(x_lcv_embed, x_con_embed, means_con, stdev_con)

                x_dis_forecast, x_con_forecast = self.model.Forecast(x_lcv_embed,x_con_embed,means_con,stdev_con)



                type_loss = type_criterion(pred_domain, true_domain.long()) / (self.args.p)
                dis_rec_loss = dis_rec_criterion(x_dis_rec.reshape(-1, 2),
                                                 batch_x_dis.reshape(-1, 1).squeeze(-1).long()) / (self.args.dis_dim)
                dis_rec_acc = batch_x_dis.clone().eq(
                    x_dis_rec.clone().argmax(dim=-1)).float().mean().detach().cpu().numpy() * 100
                con_rec_loss = con_rec_criterion(x_con_rec, batch_x_con) / (self.args.con_dim)



                smooth_mat = torch.einsum('xl,bln->xbn', smooth_arr.to(batch_x.device),
                                          x_lcv.transpose(-1, -2)).transpose(0, 1).transpose(1, 2)
                smooth_loss = (smooth_mat * smooth_mat * abs(batch_x_dis_vari)).mean()

                # Forecasting Loss


                dis_fore_loss = task_dis_criterion(x_dis_forecast.reshape(-1, 2),
                                                 batch_y_dis.reshape(-1, 1).squeeze(-1).long()) / (self.args.dis_dim)
                con_fore_loss = task_con_criterion(x_con_forecast, batch_y_con) / (self.args.con_dim)


                dis_fore_acc = batch_y_dis.clone().eq(
                    x_dis_forecast.clone().argmax(dim=-1)).float().mean().detach().cpu().numpy() * 100


                loss =  self.args.dis_rec_loss_w * dis_fore_loss+  self.args.con_rec_loss_w * con_fore_loss \
                       + (self.args.type_loss_w * type_loss + self.args.dis_rec_loss_w * dis_rec_loss \
                       + self.args.con_rec_loss_w * con_rec_loss + self.args.smooth_loss_w * smooth_loss)/5


                train_loss_l.append(loss.item())
                con_recon_loss_l.append(con_rec_loss.item())
                dis_recon_loss_l.append(dis_rec_loss.item())
                dis_recon_acc_l.append(dis_rec_acc)
                type_loss_l.append(type_loss.item())
                smooth_loss_l.append(smooth_loss.item())
                dis_forecast_loss_l.append(dis_fore_loss.item())
                con_forecast_loss_l.append(con_fore_loss.item())
                dis_forecast_acc_l.append(dis_fore_acc)


                if (i + 1) % 10 == 0:
                    # print("\titers: {0}, epoch: {1} | loss_all: {2:.7f} |dis_rec_loss: {2:.7f} |con_rec_loss: {2:.7f} ".format(i + 1,epoch + 1, loss.item(),dis_rec_loss.item(),con_rec_loss.item()))
                    print(
                        "iters: {0}, epoch: {1} | loss_all: {2:.7f} ".format(
                            i + 1, epoch + 1, loss.item()))
                    print('Con_reconstruction_loss = %f' % con_rec_loss.item())
                    print('Dis_reconstruction_loss = %f' % dis_rec_loss.item())
                    print('Dis_reconstruction_acc = %f' % dis_rec_acc)
                    print('Smooth_loss = %f' % smooth_loss.item())
                    print('Con_forecasting_loss = %f' % con_fore_loss.item())
                    print('Dis_forecasting_loss = %f' % dis_fore_loss.item())
                    print('Dis_forecasting_acc = %f' % dis_fore_acc)




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
            dis_forecast_loss_l = np.average(dis_forecast_loss_l)
            con_forecast_loss_l = np.average(con_forecast_loss_l)
            dis_forecast_acc_l = np.average( dis_forecast_acc_l)


            vali_loss = self.vali(vali_data, vali_loader, task_dis_criterion,task_con_criterion)
            test_loss = self.vali(test_data, test_loader, task_dis_criterion,task_con_criterion)



            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.5f} Vali Loss: {3:.5f} Test Loss: {4:.5f} "
                    .format(epoch + 1, train_steps, train_loss_l, vali_loss,test_loss))

            early_stopping(vali_loss, self.model, path)

            # early_stopping(999, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            train_loss_list.append(train_loss_l)
            con_recon_loss_list.append(con_recon_loss_l)
            dis_recon_loss_list.append(dis_recon_loss_l)
            dis_recon_acc_list.append(dis_recon_acc_l)
            type_loss_list.append(type_loss_l)
            smooth_loss_list.append(smooth_loss_l)
            dis_forecast_loss_list.append(dis_forecast_loss_l)
            con_forecast_loss_list.append(con_forecast_loss_l)
            dis_forecast_acc_list.append(dis_forecast_acc_l)



            if self.args.lradj_flag:

                    #  learning rate update
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model, (
            train_loss_list, con_recon_loss_list, dis_recon_loss_list, dis_recon_acc_list, type_loss_list,
            smooth_loss_list, dis_forecast_loss_list,con_forecast_loss_list,dis_forecast_acc_list)



    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds_dis = []
        trues_dis = []
        preds_con = []
        trues_con = []


        if self.args.forecast_visual_flag:
            folder_path = './test_results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device).transpose(-1, -2)
                batch_x_mark = batch_x_mark.float().to(self.device).transpose(-1, -2)

                batch_y = batch_y.float()[:, -self.args.pred_len:, :].to(self.device).transpose(-1, -2)

                batch_x_dis, batch_x_con = self.Norm_Discretization(batch_x, typeflag=self.args.typeflag,
                                                                    return_flag='S')
                batch_y_dis, batch_y_con = self.Norm_Discretization(batch_y, typeflag=self.args.typeflag,
                                                                    return_flag='S')


                x_dis_pe = None


                x_dis_forecast, x_con_forecast = self.model(batch_x_dis, batch_x_con, x_dis_pe=x_dis_pe,
                                                            x_mark_enc=batch_x_mark)

                pred_con = x_con_forecast.detach().cpu().numpy()
                true_con = batch_y_con.detach().cpu().numpy()
                preds_con.append(pred_con)
                trues_con.append(true_con)

                if i % 20 == 0 and self.args.forecast_visual_flag:
                    input = batch_x_con.detach().cpu().numpy()
                    gt = np.concatenate((input[0,  -3,:], true_con[0,  -3,:]), axis=0)
                    pd = np.concatenate((input[0,  -3,:], pred_con[0,  -3,:]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))


                pred_dis = x_dis_forecast.detach().cpu()
                true_dis = batch_y_dis.detach().cpu()
                preds_dis.append(pred_dis)
                trues_dis.append(true_dis)


        preds_con = np.array(preds_con)
        trues_con = np.array(trues_con)
        preds_con = preds_con.reshape(-1, preds_con.shape[-2], preds_con.shape[-1])
        trues_con = trues_con.reshape(-1, trues_con.shape[-2], trues_con.shape[-1])
        print('test con shape:', preds_con.shape, trues_con.shape)

        mae, mse, rmse, mape, mspe = metric(preds_con, trues_con)
        print('mse:{}, mae:{}'.format(mse, mae))


        preds_dis = torch.cat(preds_dis, 0)
        trues_dis = torch.cat(trues_dis, 0)
        probs = torch.nn.functional.softmax(preds_dis.reshape(-1,2))  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues_dis = trues_dis.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues_dis)


        return mae, mse, accuracy

