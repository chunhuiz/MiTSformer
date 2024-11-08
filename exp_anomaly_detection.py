from data_provider.data_factory import data_provider
from exp_basic import Exp_Basic
from tools import EarlyStopping, adjust_learning_rate,adjustment
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import time
import torch.multiprocessing
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')

torch.multiprocessing.set_sharing_strategy('file_system')

class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)

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

    def vali(self, vali_data, vali_loader, con_rec_criterion,dis_rec_criterion):
        total_loss = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):


                #  B*T*D -> B*D*T
                batch_x = batch_x.float().to(self.device).transpose(-1, -2)

                batch_x_dis, batch_x_con = self.Norm_Discretization(batch_x, typeflag=self.args.typeflag,
                                                                    return_flag='S')



                x_dis_pe = None


                x_lcv, x_lcv_embed, x_con_embed,means_con,stdev_con = self.model.DATA_EMBED(batch_x_dis, x_dis_pe, batch_x_con)



                x_lcv_embed, x_con_embed, dis_att, con_att, dis_con_attn, con_dis_attn = self.model.Encoding(
                    x_lcv_embed,
                    x_con_embed)


                x_dis_rec, x_con_rec = self.model.SSL_RECON(x_lcv_embed, x_con_embed,means_con,stdev_con)

                dis_rec_loss = dis_rec_criterion(x_dis_rec.reshape(-1, 2), batch_x_dis.reshape(-1, 1).squeeze(-1).long()) /(self.args.dis_dim)

                con_rec_loss = con_rec_criterion(x_con_rec, batch_x_con.detach()) / (self.args.con_dim)


                loss = self.args.dis_rec_loss_w * dis_rec_loss + self.args.con_rec_loss_w * con_rec_loss



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
        con_rec_criterion, dis_rec_criterion, type_criterion, _ = self._select_criterion()

        train_loss_list = []
        con_recon_loss_list = []
        dis_recon_loss_list = []
        dis_recon_acc_list = []
        type_loss_list = []
        smooth_loss_list = []


        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss_l = []
            con_recon_loss_l = []
            dis_recon_loss_l = []
            dis_recon_acc_l = []
            type_loss_l = []
            smooth_loss_l = []

            # cls_acc_l = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, label) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                #  B*T*D -> B*D*T !
                batch_x = batch_x.float().to(self.device).transpose(-1, -2)
                label = label.to(self.device)

                batch_x_dis, batch_x_con = self.Norm_Discretization(batch_x, typeflag=self.args.typeflag,return_flag='S')
                batch_x_dis_vari = (batch_x_dis[:, :, 1:] - batch_x_dis[:, :, :-1]).detach()


                x_dis_pe = None

                x_lcv, x_lcv_embed, x_con_embed, means_con, stdev_con = self.model.DATA_EMBED(batch_x_dis, x_dis_pe,
                                                                                              batch_x_con)
                pred_domain, true_domain = self.model.variable_discriminate(x_lcv_embed, x_con_embed)
                x_lcv_embed, x_con_embed, dis_att, con_att, dis_con_attn, con_dis_attn = self.model.Encoding(
                    x_lcv_embed,
                    x_con_embed)
                x_dis_rec, x_con_rec = self.model.SSL_RECON(x_lcv_embed, x_con_embed, means_con, stdev_con)


                type_loss = type_criterion(pred_domain, true_domain.long()) / (self.args.p)
                dis_rec_loss = dis_rec_criterion(x_dis_rec.reshape(-1, 2),
                                                 batch_x_dis.reshape(-1, 1).squeeze(-1).long()) / (self.args.dis_dim)
                con_rec_loss = con_rec_criterion(x_con_rec, batch_x_con) / (self.args.con_dim)


                smooth_mat = torch.einsum('xl,bln->xbn', smooth_arr.to(batch_x.device),
                                          x_lcv.transpose(-1, -2)).transpose(0, 1).transpose(1, 2)
                smooth_loss = (smooth_mat * smooth_mat * abs(batch_x_dis_vari)).mean()



                dis_rec_acc = batch_x_dis.clone().eq(
                    x_dis_rec.clone().argmax(dim=-1)).float().mean().detach().cpu().numpy() * 100

                loss =  self.args.type_loss_w * type_loss + self.args.dis_rec_loss_w * dis_rec_loss \
                       + self.args.con_rec_loss_w * con_rec_loss + self.args.smooth_loss_w * smooth_loss


                train_loss_l.append(loss.item())
                con_recon_loss_l.append(con_rec_loss.item())
                dis_recon_loss_l.append(dis_rec_loss.item())
                dis_recon_acc_l.append(dis_rec_acc)
                type_loss_l.append(type_loss.item())
                smooth_loss_l.append(smooth_loss.item())


                if (i + 1) % 2 == 0:
                    # print("\titers: {0}, epoch: {1} | loss_all: {2:.7f} |dis_rec_loss: {2:.7f} |con_rec_loss: {2:.7f} ".format(i + 1,epoch + 1, loss.item(),dis_rec_loss.item(),con_rec_loss.item()))
                    print(
                        "iters: {0}, epoch: {1} | loss_all: {2:.7f} ".format(
                            i + 1, epoch + 1, loss.item()))
                    print('Con_reconstruction_loss = %f' % con_rec_loss.item())
                    print('Dis_reconstruction_loss = %f' % dis_rec_loss.item())
                    print('Dis_reconstruction_acc = %f' % dis_rec_acc)
                    print('Smooth_loss = %f' % smooth_loss.item())
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


            vali_loss = self.vali(vali_data, vali_loader, con_rec_criterion,dis_rec_criterion)



            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.5f} Vali Loss: {3:.5f} "
                    .format(epoch + 1, train_steps, train_loss_l, vali_loss))

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


            if self.args.lradj_flag:

                    #  learning rate update
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model, (
        train_loss_list, con_recon_loss_list, dis_recon_loss_list, dis_recon_acc_list, type_loss_list, smooth_loss_list)



    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))



        attens_energy_con = []
        attens_energy_dis = []
        self.model.eval()
        self.anomaly_criterion_con = nn.MSELoss(reduce=False)
        self.anomaly_criterion_dis = nn.CrossEntropyLoss(reduce=False)


        # (1) stastic on the train set
        print('process training data')
        with torch.no_grad():
            for i, (batch_x, label) in enumerate(train_loader):
                print(i)
                #  B*T*D -> B*D*T
                batch_x = batch_x.float().to(self.device).transpose(-1, -2)

                batch_x_dis, batch_x_con = self.Norm_Discretization(batch_x, typeflag=self.args.typeflag,
                                                                    return_flag='S')

                x_dis_pe = None

                x_lcv, x_lcv_embed, x_con_embed, means_con, stdev_con = self.model.DATA_EMBED(batch_x_dis, x_dis_pe,
                                                                                              batch_x_con)


                x_lcv_embed, x_con_embed, dis_att, con_att, dis_con_attn, con_dis_attn = self.model.Encoding(
                    x_lcv_embed,
                    x_con_embed)

                x_dis_rec, x_con_rec = self.model.SSL_RECON(x_lcv_embed, x_con_embed, means_con, stdev_con)

                # continuous-variable criterion
                score_con = torch.mean(self.anomaly_criterion_con(batch_x_con, x_con_rec), dim=-2)
                score_con = score_con.detach().cpu().numpy()
                attens_energy_con.append(score_con)

                # discrete-variable criterion
                score_dis = torch.mean(self.anomaly_criterion_dis(x_dis_rec.reshape(-1, 2),
                                                                  batch_x_dis.reshape(-1, 1).squeeze(
                                                                      -1).long()).reshape(-1, self.args.dis_dim,
                                                                                          self.args.T), dim=-2)

                score_dis = score_dis.detach().cpu().numpy()
                attens_energy_dis.append(score_dis)


        attens_energy_con = np.concatenate(attens_energy_con, axis=0).reshape(-1)
        train_energy_con = np.array(attens_energy_con)

        attens_energy_dis = np.concatenate(attens_energy_dis, axis=0).reshape(-1)
        train_energy_dis = np.array(attens_energy_dis)

        print('Stastic on the train set Done !!')
        # (2) find the threshold
        attens_energy_con = []
        attens_energy_dis = []
        test_labels = []
        print('process test data')
        for i, (batch_x, batch_y) in enumerate(test_loader):
            print(i)
            batch_x = batch_x.float().to(self.device).transpose(-1, -2)

            batch_x_dis, batch_x_con = self.Norm_Discretization(batch_x, typeflag=self.args.typeflag,
                                                                return_flag='S')

            x_dis_pe = None

            x_lcv, x_lcv_embed, x_con_embed, means_con, stdev_con = self.model.DATA_EMBED(batch_x_dis, x_dis_pe,
                                                                                          batch_x_con)

            x_lcv_embed, x_con_embed, dis_att, con_att, dis_con_attn, con_dis_attn = self.model.Encoding(
                x_lcv_embed,
                x_con_embed)

            x_dis_rec, x_con_rec = self.model.SSL_RECON(x_lcv_embed, x_con_embed, means_con, stdev_con)

            # continuous-variable criterion
            score_con = torch.mean(self.anomaly_criterion_con(batch_x_con, x_con_rec) ,dim=-2)
            score_con = score_con.detach().cpu().numpy()
            attens_energy_con.append(score_con)

            # discrete-variable criterion
            score_dis = torch.mean(self.anomaly_criterion_dis(x_dis_rec.reshape(-1, 2),
                                                   batch_x_dis.reshape(-1, 1).squeeze(-1).long()).reshape(-1,self.args.dis_dim,self.args.T),dim=-2)

            score_dis = score_dis.detach().cpu().numpy()
            attens_energy_dis.append(score_dis)

            test_labels.append(batch_y)

        attens_energy_con = np.concatenate(attens_energy_con, axis=0).reshape(-1)
        test_energy_con = np.array(attens_energy_con)

        attens_energy_dis = np.concatenate(attens_energy_dis, axis=0).reshape(-1)
        test_energy_dis = np.array(attens_energy_dis)



        combined_energy_con = np.concatenate([train_energy_con, test_energy_con], axis=0)
        threshold_con = np.percentile(combined_energy_con, 100 - self.args.anomaly_ratio)

        combined_energy_dis = np.concatenate([train_energy_dis, test_energy_dis], axis=0)
        threshold_dis = np.percentile(combined_energy_dis, 100 - self.args.anomaly_ratio)


        print("Threshold_continuous :", threshold_con)
        print("Threshold_discrete :", threshold_dis)

        print('Fine the threshold Done !!')
        # (3) evaluation on the test set

        if self.args.anomaly_metric == 'all':
            pred = ((test_energy_con > threshold_con) +(test_energy_dis > threshold_dis)).astype(int)
        else:
            pred = ((test_energy_con > threshold_con)).astype(int)



        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)

        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        print('Evaluation on the test set Done !!')

        # (4) detection adjustment
        gt, pred = adjustment(gt, pred)

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))
        print('Detection adjustment Done !!')



        return accuracy, precision, recall, f_score







