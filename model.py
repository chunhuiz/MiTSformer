import torch
import torch.nn as nn
import torch.nn.functional as F

from layers_mits import FullAttention, AttentionLayer, Recover_CNN, ReverseLayerF



class VAR_EMBED(nn.Module):
    '''
        input: recovered_LCV or CV of shape B*D*T
        output: variable-wise embedded features of shape B*D*E
        '''
    def __init__(self, T, d_model, dropout=0.1):
        super(VAR_EMBED, self).__init__()
        self.value_embedding = nn.Linear(T, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class Discriminator(nn.Module):
    '''
    variable modality discriminator (adversarial training)
    input:  embeded variable sets (B*D)*E
    output: type 0 or 1           (B*D)*2
    '''


    def __init__(self, d_model):
        super(Discriminator, self).__init__()

        self.dis1 = nn.Linear(d_model, d_model*4)
        self.bn1 = nn.BatchNorm1d(d_model*4)
        self.dis2 = nn.Linear(d_model*4, d_model * 4)
        self.bn2 = nn.BatchNorm1d(d_model*4)
        self.dis3 = nn.Linear(d_model*4, 2)


    def forward(self, x):
        x = self.bn1(F.relu(self.dis1(x)))
        x = self.bn2(F.relu(self.dis2(x)))
        x = self.dis3(x)
        return x


class SELF_EncoderLayer(nn.Module):
    '''
       input:  variable-wise embeddings of shape B*D*E
       output: variable-wise embeddings of shape B*D*E
    '''

    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="gelu"):
        super(SELF_EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn

class CROSS_EncoderLayer(nn.Module):
    '''
       input:  DIScrete variable-wise embeddings of shape B*D*E
               CONtinuous variable-wise embeddings of shape B*C*E
       output: DIScrete variable-wise embeddings of shape B*D*E
               CONtinuous variable-wise embeddings of shape B*C*E

               DIS-CON attention H*D*C
               CON-DIS attention H*C*D
    '''

    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="gelu"):
        super(CROSS_EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.cross_attention_dis = attention
        self.conv1_dis = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2_dis = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1_dis = nn.LayerNorm(d_model)
        self.norm2_dis = nn.LayerNorm(d_model)
        self.dropout_dis = nn.Dropout(dropout)
        self.activation_dis = F.relu if activation == "relu" else F.gelu

        self.cross_attention_con = attention
        self.conv1_con = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2_con = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1_con = nn.LayerNorm(d_model)
        self.norm2_con = nn.LayerNorm(d_model)
        self.dropout_con = nn.Dropout(dropout)
        self.activation_con = F.relu if activation == "relu" else F.gelu


    def forward(self, x_dis,x_con, attn_mask=None, tau=None, delta=None):
        x_dis_raw = x_dis
        x_con_raw = x_con
        x_dis_add, x_dis_attn = self.cross_attention_dis(x_dis_raw, x_con_raw, x_con_raw)
        x_dis_out = x_dis_raw + self.dropout_dis(x_dis_add)
        x_dis_out = self.norm1_dis(x_dis_out)

        y_dis = x_dis_out
        y_dis = self.dropout_dis(self.activation_dis(self.conv1_dis(y_dis.transpose(-1, -2))))
        y_dis = self.dropout_dis(self.conv2_dis(y_dis).transpose(-1, -2))
        x_dis_out = x_dis_out + y_dis

        x_dis_out = self.norm2_dis(x_dis_out + x_dis_raw)



        x_con_add, x_con_attn = self.cross_attention_con(x_con_raw, x_dis_raw, x_dis_raw)
        x_con_out = x_con_raw + self.dropout_con(x_con_add)

        x_con_out = self.norm1_con(x_con_out)

        y_con = x_con_out
        y_con = self.dropout_con(self.activation_con(self.conv1_con(y_con.transpose(-1, -2))))
        y_con = self.dropout_con(self.conv2_con(y_con).transpose(-1, -2))
        x_con_out = x_con_out + y_con

        #  residual
        x_con_out = self.norm2_con(x_con_out+x_con_raw)

        return x_dis_out, x_con_out, x_dis_attn, x_con_attn

class Attention_EncoderBlock(nn.Module):
    '''
       input:  DIScrete variable-wise embeddings of shape B*D*E
               Continuous variable-wise embeddings of shape B*C*E

       middle: intra-modality self-attention  and inter-modality cross-attention


       output: DIScrete variable-wise embeddings of shape B*D*E
               Continuous variable-wise embeddings of shape B*C*E
    '''

    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="gelu"):
        super(Attention_EncoderBlock, self).__init__()
        self.intra_dis_attention = SELF_EncoderLayer(attention, d_model,dropout=dropout,activation=activation)
        self.intra_con_attention = SELF_EncoderLayer(attention, d_model,dropout=dropout,activation=activation)

        self.cross_attention = CROSS_EncoderLayer(attention, d_model,dropout=dropout,activation=activation)



    def forward(self, x_dis,x_con, attn_mask=None, tau=None, delta=None):
        x_dis_raw,dis_att = self.intra_dis_attention(x_dis)
        x_con_raw,con_att = self.intra_con_attention(x_con)

        x_dis_out, x_con_out, dis_con_att, con_dis_att = self.cross_attention (x_dis_raw,x_con_raw)


        return x_dis_out, x_con_out, dis_att,con_att,dis_con_att,con_dis_att

class DIS_Decoder(nn.Module):
    '''
       input:  discrete variable embeddings of shape B*D*E
       output: original discrete variable of shape B*D*T*2
    '''

    def __init__(self,d_model,T):
        super(DIS_Decoder, self).__init__()
        self.T = T
        self.decoder = nn.Linear(d_model,2*T,bias=True)


    def forward(self, x):
        B = x.shape[0]
        D = x.shape[1]
        return self.decoder(x).reshape(B,D,self.T,2)

class CON_Decoder(nn.Module):
    '''
       input:  continuous variable embeddings of shape B*C*E
       output: original discrete variable of shape B*C*T
    '''

    def __init__(self,d_model,T):
        super(CON_Decoder, self).__init__()
        self.T = T
        self.decoder = nn.Linear(d_model,T,bias=True)


    def forward(self, x):

        return self.decoder(x)

class Model(nn.Module):
    def __init__(self,args):
        d_model=args.d_model
        T = args.T
        block_layers = args.block_layers
        d_heads = args.n_heads
        drop = args.dropout
        activation=args.activation
        self.args=args
        self.task_name = args.task_name

        super(Model, self).__init__()

        # smooth array
        arr = torch.zeros((T - 2, T))
        for i in range(T-2):
            arr[i, i] = -1
            arr[i, i + 1] = 2
            arr[i, i + 2] = -1
        self.smooth_arr = arr

        self.block_layers = block_layers


        self.recover_CNN = Recover_CNN(in_channels=1, hidden_rec=d_model, n_layers=3, kernel_size=3)

        self.embed_dis = VAR_EMBED(T = T, d_model= d_model)
        self.embed_con = VAR_EMBED(T = T, d_model= d_model)

        self.variable_discriminator = Discriminator(d_model=d_model)

        self.attention_encoder = \
            nn.ModuleList([Attention_EncoderBlock(
                attention=AttentionLayer(FullAttention(False, attention_dropout=drop,output_attention=True), d_model, d_heads),d_model=d_model,dropout=drop,activation=activation) for i in range(block_layers)])


        self.dis_decoder = DIS_Decoder(d_model,T)
        self.con_decoder = CON_Decoder(d_model,T)

        if self.args.task_name == "extrinsic_regression":
            self.task_act = F.gelu
            self.task_dropout = nn.Dropout(args.dropout)
            self.task_projection = nn.Linear(
                args.d_model * args.p, 1)

        if self.args.task_name == "classification":
            self.task_act = F.gelu
            self.task_dropout = nn.Dropout(args.dropout)
            self.task_projection = nn.Linear(
                args.d_model * args.p, args.num_classes)

        if self.args.task_name == 'anomaly_detection':
            #  self-supervised reconstruction task !
            self.task_projector = None

        if self.args.task_name == 'imputation':
            #  self-supervised reconstruction task !
            self.task_projector_con = CON_Decoder(d_model,T)

        if self.args.task_name == 'long_term_forecast':

            self.dis_task_projector = DIS_Decoder(d_model, args.pred_len)
            self.con_task_projector = CON_Decoder(d_model, args.pred_len)



    def DATA_EMBED(self,x_dis,x_dis_pe,x_con,mask = None,x_mark_enc=None):
        '''

        (1). Variable Embedding STEP
        :param x_dis: discrete variable of shape B*D*T
        :param x_dis_pe: positional embedding of DV of shape D*B*2*T
        :param x_con: continuous variable of shape B*C*T
        :param mask: specifically for imputation task (only for continuous variables)  shape B*C*T
        :param x_mark_enc :mark B*d*T
        :return: embeddings of LCV B*D*E  and CV  B*C*E
                and LCV B*D*T
                and mean_con, std_con
        '''



        B = x_dis.shape[0]
        dis_dim = x_dis.shape[1]
        T = x_dis.shape[2]
        con_dim = x_con.shape[1]

        if self.args.dis_embed:
            #  x_dis_PE D*B*3*T
            x_dis_PE = torch.cat([x_dis.transpose(0, 1).unsqueeze(-2), x_dis_pe], dim=-2)

            #  x_dis_PE (DB)*3*T for TCN processing
            x_dis_PE = x_dis_PE.reshape(B * dis_dim, 3, T)
        else:
            x_dis_PE = x_dis.reshape(B * dis_dim, T).unsqueeze(1)


        # x_lcv B*D*T
        x_lcv = self.recover_CNN(x_dis_PE).reshape(B, dis_dim, 1, T).squeeze(-2)


        # Instance Normalization for LCV
        means_lcv = x_lcv.mean(-1, keepdim=True).detach()
        x_lcv = x_lcv - means_lcv
        stdev_lcv = torch.sqrt(torch.var(x_lcv, dim=-1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_lcv /= stdev_lcv

        x_lcv_embed = self.embed_dis(x_lcv)




        if mask != None:
            # Instance Normalization for Continuous Variables

            means_con = (torch.sum(x_con, dim=-1) / torch.sum(mask == 1, dim=-1)).unsqueeze(-1).detach()
            x_con = x_con - means_con
            x_con = x_con.masked_fill(mask == 0, 0)
            stdev_con = (torch.sqrt(torch.sum(x_con * x_con, dim=-1) /torch.sum(mask == 1, dim=-1) + 1e-5)).unsqueeze(-1).detach()
            x_con /= stdev_con
            if x_mark_enc !=None:
                x_con_embed = self.embed_con(torch.cat([x_con,x_mark_enc],dim=1))
            else:
                x_con_embed = self.embed_con(x_con)

        else:
            means_con = x_con.mean(-1, keepdim=True).detach()
            x_con = x_con - means_con
            stdev_con = torch.sqrt(torch.var(x_con, dim=-1, keepdim=True, unbiased=False) + 1e-5).detach()
            x_con /= stdev_con

            if x_mark_enc != None:
                x_con_embed = self.embed_con(torch.cat([x_con, x_mark_enc], dim=1))
            else:
                x_con_embed = self.embed_con(x_con)



        return x_lcv, x_lcv_embed, x_con_embed,means_con,stdev_con

    def variable_discriminate(self,x_lcv_embed,x_con_embed,alpha=1):
        '''

        (2). Variable Discriminate STEP

        :param x_lcv_embed: embeddings of LCV B*D*E
        :param x_con_embed: embeddings of CV  B*C*E
        :return: predicted label  (B*D+B*C)*2
                 and true  label   (B*D+B*C)*1
        '''

        x_con_embed = x_con_embed[:,:self.args.con_dim,:]

        B = x_lcv_embed.shape[0]
        dis_dim = x_lcv_embed.shape[1]
        E = x_lcv_embed.shape[2]
        con_dim = x_con_embed.shape[1]

        rev_x_lcv_embed = ReverseLayerF.apply(x_lcv_embed, alpha).reshape(B*dis_dim,E)
        rev_x_con_embed = ReverseLayerF.apply(x_con_embed, alpha).reshape(B*con_dim,E)

        domain_label_pred = self.variable_discriminator(torch.cat([rev_x_lcv_embed,rev_x_con_embed],dim=0))
        domain_label_true = torch.cat([torch.ones(B*dis_dim).to(x_lcv_embed.device),torch.zeros(B*con_dim).to(x_lcv_embed.device)],dim=0)

        return domain_label_pred,domain_label_true

    def Encoding(self,x_lcv_embed,x_con_embed):
        '''
        (3). Feature Encoding

        :param x_lcv_embed: embeddings of LCV B*D*E
        :param x_con_embed: embeddings of CV  B*C*E
        :return: x_lcv_embed: embeddings of LCV B*D*E and  x_con_embed: embeddings of CV  B*C*E
                and corresponding attention list
        '''

        dis_attn = []
        con_attn = []
        dis_con_attn = []
        con_dis_attn = []
        for i in range(self.block_layers):
            x_lcv_embed, x_con_embed, dis_att, con_att, dis_con_att, con_dis_att = self.attention_encoder[i](
                x_lcv_embed, x_con_embed)
            dis_attn.append(dis_att)
            con_attn.append(con_att)
            dis_con_attn.append(dis_con_att)
            con_dis_attn.append(con_dis_att)

        return x_lcv_embed, x_con_embed,dis_attn,con_attn,dis_con_attn,con_dis_attn



    def SSL_RECON(self,x_lcv_embed,x_con_embed,means_con,stdev_con):

        '''

        (4). Self-reconstruction Step

        :param x_lcv_embed: embeddings of LCV B*D*E
        :param x_con_embed: embeddings of CV  B*C*E
        :return: reconstructed DV of B*D*T*2 and CV B*C*T

        '''

        x_dis_rec = self.dis_decoder(x_lcv_embed)

        # discard the mark_token
        x_con_rec = self.con_decoder(x_con_embed)[:,:self.args.con_dim,:]
        con_dim = x_con_embed.shape[1]

        # De-Normalization for Continuous variable
        x_con_rec = x_con_rec * (stdev_con[:, :, 0].unsqueeze(-1).repeat(1, 1, self.args.T))
        x_con_rec = x_con_rec + (means_con[:, :, 0].unsqueeze(-1).repeat(1, 1, self.args.T))

        return x_dis_rec, x_con_rec


    def Classification(self,x_lcv_embed,x_con_embed):
        '''
        task(1). Classification

        :param x_lcv_embed: embeddings of LCV B*D*E
        :param x_con_embed: embeddings of CV  B*C*E
        :return: prediction of labels B*K
        '''

        # B*(D+C)*E
        B = x_lcv_embed.shape[0]
        fused_feature = torch.cat([x_lcv_embed,x_con_embed[:,:self.args.con_dim,:]],dim=1).reshape(B,-1)
        output = self.task_projection(self.task_dropout(self.task_act(fused_feature)))
        return output

    def Extrinsic_Regression(self,x_lcv_embed,x_con_embed):
        '''
        task(2). Extrinsic_Regression

        :param x_lcv_embed: embeddings of LCV B*D*E
        :param x_con_embed: embeddings of CV  B*C*E
        :return: prediction of labels B*1
        '''

        # B*(D+C)*E
        B = x_lcv_embed.shape[0]
        fused_feature = torch.cat([x_lcv_embed,x_con_embed[:,:self.args.con_dim,:]],dim=1).reshape(B,-1)
        output = self.task_projection(self.task_dropout(self.task_act(fused_feature)))

        return output

    def Forecast(self,x_lcv_embed,x_con_embed,means_con,stdev_con):
        '''
        task(3). Forecast

        :param x_lcv_embed: embeddings of LCV B*D*E
        :param x_con_embed: embeddings of CV  B*C*E
        :return: Forecasting of LCV B*D*T_fore*2
                                 CV B*C*T_fore
        '''


        dis_fore = self.dis_task_projector(x_lcv_embed)
        # discard the mark_token
        con_fore = self.con_task_projector(x_con_embed)[:,:self.args.con_dim,:]

        con_fore = con_fore * (stdev_con[:, :, 0].unsqueeze(-1).repeat(1, 1, self.args.pred_len))
        con_fore = con_fore + (means_con[:, :, 0].unsqueeze(-1).repeat(1, 1, self.args.pred_len))

        return dis_fore, con_fore


    def Imputation(self,x_con_embed,means_con,stdev_con):
        '''
        task(1). Imputation by Masked Autoencoding (Special)
                 Note the SSL=reconstruction only applies for unmasked points, and masked autoencoding for masked points

        :param x_con_embed: embeddings of CV  B*C*E
        :return: prediction of labels B*1
        '''

        x_con_rec = self.task_projector_con(x_con_embed)[:,:self.args.con_dim,:]
        con_dim = x_con_embed.shape[1]

        # De-Normalization for Continuous variable
        # mask
        x_con_rec = x_con_rec * (stdev_con[:, :, 0].unsqueeze(-1).repeat(1, 1, self.args.T))
        x_con_rec = x_con_rec + (means_con[:, :, 0].unsqueeze(-1).repeat(1, 1, self.args.T))

        return x_con_rec


    def forward(self,x_dis,x_con,x_dis_pe=None,mask = None,x_mark_enc = None):

        x_lcv, x_lcv_embed, x_con_embed, means_con, stdev_con = self.DATA_EMBED(x_dis, x_dis_pe,x_con, mask = mask,x_mark_enc=x_mark_enc)


        x_lcv_embed, x_con_embed, dis_att, con_att, dis_con_attn, con_dis_attn = self.Encoding(x_lcv_embed, x_con_embed)


        if self.task_name == 'long_term_forecast' :
            dis_fore,con_fore = self.Forecast(x_lcv_embed, x_con_embed,means_con, stdev_con)
            return dis_fore,con_fore

        if self.task_name == 'imputation':
            dec_out = self.Imputation(x_con_embed,means_con,stdev_con)
            return dec_out  # [B, C, T]

        if self.task_name == 'anomaly_detection':
            x_dis_rec, x_con_rec = self.SSL_RECON(x_lcv_embed, x_con_embed,means_con,stdev_con)
            return x_dis_rec, x_con_rec

        if self.task_name == 'classification':
            dec_out = self.Classification(x_lcv_embed,x_con_embed)
            return dec_out  # [B, K]

        if self.task_name == 'extrinsic_regression':
            dec_out = self.Extrinsic_Regression(x_lcv_embed,x_con_embed)
            return dec_out  # [B, 1]

        return None

