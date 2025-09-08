'''
ASTGCN
'''
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.linalg import eigs
import numpy as np



class Spatial_Attention_layer(nn.Module):
    '''
    compute spatial attention scores
    '''
    def __init__(self, in_channels, num_of_vertices, num_of_timesteps):
        super(Spatial_Attention_layer, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps).cuda())
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps).cuda())
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels).cuda())
        self.bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices).cuda())
        self.Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).cuda())
        nn.init.uniform_(self.W1, -0.1, 0.1)
        nn.init.xavier_uniform_(self.W2)
        nn.init.uniform_(self.W3, -0.1, 0.1)
        nn.init.constant_(self.bs, 0.0)
        nn.init.xavier_uniform_(self.Vs)    
    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B,N,N)
        '''

        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)  # (b,N,F,T)(T)->(b,N,F)(F,T)->(b,N,T)

        rhs = torch.matmul(self.W3, x).transpose(-1, -2)  # (F)(b,N,F,T)->(b,N,T)->(b,T,N)

        product = torch.matmul(lhs, rhs)  # (b,N,T)(b,T,N) -> (B, N, N)

        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # (N,N)(B, N, N)->(B,N,N)

        S_normalized = F.softmax(S, dim=1)

        return S_normalized


class cheb_conv_withSAt(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv_withSAt, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).cuda()) for _ in range(K)])
        for theta in self.Theta:
                nn.init.xavier_uniform_(theta)
    def forward(self, x, spatial_attention):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).cuda()  # (b, N, F_out)

            for k in range(self.K):
                T_k = self.cheb_polynomials[k]  # (N,N)

                T_k_with_at = T_k.mul(spatial_attention)  # (N,N)*(N,N) = (N,N) ���к�Ϊ1, �����н��й�һ��

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)
                # (N, N)(b, N, F_in) = (b, N, F_in) ��Ϊ����ˣ����Զ��к�Ϊ1��Ϊ���к�Ϊ1����һ��֮��Ϊ1���������

                output = output + rhs.matmul(theta_k)  # (b, N, F_in)(F_in, F_out) = (b, N, F_out)

            outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)

        return F.relu(torch.cat(outputs, dim=-1))  # (b, N, F_out, T)


class Temporal_Attention_layer(nn.Module):

    def __init__(self, in_channels, num_of_vertices, num_of_timesteps):
        super(Temporal_Attention_layer, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices).cuda())
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices).cuda())
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels).cuda())
        self.be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps).cuda())
        self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps).cuda())
        nn.init.uniform_(self.U1, -0.1, 0.1)
        nn.init.xavier_uniform_(self.U2)
        nn.init.uniform_(self.U3, -0.1, 0.1)
        nn.init.constant_(self.be, 0.0)
        nn.init.xavier_uniform_(self.Ve)    
    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B, T, T)
        '''
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        # x:(B, N, F_in, T) -> (B, T, F_in, N)
        # (B, T, F_in, N)(N) -> (B,T,F_in)
        # (B,T,F_in)(F_in,N)->(B,T,N)

        rhs = torch.matmul(self.U3, x)  # (F)(B,N,F,T)->(B, N, T)

        product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)

        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)

        E_normalized = F.softmax(E, dim=1)

        return E_normalized


class cheb_conv(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(in_channels, out_channels).cuda()) for _ in range(K)])

    def forward(self, x):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).cuda() # (b, N, F_out)

            for k in range(self.K):
                T_k = self.cheb_polynomials[k]  # (N,N)

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1)

                output = output + rhs.matmul(theta_k)

            outputs.append(output.unsqueeze(-1))

        return F.relu(torch.cat(outputs, dim=-1))


class ASTGCN_block(nn.Module):

    def __init__(self, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials,
                 num_of_vertices, num_of_timesteps):
        super(ASTGCN_block, self).__init__()
        self.TAt = Temporal_Attention_layer(in_channels, num_of_vertices, num_of_timesteps)
        self.SAt = Spatial_Attention_layer(in_channels, num_of_vertices, num_of_timesteps)
        self.cheb_conv_SAt = cheb_conv_withSAt(K, cheb_polynomials, in_channels, nb_chev_filter)
        self.time_conv = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides),
                                   padding=(0, 1))
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(nb_time_filter)  # ��Ҫ��channel�ŵ����һ��ά����

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, nb_time_filter, T)
        '''
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        # TAt
        temporal_At = self.TAt(x)  # (b, T, T)
        # print(f"After layer1: {temporal_At.mean().item()}, {temporal_At.std().item()}")
        x_TAt = torch.matmul(x.reshape(batch_size, -1, num_of_timesteps), temporal_At).reshape(batch_size,
                                                                                               num_of_vertices,
                                                                                               num_of_features,
                                                                                               num_of_timesteps)

        # SAt
        spatial_At = self.SAt(x_TAt)
        # print(f"After layer2: {spatial_At.mean().item()}, {spatial_At.std().item()}")
        # cheb gcn
        spatial_gcn = self.cheb_conv_SAt(x, spatial_At)  # (b,N,F,T)
        # spatial_gcn = self.cheb_conv(x)
        # print(f"After layer3: {spatial_gcn.mean().item()}, {spatial_gcn.std().item()}")
        # convolution along the time axis
        time_conv_output = self.time_conv(
            spatial_gcn.permute(0, 2, 1, 3))  # (b,N,F,T)->(b,F,N,T) ��(1,3)�ľ����ȥ��->(b,F,N,T)
        
        # print(f"After layer4: {time_conv_output.mean().item()}, {time_conv_output.std().item()}")
        # residual shortcut
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))  # (b,N,F,T)->(b,F,N,T) ��(1,1)�ľ����ȥ��->(b,F,N,T)
       # print(f"After layer5: {x_residual.mean().item()}, {x_residual.std().item()}")
        x_residual = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)
        # (b,F,N,T)->(b,T,N,F) -ln-> (b,T,N,F)->(b,N,F,T)
        # print(f"After layer6: {x_residual.mean().item()}, {x_residual.std().item()}")
        return x_residual


class Model(nn.Module):
    def __init__(self, configs, nb_block=2, in_channels=1, K=3, nb_chev_filter=64, nb_time_filter=64,
                 time_strides=2):
        '''
        :param nb_block:
        :param in_channels:
        :param K:
        :param nb_chev_filter:
        :param nb_time_filter:
        :param time_strides:
        :param cheb_polynomials:
        :param nb_predict_step:
        '''
        super(Model, self).__init__()
        num_of_vertices = configs.enc_in
        len_input = configs.seq_len
        num_for_predict = configs.pred_len
        adj_mx = pd.read_csv(configs.adj_path).values.astype('float32')
        # adj_mx = adj_mx/adj_mx.max()
        adj_mx = np.where(adj_mx != 0, 1, 0).astype('float32')

        # adj_mx is 0-1 pure adajecent matrix
        L_tilde = scaled_Laplacian(adj_mx)
        cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).cuda() for i in cheb_polynomial(L_tilde, K=3)]       
        self.BlockList = nn.ModuleList([ASTGCN_block(in_channels, K, nb_chev_filter, nb_time_filter,
                                                     time_strides, cheb_polynomials, num_of_vertices, len_input)])
        self.BlockList.extend([
            ASTGCN_block(nb_time_filter, K, nb_chev_filter, nb_time_filter, 1,
                         cheb_polynomials, num_of_vertices,
                         len_input // time_strides) for _ in range(nb_block - 1)])
        self.final_conv = nn.Conv2d(int(len_input / time_strides), num_for_predict, kernel_size=(1, nb_time_filter))

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        # print('x shape is', x.shape )
        x=x.unsqueeze(-1).permute(0,2,3,1)   # [b,t,n,1] > [b,n,1,t]
        # print('x 2 shape is', x.shape )
        
        for block in self.BlockList:
            x = block(x)
        
        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1]      # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)
        # print('out shape ',output.shape)
        return output


def adj_tans(sensor_ids_file, distance_file):
    with open(sensor_ids_file) as f:
        sensor_ids = f.read().strip().split(',')
    sensor_ids2 = pd.DataFrame({'from': sensor_ids, 'to': sensor_ids})
    sensor_ids2['No'] = sensor_ids2.index
    distance_df = pd.read_csv(distance_file, dtype={'from': 'str', 'to': 'str'})
    result1 = pd.merge(distance_df, sensor_ids2, on='from', suffixes=['', '_sen'])
    del result1['to_sen']
    result = pd.merge(result1, sensor_ids2, on='to')
    result.rename(columns={'from_x': 'from'}, inplace=True)
    result = result.drop(result[result.cost == 0].index)
    result = result.sort_values(by=['No_x', 'No_y'])
    del result['from']
    del result['to']
    del result['from_y']
    result.rename(columns={'No_x': 'from'}, inplace=True)
    result.rename(columns={'No_y': 'to'}, inplace=True)
    result = result[distance_df.columns]
    result = result.reset_index(drop=True)
    return result


def get_adjacency_matrix(distance_df_filename, num_of_vertices):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information
    num_of_vertices: int, the number of vertices
    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    for index in distance_df_filename.index:
        i, j, distance = distance_df_filename['from'][index], distance_df_filename['to'][index], float(
            distance_df_filename['cost'][index])
        A[i, j] = 1
        distaneA[i, j] = distance
    return A, distaneA


def scaled_Laplacian(W):
    '''
    compute \tilde{L}
    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices
    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    lambda_max = eigs(L, k=1, which='LR')[0].real
    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}
    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)
    K: the maximum order of chebyshev polynomials
    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}
    '''
    N = L_tilde.shape[0]
    cheb_polynomials = [np.identity(N), L_tilde.copy()]
    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])
    return cheb_polynomials