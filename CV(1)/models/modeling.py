import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
from obspy.signal.trigger import ar_pick

from .criterion import _loss


class ARPicker():
    def __init__(self, sample_rate=100, f1=1.0, f2=20.0,
                 lta_p=1.5, sta_p=0.2, lta_s=4.0, sta_s=1.0,
                 m_p=2, m_s=8, l_p=0.1, l_s=0.2, s_pick=True):
        self.sample_rate = sample_rate
        self.f1 = f1
        self.f2 = f2
        self.lta_p = lta_p
        self.sta_p = sta_p
        self.lta_s = lta_s
        self.sta_s = sta_s
        self.m_p = m_p
        self.m_s = m_s
        self.l_p = l_p
        self.l_s = l_s
        self.s_pick = s_pick

    def __call__(self, x):
        # single signal phase picking
        # x is shaped (3, wave_length)
        # ground_truth is shaped (2, )
        p_pick, s_pick = ar_pick(x[2, :], x[1, :], x[0, :], samp_rate=self.sample_rate, 
                                 f1=self.f1, f2=self.f2, lta_p=self.lta_p, sta_p=self.sta_p, 
                                 lta_s=self.lta_s, sta_s=self.sta_s, m_p=self.m_p, m_s=self.m_s,
                                 l_p=self.l_p, l_s=self.l_s, s_pick=self.s_pick)
        return p_pick, s_pick


class Conv1dSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(Conv1dSame, self).__init__()
        self.cut_last = ((kernel_size % 2 == 0) and (stride == 1) and (dilation % 2 == 1))
        self.padding = math.ceil((1 - stride + dilation * (kernel_size - 1)) / 2)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=self.padding+1,
            stride=stride,
            dilation=dilation
        )

    def forward(self, x):
        if self.cut_last:
            return self.conv(x)[:, :, :-1]
        else:
            return self.conv(x)


class DownSample(nn.Module):
    '''
    (in_channel, in_length) 
    --conv1--> --ReLU--> (in_channel, out_length) 
    --conv2--> --ReLU--> (out_channel, out_length)
    '''
    
    def __init__(self, in_channel, in_length, out_channel, stride, kernel_size, padding):
        super(DownSample, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.in_length = in_length
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
        # Conv1d: (N, C_in, L) --> (N, C_out, L_out)
        # (N, in_channel, in_length) --> (N, in_channel, out_length)
        self.conv1 = nn.Conv1d(
            in_channels=self.in_channel, 
            out_channels=self.in_channel, 
            kernel_size=self.kernel_size, 
            padding='same',
            bias=False)
        # (N, in_channel, in_length) --> (N, out_channel, out_length)
        self.conv2 = nn.Conv1d(
            in_channels=self.in_channel, 
            out_channels=self.out_channel, 
            kernel_size=self.kernel_size, 
            padding=self.padding, 
            stride=self.stride,
            bias=False)
        self.downsamplenet = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.ReLU()
        )

    def forward(self, x):
        return self.downsamplenet(x)


class UpSample(nn.Module):
    '''
    (in_channel, in_length) 
    --deconv--> --ReLU--> (hidden_channel, out_length) 
    --concat--> (hidden_channel + concat_channel, out_length)
    --conv--> --ReLU--> (out_channel, out_length)
    '''
    def __init__(self, in_channel, out_channel, concat_channel, 
                 hidden_channel, in_length, kernel_size, 
                 stride, padding, output_padding):
        super(UpSample, self).__init__()
        # ConvTranspose1d: (N, C_in, L_in) --> (N, C_out, L_out)
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.concat_channel = concat_channel
        self.hidden_channel = hidden_channel
        self.in_length = in_length
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        self.deconv = nn.ConvTranspose1d(
            in_channels=self.in_channel, 
            out_channels=self.hidden_channel,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            bias=False)
        # After ReLU, (hidden_channel + concat_channel, out_length)
        self.conv = nn.Conv1d(
            in_channels=self.hidden_channel+self.concat_channel, 
            out_channels=self.out_channel,
            kernel_size=self.kernel_size,
            padding='same',
            bias= False)

    def forward(self, x, skip):
        unsamplenet = nn.Sequential(self.deconv, nn.ReLU())
        hidden_layer = unsamplenet(x)
        after_concat = torch.cat((hidden_layer, skip), dim=1)
        convnet = nn.Sequential(self.conv, nn.ReLU())
        return convnet(after_concat)


class PhaseNet(nn.Module):
    def __init__(self, config):
        # config: dict str -> any
        super(PhaseNet, self).__init__()
        self.config = config
        self.conv1 = nn.Conv1d(
            in_channels=config['conv1_in_channel'], 
            out_channels=config['conv1_out_channel'], 
            kernel_size=config['conv1_kernel_size'],
            padding='same',
            bias=False)
        self.conv2 = nn.Conv1d(
            in_channels=config['conv2_in_channel'], 
            out_channels=config['conv2_out_channel'], 
            kernel_size=config['conv2_kernel_size'],
            padding='same',
            bias=False)
        self.downsample1 = DownSample(
            in_channel=config['downsample1_in_channel'], 
            in_length=config['downsample1_in_length'], 
            out_channel=config['downsample1_out_channel'], 
            stride=config['downsample1_stride'], 
            kernel_size=config['downsample1_kernal_size'], 
            padding=config['downsample1_padding'])
        self.downsample2 = DownSample(
            in_channel=config['downsample2_in_channel'], 
            in_length=config['downsample2_in_length'], 
            out_channel=config['downsample2_out_channel'], 
            stride=config['downsample2_stride'], 
            kernel_size=config['downsample2_kernal_size'], 
            padding=config['downsample2_padding'])
        self.downsample3 = DownSample(
            in_channel=config['downsample3_in_channel'], 
            in_length=config['downsample3_in_length'], 
            out_channel=config['downsample3_out_channel'], 
            stride=config['downsample3_stride'], 
            kernel_size=config['downsample3_kernal_size'], 
            padding=config['downsample3_padding'])    
        self.downsample4 = DownSample(
            in_channel=config['downsample4_in_channel'], 
            in_length=config['downsample4_in_length'], 
            out_channel=config['downsample4_out_channel'], 
            stride=config['downsample4_stride'], 
            kernel_size=config['downsample4_kernal_size'], 
            padding=config['downsample4_padding'])
        self.upsample1 = UpSample(
            in_channel=config['upsample1_in_channel'], 
            out_channel=config['upsample1_out_channel'], 
            concat_channel=config['upsample1_concat_channel'], 
            hidden_channel=config['upsample1_hidden_channel'], 
            in_length=config['upsample1_in_length'], 
            kernel_size=config['upsample1_kernel_size'], 
            stride=config['upsample1_stride'], 
            padding=config['upsample1_padding'],
            output_padding=config['upsample1_output_padding'])
        self.upsample2 = UpSample(
            in_channel=config['upsample2_in_channel'], 
            out_channel=config['upsample2_out_channel'], 
            concat_channel=config['upsample2_concat_channel'], 
            hidden_channel=config['upsample2_hidden_channel'], 
            in_length=config['upsample2_in_length'], 
            kernel_size=config['upsample2_kernel_size'], 
            stride=config['upsample2_stride'], 
            padding=config['upsample2_padding'],
            output_padding=config['upsample2_output_padding'])
        self.upsample3 = UpSample(
            in_channel=config['upsample3_in_channel'], 
            out_channel=config['upsample3_out_channel'], 
            concat_channel=config['upsample3_concat_channel'], 
            hidden_channel=config['upsample3_hidden_channel'], 
            in_length=config['upsample3_in_length'], 
            kernel_size=config['upsample3_kernel_size'], 
            stride=config['upsample3_stride'], 
            padding=config['upsample3_padding'],
            output_padding=config['upsample3_output_padding'])
        self.upsample4 = UpSample(
            in_channel=config['upsample4_in_channel'], 
            out_channel=config['upsample4_out_channel'], 
            concat_channel=config['upsample4_concat_channel'], 
            hidden_channel=config['upsample4_hidden_channel'], 
            in_length=config['upsample4_in_length'], 
            kernel_size=config['upsample4_kernel_size'], 
            stride=config['upsample4_stride'], 
            padding=config['upsample4_padding'],
            output_padding=config['upsample4_output_padding'])
        self.conv3 = nn.Conv1d(
            in_channels=config['conv3_in_channel'], 
            out_channels=config['conv3_out_channel'], 
            kernel_size=config['conv3_kernel_size'],
            padding='same',
            bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, labels=None):
        # x: (N, 3, ~6000)
        xx = self.conv1(x)      # xx: (N, 8, ~6000)
        skip4 = self.conv2(xx)    # skip4: (N, 8, ~6000)
        skip3 = self.downsample1(skip4)     # skip3: (N, 11, ~1500)
        skip2 = self.downsample2(skip3)     # skip2: (N, 16, ~375)
        skip1 = self.downsample3(skip2)     # skip1: (N, 22, ~94)
        before_up = self.downsample4(skip1) # before_up: (N, 32, ~24)
        after_up1 = self.upsample1(before_up, skip1)    # after_up1: (N, 22, ~94)
        after_up2 = self.upsample2(after_up1, skip2)    # after_up2: (N, 16, ~375)
        after_up3 = self.upsample3(after_up2, skip3)    # after_up4: (N, 11, ~1500)
        after_up4 = self.upsample4(after_up3, skip4)    # after_up5: (N, 8, ~6000)
        to_softmax = self.conv3(after_up4)

        pred = self.softmax(to_softmax)

        if labels is not None:
            loss = _loss(pred, labels)
            return pred, loss
        else:
            return pred 


class _PhaseNet(nn.Module):
    def __init__(self, in_channels=3):
        super(PhaseNet, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = 7
        self.stride = 4
        self.activation = torch.relu

        self.inc = nn.Conv1d(self.in_channels, 8, 1)
        self.in_bn = nn.BatchNorm1d(8)

        self.conv1 = Conv1dSame(
            in_channels=8,
            out_channels=11,
            kernel_size=self.kernel_size,
            stride=self.stride)
        self.bnd1 = nn.BatchNorm1d(11)

        self.conv2 = Conv1dSame(
            in_channels=11,
            out_channels=16,
            kernel_size=self.kernel_size,
            stride=self.stride)
        self.bnd2 = nn.BatchNorm1d(16)

        self.conv3 = Conv1dSame(
            in_channels=16,
            out_channels=22,
            kernel_size=self.kernel_size,
            stride=self.stride)
        self.bnd3 = nn.BatchNorm1d(22)

        self.conv4 = Conv1dSame(
            in_channels=22,
            out_channels=32,
            kernel_size=self.kernel_size,
            stride=self.stride)
        self.bnd4 = nn.BatchNorm1d(32)

        self.up1 = nn.ConvTranspose1d(
            in_channels=32,
            out_channels=22,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.conv4.padding)
        self.bnu1 = nn.BatchNorm1d(22)

        self.up2 = nn.ConvTranspose1d(
            in_channels=44,
            out_channels=16,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.conv3.padding,
            output_padding=1)
        self.bnu2 = nn.BatchNorm1d(16)

        self.up3 = nn.ConvTranspose1d(
            in_channels=32,
            out_channels=11,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.conv2.padding)
        self.bnu3 = nn.BatchNorm1d(11)

        self.up4 = nn.ConvTranspose1d(
            in_channels=22,
            out_channels=8,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=3)
        self.bnu4 = nn.BatchNorm1d(8)

        self.out = nn.ConvTranspose1d(
            in_channels=16,
            out_channels=3,
            kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, labels=None):
        x_in = self.activation(self.in_bn(self.inc(x)))

        x1 = self.activation(self.bnd1(self.conv1(x_in)))
        x2 = self.activation(self.bnd2(self.conv2(x1)))
        x3 = self.activation(self.bnd3(self.conv3(x2)))
        x4 = self.activation(self.bnd4(self.conv4(x3)))

        x = torch.cat([self.activation(self.bnu1(self.up1(x4))), x3], dim=1)
        x = torch.cat([self.activation(self.bnu2(self.up2(x))), x2], dim=1)
        x = torch.cat([self.activation(self.bnu3(self.up3(x))), x1], dim=1)
        x = torch.cat([self.activation(self.bnu4(self.up4(x))), x_in], dim=1)

        x = self.out(x)
        pred = self.softmax(x)

        return pred


class Encoder(nn.Module):
    def __init__(self, input_channels, filters, kernel_sizes, in_samples):
        super(Encoder, self).__init__()
        convs = []
        pools = []
        self.paddings = []
        for in_channels, out_channels, kernel_size in zip(
            [input_channels] + filters[:-1], filters, kernel_sizes
        ):
            convs.append(nn.Conv1d(
                in_channels, out_channels, kernel_size, padding=kernel_size//2
            ))
            padding = in_samples % 2
            self.paddings.append(padding)
            pools.append(nn.MaxPool1d(2, padding=0))
            in_samples = (in_samples + padding) // 2
        self.convs = nn.ModuleList(convs)
        self.pools = nn.ModuleList(pools)

    def forward(self, x):
        for conv, pool, padding in zip(self.convs, self.pools, self.paddings):
            x = torch.relu(conv(x))
            if padding != 0:
                x = F.pad(x, (0, padding), 'constant', -1e10)
            x = pool(x)

        return x


class Decoder(nn.Module):
    def __init__(self, input_channels, filters, kernel_sizes, out_samples):
        super(Decoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.crops = []
        cur_samples = out_samples
        for i in range(len(filters)):
            padding = cur_samples % 2
            cur_samples = (cur_samples + padding) // 2
            if padding == 1:
                self.crops.append(len(filters) - 1 - i)

        convs = []
        for in_channels, out_channels, kernel_size in zip(
            [input_channels] + filters[:-1], filters, kernel_sizes
        ):
            convs.append(nn.Conv1d(
                in_channels, out_channels, kernel_size, padding=kernel_size//2
            ))

        self.convs = nn.ModuleList(convs)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            x = self.upsample(x)

            if i in self.crops:
                x = x[:, :, :-1]

            x = F.relu(conv(x))
        return x


class SpatialDropout(nn.Module):
    def __init__(self, drop_rate):
        super(SpatialDropout, self).__init__()
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout2d(drop_rate)

    def forward(self, x):
        x = x.unsqueeze(dim=-1)
        x = self.dropout(x)
        x = x.squeeze(dim=-1)
        return x


class ResCNNBlock(nn.Module):
    def __init__(self, channels, kernel_size, drop_rate):
        super(ResCNNBlock, self).__init__()
        self.pre_padding = False
        if kernel_size == 3:
            padding = 1
        else:
            # kernel size == 2
            self.pre_padding = True
            padding = 0

        self.dropout = SpatialDropout(drop_rate)
        self.norm1 = nn.BatchNorm1d(channels, eps=1e-3)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.norm2 = nn.BatchNorm1d(channels, eps=1e-3)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)

    def forward(self, x):
        y = self.norm1(x)
        y = F.relu(y)
        y = self.dropout(y)
        if self.pre_padding:
            y = F.pad(y, (0, 1), 'constant', 0)
        y = self.conv1(y)

        y = self.norm2(y)
        y = F.relu(y)
        y = self.dropout(y)
        if self.pre_padding:
            y = F.pad(y, (0, 1), 'constant', 0)
        y = self.conv2(y)

        return x + y


class ResCNNStack(nn.Module):
    def __init__(self, kernel_sizes, channels, drop_rate):
        super(ResCNNStack, self).__init__()
        members = []
        for kernel_size in kernel_sizes:
            members.append(ResCNNBlock(channels, kernel_size, drop_rate))

        self.members = nn.ModuleList(members)

    def forward(self, x):
        for member in self.members:
            x = member(x)
        return x


class BiLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, drop_rate):
        super(BiLSTMBlock, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.dropout = nn.Dropout(drop_rate)
        self.conv = nn.Conv1d(2 * hidden_size, hidden_size, 1)
        self.norm = nn.BatchNorm1d(hidden_size, eps=1e-3)

    def forward(self, x):
        x = x.permute(2, 0, 1)   # (seq, batch, channels)
        x = self.lstm(x)[0]
        x = self.dropout(x)
        x = x.permute(1, 2, 0)
        x = self.conv(x)
        x = self.norm(x)
        return x


class BiLSTMStack(nn.Module):
    def __init__(self, num_blocks, input_size, drop_rate, hidden_size=16):
        super(BiLSTMStack, self).__init__()
        members = []
        members.append(BiLSTMBlock(
            input_size, hidden_size, drop_rate
        ))
        for _ in range(num_blocks-1):
            members.append(BiLSTMBlock(
                hidden_size, hidden_size, drop_rate
            ))
        self.members = nn.ModuleList(members)

    def forward(self, x):
        for member in self.members:
            x = member(x)
        return x


def _get_rand(a, b, *args):
    return a + (b - a) * torch.rand(*args)


class SeqSelfAttention(nn.Module):
    def __init__(self, input_size, heads=32, attention_width=None, eps=1e-5):
        super(SeqSelfAttention, self).__init__()
        self.attention_width = attention_width
        self.Wx = nn.Parameter(_get_rand(-0.02, 0.02, input_size, heads))
        self.Wt = nn.Parameter(_get_rand(-0.02, 0.02, input_size, heads))
        self.bh = nn.Parameter(torch.zeros(heads))
        self.Wa = nn.Parameter(_get_rand(-0.02, 0.02, heads, 1))
        self.ba = nn.Parameter(torch.zeros(1))
        self.eps = eps

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, seq, channels)
        q = torch.unsqueeze(torch.matmul(x, self.Wt), 2)
        # (batch, seq, 1, channels)
        k = torch.unsqueeze(torch.matmul(x, self.Wx), 1)
        # (batch, 1, seq, channels)
        h = torch.tanh(q + k + self.bh)
        # (batch, seq, seq, channels)
        e = torch.squeeze(torch.matmul(h, self.Wa) + self.ba, -1)
        # (batch, seq, seq)
        e = e - torch.max(e, dim=-1, keepdim=True).values
        e = torch.exp(e)
        if self.attention_width is not None:
            lower_bound = torch.arange(0, e.shape[1], device=e.device) - self.attention_width // 2
            upper_bound = lower_bound + self.attention_width
            indices = torch.unsqueeze(torch.arange(0, e.shape[1], device=e.device), 1)
            mask = torch.logical_and(lower_bound <= indices, indices < upper_bound)
            e = torch.where(mask, e, torch.zeros_like(e))

        a = e / (torch.sum(e, dim=-1, keepdim=True) + self.eps)
        v = torch.matmul(a, x)
        v = v.permute(0, 2, 1)
        return v, a


class LayerNormalization(nn.Module):
    def __init__(self, filters, eps=1e-14):
        super(LayerNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(filters, 1))
        self.beta = nn.Parameter(torch.zeros(filters, 1))
        self.eps = eps

    def forward(self, x):
        mean = torch.mean(x, dim=1, keepdim=True)
        var = torch.var(x, dim=1, keepdim=True)
        norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * norm + self.beta


class FeedForward(nn.Module):
    def __init__(self, io_size, drop_rate, hidden_size=128):
        super(FeedForward, self).__init__()
        self.lin1 = nn.Linear(io_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, io_size)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)
        x = x.permute(0, 2, 1)
        return x


class Transformer(nn.Module):
    def __init__(self, input_size, drop_rate, attention_width=None, eps=1e-5):
        super(Transformer, self).__init__()
        self.attention = SeqSelfAttention(input_size, attention_width=attention_width, eps=eps)
        self.norm1 = LayerNormalization(input_size)
        self.ff = FeedForward(input_size, drop_rate)
        self.norm2 = LayerNormalization(input_size)

    def forward(self, x):
        y, weight = self.attention(x)
        y = x + y
        y = self.norm1(y)
        y2 = self.ff(y)
        y2 = y + y2
        y2 = self.norm2(y)
        return y2, weight


class EQTransformer(nn.Module):
    # reference: the seisbench lib
    def __init__(
        self,
        in_channels=3,
        in_samples=6000,
        classes=2,
        lstm_blocks=3,
        drop_rate=0.1,
        sampling_rate=100
    ):
        super(EQTransformer, self).__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.lstm_blocks = lstm_blocks
        self.drop_rate = drop_rate
        self.in_samples = in_samples
        eps = 1e-5

        self.filters = [8, 16, 16, 32, 32, 64, 64]
        self.kernel_sizes = [11, 9, 7, 7, 5, 5, 3]
        self.res_cnn_kernels = [3, 3, 3, 3, 2, 3, 2]

        self.encoder = Encoder(
            input_channels=self.in_channels,
            filters=self.filters,
            kernel_sizes=self.kernel_sizes,
            in_samples=self.in_samples
        )

        self.res_cnn_stack = ResCNNStack(
            kernel_sizes=self.res_cnn_kernels,
            channels=self.filters[-1],
            drop_rate=self.drop_rate
        )

        self.bi_lstm_stack = BiLSTMStack(
            num_blocks=self.lstm_blocks,
            input_size=self.filters[-1],
            drop_rate=self.drop_rate,
        )

        self.transformer_d0 = Transformer(
            input_size=16,
            drop_rate=self.drop_rate,
            eps=eps
        )
        self.transformer_d = Transformer(
            input_size=16,
            drop_rate=self.drop_rate,
            eps=eps
        )

        self.decoder_d = Decoder(
            input_channels=16,
            filters=self.filters[::-1],
            kernel_sizes=self.kernel_sizes[::-1],
            out_samples=in_samples
        )

        self.conv_d = nn.Conv1d(
            in_channels=self.filters[0],
            out_channels=1,
            kernel_size=11,
            padding=5
        )

        pick_lstms = []
        pick_attentions = []
        pick_decoders = []
        pick_convs = []
        self.dropout = nn.Dropout(drop_rate)

        for _ in range(self.classes):
            lstm = nn.LSTM(16, 16, bidirectional=False)
            pick_lstms.append(lstm)

            attention = SeqSelfAttention(
                input_size=16,
                attention_width=3,
                eps=eps
            )
            pick_attentions.append(attention)

            decoder = Decoder(
                input_channels=16,
                filters=self.filters[::-1],
                kernel_sizes=self.kernel_sizes[::-1],
                out_samples=in_samples
            )
            pick_decoders.append(decoder)

            conv = nn.Conv1d(
                in_channels=self.filters[0],
                out_channels=1,
                kernel_size=11,
                padding=5
            )
            pick_convs.append(conv)

        self.pick_lstms = nn.ModuleList(pick_lstms)
        self.pick_attentions = nn.ModuleList(pick_attentions)
        self.pick_decoders = nn.ModuleList(pick_decoders)
        self.pick_convs = nn.ModuleList(pick_convs)

    def forward(self, x):
        x = self.encoder(x)
        x = self.res_cnn_stack(x)
        x = self.bi_lstm_stack(x)
        x, _ = self.transformer_d0(x)
        x, _ = self.transformer_d(x)

        detection = self.decoder_d(x)
        detection = torch.sigmoid(self.conv_d(detection))

        outputs = [detection]

        for lstm, attention, decoder, conv in zip(
            self.pick_lstms, self.pick_attentions, self.pick_decoders, self.pick_convs
        ):
            px = x.permute(2, 0, 1)   # (seq, batch, channels)
            px = lstm(px)[0]
            px = self.dropout(px)
            px = px.permute(1, 2, 0)   # (batch, channels, seq)
            px, _ = attention(px)
            px = decoder(px)
            pred = torch.sigmoid(conv(px))
            pred = torch.squeeze(pred, dim=1)
            outputs.append(pred)

        return tuple(outputs)  # (detection, P_pred, S_pred)
        # noise = 1 - outputs[2] - outputs[1]
        # return torch.stack([noise, outputs[1], outputs[2]], dim=1)


    