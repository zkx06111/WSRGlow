import librosa
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pdb
import torch.nn as nn


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


class WaveGlowLoss(torch.nn.Module):
    def __init__(self, sigma=1.0):
        super(WaveGlowLoss, self).__init__()
        self.sigma = sigma

    def forward(self, model_output):
        # print(f'model_output = {model_output}')
        z, log_s_list, log_det_W_list = model_output
        for i, log_s in enumerate(log_s_list):
            if i == 0:
                log_s_total = torch.sum(log_s)
                log_det_W_total = log_det_W_list[i]
            else:
                log_s_total = log_s_total + torch.sum(log_s)
                log_det_W_total += log_det_W_list[i]

        loss = torch.sum(z * z) / (2 * self.sigma * self.sigma) - \
            log_s_total - log_det_W_total
        return loss / (z.size(0) * z.size(1) * z.size(2))


class Invertible1x1Conv(torch.nn.Module):
    """
    The layer outputs both the convolution, and the log determinant
    of its weight matrix.  If reverse=True it does convolution with
    inverse
    """

    def __init__(self, c):
        super(Invertible1x1Conv, self).__init__()
        self.conv = torch.nn.Conv1d(c, c, kernel_size=1, stride=1, padding=0,
                                    bias=False)

        # Sample a random orthonormal matrix to initialize weights
        W = torch.empty(c, c)
        torch.nn.init.orthogonal_(W)
        while torch.isnan(torch.logdet(W)):
            torch.nn.init.orthogonal_(W)
        # print(f'W={W}')
        # print(f'detW={torch.det(W)}', flush=True)
        # print(f'WTW={torch.mm(W.t(), W)}', flush=True)

        # Ensure determinant is 1.0 not -1.0
        if torch.det(W) < 0:
            W[:, 0] = -1 * W[:, 0]
        W = W.view(c, c, 1)
        self.conv.weight.data = W

    def forward(self, z, reverse=False):
        # shape
        batch_size, group_size, n_of_groups = z.size()

        W = self.conv.weight.squeeze()

        if reverse:
            if not hasattr(self, 'W_inverse'):
                # Reverse computation
                W_inverse = W.float().inverse()
                W_inverse = Variable(W_inverse[..., None])
                if z.type() == 'torch.cuda.HalfTensor':
                    W_inverse = W_inverse.half()
                self.W_inverse = W_inverse
            z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
            return z
        else:
            # Forward computation
            log_det_W = batch_size * n_of_groups * torch.logdet(W)
            z = self.conv(z)
            return z, log_det_W


class WN(torch.nn.Module):
    """
    This is the WaveNet like layer for the affine coupling.  The primary difference
    from WaveNet is the convolutions need not be causal.  There is also no dilation
    size reset.  The dilation only doubles on each layer
    """

    def __init__(self, n_in_channels, n_mel_channels, n_layers, n_channels,
                 kernel_size):
        super(WN, self).__init__()
        assert (kernel_size % 2 == 1)
        assert (n_channels % 2 == 0)
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()

        start = torch.nn.Conv1d(n_in_channels, n_channels, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        end = torch.nn.Conv1d(n_channels, 2 * n_in_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end
        #print(f'n_channels = {n_channels}, n_channels = {n_layers}')
        cond_layer = torch.nn.Conv1d(
            n_mel_channels, 2 * n_channels * n_layers, 1)
        self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')

        for i in range(n_layers):
            dilation = 2 ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(n_channels, 2 * n_channels, kernel_size,
                                       dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * n_channels
            else:
                res_skip_channels = n_channels
            res_skip_layer = torch.nn.Conv1d(n_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(
                res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, forward_input):
        audio, spect = forward_input
        # print(self.start, flush=True)
        audio = self.start(audio)
        output = torch.zeros_like(audio)
        n_channels_tensor = torch.IntTensor([self.n_channels])

        spect = self.cond_layer(spect)

        #print(f'n_channels * 2 = {self.n_channels * 2}')
        for i in range(self.n_layers):
            spect_offset = i * 2 * self.n_channels
            # print(f'i = {i}')
            # print(f'shape1 = {self.in_layers[i](audio).shape}')
            # print(f'shape2 = {spect[:, spect_offset:spect_offset + 2 * self.n_channels, :].shape}', flush=True)
            acts = fused_add_tanh_sigmoid_multiply(
                self.in_layers[i](audio),
                spect[:, spect_offset:spect_offset + 2 * self.n_channels, :],
                n_channels_tensor)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                audio = audio + res_skip_acts[:, :self.n_channels, :]
                output = output + res_skip_acts[:, self.n_channels:, :]
            else:
                output = output + res_skip_acts
            # print(f'WN: i={i}, output={output}')

        return self.end(output)


class MuLawEmbedding(torch.nn.Module):
    def __init__(self, mu, embed_num, hidden_dim):
        super(MuLawEmbedding, self).__init__()
        self.mu = mu
        self.embed_num = embed_num
        self.embed = Embedding(num_embeddings=embed_num,
                               embedding_dim=hidden_dim)

    def forward(self, index):
        # forward_input: batch x (time / 2)
        # print(index.device, flush=True)
        index = index.sign()
        index = index * torch.log(1 + self.mu *
                                  torch.abs(index)) / np.log(1 + self.mu)
        # (-1, 1)
        embed_num = self.embed_num
        index = ((index + 1) * (self.embed_num // 2)).floor().long()
        index = (index < 0) * 0 + (index >= 0) * (index < embed_num) * \
            index + (index >= embed_num) * (embed_num - 1)
        # [0, 256)
        assert torch.min(index).item() >= 0 and torch.max(
            index).item() < embed_num
        index = index.cuda()
        return self.embed(index)


class AngleEmbedding(torch.nn.Module):
    def __init__(self, embed_num, hidden_dim):
        super(AngleEmbedding, self).__init__()
        self.embed_num = embed_num
        self.embed = Embedding(num_embeddings=embed_num,
                               embedding_dim=hidden_dim)

    def forward(self, index):
        embed_num = self.embed_num
        index = ((index / np.pi + 1) * (embed_num // 2)).floor().long()
        index = (index < 0) * 0 + (index >= 0) * (index < embed_num) * \
            index + (index >= embed_num) * (embed_num - 1)
        assert torch.min(index).item() >= 0 and torch.max(
            index).item() < embed_num
        index = index.cuda()
        return self.embed(index)


class WaveGlowMelHF(torch.nn.Module):
    def __init__(self, mu, embed_num, embed_dim, n_flows, n_group, n_early_every,
                 n_early_size, WN_config):
        super(WaveGlowMelHF, self).__init__()
        self.muembed = MuLawEmbedding(mu, embed_num, embed_dim)
        #print(f'embed_dim * n_group = {embed_dim * n_group}', flush=True)

        assert (n_group % 2 == 0)
        self.n_flows = n_flows
        self.n_group = n_group
        self.n_early_every = n_early_every
        self.n_early_size = n_early_size
        self.WN = torch.nn.ModuleList()
        self.convinv = torch.nn.ModuleList()
        self.phase_embedding = AngleEmbedding(embed_num=120, hidden_dim=50)

        n_half = n_group

        # Set up layers with the right sizes based on how many dimensions
        # have been output already
        n_remaining_channels = 2 * n_group
        #print(f'n_half = {n_half}', flush=True)
        for k in range(n_flows):
            if k % self.n_early_every == 0 and k > 0:
                n_half = n_half - int(self.n_early_size / 2)
                n_remaining_channels = n_remaining_channels - self.n_early_size
            self.convinv.append(Invertible1x1Conv(n_remaining_channels))
            self.WN.append(WN(n_half, embed_dim * n_group + 50 * (n_group + 1) + n_group + 1, **WN_config))  # ??
        self.n_remaining_channels = n_remaining_channels  # Useful during inference

    def forward(self, lr, hr):
        """
        audio: batch x time
        """
        #print(lr.shape)
        T = lr.shape[1]
        lr = lr

        n_group = self.n_group

        Ds = [librosa.stft(x, n_fft=n_group * 2, hop_length=n_group) for x in lr.cpu().numpy()]
        spect = torch.Tensor([np.abs(d) for d in Ds]).cuda()  # (B, n_group + 1, T / 2 / n_group)
        phase = torch.Tensor([np.angle(d) for d in Ds]).cuda()  # (B, n_group + 1, T / 2 / n_group)
        phaseemb = self.phase_embedding(phase.permute(0, 2, 1))  # (B, n_group + 1, T / 2 / n_group, H)
        #print(f'spect.shape = {spect.shape}')
        phaseemb = phaseemb.reshape(phaseemb.shape[0], phaseemb.shape[1], -1).permute(0, 2, 1)
        # (B, (n_group + 1) * H, T / 2 / n_group)
        #print(f'phaseemb.shape = {phaseemb.shape}')

        #  use mu-law embedding to depict low res audio
        lr = self.muembed(lr).permute(0, 2, 1)  # (B, H, T / 2)


        #print(f'lr_shape after muembed = {lr.shape}')
        lr = lr.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        # (B, T / 2 / n_group, H, n_group)
        lr = lr.contiguous().view(lr.size(0), lr.size(1), -1).permute(0, 2, 1)
        #print(f'lr.shape = {lr.shape}', lr.shape, flush=True)
        # (B, H x n_group, T / 2 / n_group)

        min_dim2 = min([lr.shape[2], spect.shape[2], phaseemb.shape[2]])
        lr = lr[:, :, :min_dim2]
        spect = spect[:, :, :min_dim2]
        phaseemb = phaseemb[:, :, :min_dim2]

        lr = torch.cat((lr, spect, phaseemb), dim=1)
        # H1 = embed_dim for phase
        # H2 = embed_dim for waveform
        # (B, H1 x (n_group + 1) + H2 x n_group + n_group + 1, T / 2 / n_group)
        #print(f'lr.shape = {lr.shape}', flush=True)

        audio = hr.reshape(hr.shape[0], -1)  # (B, T)
        audio = audio.unfold(1, self.n_group * 2, self.n_group * 2).permute(0, 2, 1)
        #print(f'lr.shape = {lr.shape}, audio.shape = {audio.shape}', flush=True)
        #print(self.n_group, audio.shape, flush=True)
        # batch x (n_group * 2) x (time / n_group / 2)
        #print(f'audio.shape = {audio.shape}', flush=True)
        output_audio = []
        log_s_list = []
        log_det_W_list = []

        #print(f'lr.shape = {lr.shape}')
        #print(f'audio.shape = {audio.shape}')

        for k in range(self.n_flows):
            #print(f'k = {k}', flush=True)
            if k % self.n_early_every == 0 and k > 0:
                output_audio.append(audio[:, :self.n_early_size, :])
                audio = audio[:, self.n_early_size:, :]

            audio, log_det_W = self.convinv[k](audio)
            log_det_W_list.append(log_det_W)

            n_half = int(audio.size(1) / 2)
            audio_0 = audio[:, :n_half, :]
            audio_1 = audio[:, n_half:, :]

            output = self.WN[k]((audio_0, lr))
            log_s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = torch.exp(log_s) * audio_1 + b
            log_s_list.append(log_s)

            audio = torch.cat([audio_0, audio_1], 1)

        output_audio.append(audio)
        return torch.cat(output_audio, 1), log_s_list, log_det_W_list

    def infer(self, lr, sigma=1.0):
        n_group = self.n_group
        Ds = [librosa.stft(x, n_fft=n_group * 2, hop_length=n_group) for x in lr.cpu().numpy()]
        spect = torch.Tensor([np.abs(d) for d in Ds]).cuda()  # (B, n_group + 1, T / 2 / n_group)
        phase = torch.Tensor([np.angle(d) for d in Ds]).cuda()  # (B, n_group + 1, T / 2 / n_group)
        phaseemb = self.phase_embedding(phase.permute(0, 2, 1))  # (B, n_group + 1, T / 2 / n_group, H)
        phaseemb = phaseemb.reshape(phaseemb.shape[0], phaseemb.shape[1], -1).permute(0, 2, 1)
        lr = self.muembed(lr).permute(0, 2, 1)  # (B, H, T / 2)
        lr = lr.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        lr = lr.contiguous().view(lr.size(0), lr.size(1), -1).permute(0, 2, 1)

        min_dim2 = min([lr.shape[2], spect.shape[2], phaseemb.shape[2]])
        lr = lr[:, :, :min_dim2]
        spect = spect[:, :, :min_dim2]
        phaseemb = phaseemb[:, :, :min_dim2]
        lr = torch.cat((lr, spect, phaseemb), dim=1)

        if lr.type() == 'torch.cuda.HalfTensor':
            audio = torch.cuda.HalfTensor(lr.size(0),
                                          self.n_remaining_channels,
                                          lr.size(2)).normal_()
        else:
            audio = torch.cuda.FloatTensor(lr.size(0),
                                           self.n_remaining_channels,
                                           lr.size(2)).normal_()

        # print(f'sigma = {sigma}')
        # print(f'audio.shape = {audio.shape}', flush=True)
        audio = torch.autograd.Variable(sigma * audio)

        for k in reversed(range(self.n_flows)):
            n_half = int(audio.size(1) / 2)
            audio_0 = audio[:, :n_half, :]
            audio_1 = audio[:, n_half:, :]

            output = self.WN[k]((audio_0, lr))

            s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = (audio_1 - b) / torch.exp(s)
            audio = torch.cat([audio_0, audio_1], 1)

            audio = self.convinv[k](audio, reverse=True)

            if k % self.n_early_every == 0 and k > 0:
                if lr.type() == 'torch.cuda.HalfTensor':
                    z = torch.cuda.HalfTensor(lr.size(0), self.n_early_size, lr.size(2)).normal_()
                else:
                    z = torch.cuda.FloatTensor(lr.size(0), self.n_early_size, lr.size(2)).normal_()
                audio = torch.cat((sigma * z, audio), 1)

        audio = audio.permute(0, 2, 1).contiguous().view(audio.size(0), -1)
        audio = audio.reshape(audio.shape[0], 1, -1)
        #print(audio.shape, lr__.shape)
        return audio

    @staticmethod
    def remove_weightnorm(model):
        waveglow = model
        for WN in waveglow.WN:
            WN.start = torch.nn.utils.remove_weight_norm(WN.start)
            WN.in_layers = remove(WN.in_layers)
            WN.cond_layer = torch.nn.utils.remove_weight_norm(WN.cond_layer)
            WN.res_skip_layers = remove(WN.res_skip_layers)
        return waveglow


def remove(conv_list):
    new_conv_list = torch.nn.ModuleList()
    for old_conv in conv_list:
        old_conv = torch.nn.utils.remove_weight_norm(old_conv)
        new_conv_list.append(old_conv)
    return new_conv_list
