import torch.nn
from asteroid.masknn import norms
import torchaudio
from torch import nn
from asteroid.utils.torch_utils import pad_x_to_y


class Conv1DBlock(nn.Module):

    def __init__(
        self,
        in_chan,
        hid_chan,
        kernel_size,
        dilation,
        norm_type="bN",
    ):
        super(Conv1DBlock, self).__init__()
        conv_norm = norms.get(norm_type)
        in_conv1d = nn.Conv1d(in_chan, hid_chan, 1)
        depth_conv1d = nn.Conv1d(
            hid_chan, hid_chan, kernel_size, dilation=dilation, groups=hid_chan#, padding=dilation
        )
        self.shared_block = nn.Sequential(
            in_conv1d,
            nn.PReLU(),
            conv_norm(hid_chan),
            depth_conv1d,
            nn.PReLU(),
            conv_norm(hid_chan),
        )
        self.res_conv = nn.Conv1d(hid_chan, in_chan, 1)

    def forward(self, x):
        r"""Input shape $(batch, feats, seq)$."""
        shared_out = self.shared_block(x)
        res_out = self.res_conv(shared_out)
        return res_out + x[..., :res_out.shape[-1]]


class NormSpecAugment(torch.nn.Module):
    def __init__(self, in_chan, norm_type="bN", n_t_masks=4, n_f_masks=2, t_mask_len=4, f_mask_len=10):
        super(NormSpecAugment, self).__init__()
        self.norm_type = norm_type
        self.norm = norms.get(norm_type)(in_chan)
        specaugm = []
        for _ in range(n_t_masks):
            specaugm.append(torchaudio.transforms.TimeMasking(t_mask_len))
        for _ in range(n_f_masks):
            specaugm.append(torchaudio.transforms.FrequencyMasking(f_mask_len))
        self.specaugm = torch.nn.Sequential(*specaugm)

    def forward(self, inputs):
        normed = self.norm(inputs)
        if self.training:
           normed = self.specaugm(normed)
           mask_dropped = (normed == 0.0)
           if self.norm_type == "bN":
                normed[mask_dropped] = (normed + self.norm.bias.reshape(1, -1, 1))[mask_dropped]
           else:
                normed[mask_dropped] = (normed + self.norm.beta.reshape(1, -1, 1))[mask_dropped]

        return normed


class TCN(torch.nn.Module):
    def __init__(self, feats_config, n_out=1, bn_chan=64, hid_chan=128,
                 n_repeats=3, n_blocks=5,
                 norm_type="cLN", ksz=3, freeze_bn=False,  n_t_masks=4, n_f_masks=2, t_mask_len=4, f_mask_len=10,
                 use_input_mix=False):
        super(TCN, self).__init__()
        #in_chan = feats_config.n_mels
        in_chan = feats_config.n_mels * (n_out + int(use_input_mix))
        self.in_chan = in_chan
        self.n_out = n_out
        self.n_repeats = n_repeats
        self.n_blocks = n_blocks
        self.norm_type = norm_type
        self.ksz = ksz
        self.freeze_bn = freeze_bn
        self.mels = torchaudio.transforms.MelSpectrogram(**feats_config)
        self.use_input_mix = use_input_mix

        self.bn = torch.nn.Sequential(
                NormSpecAugment(in_chan, norm_type, n_t_masks, n_f_masks, t_mask_len, f_mask_len),
                torch.nn.Conv1d(in_chan, bn_chan, 1, 1))

        self.net = []
        for r in range(n_repeats):
            for b in range(n_blocks):
                self.net.append(Conv1DBlock(bn_chan, hid_chan, ksz, dilation=2**b))

        self.net = torch.nn.Sequential(*self.net)
        self.out = torch.nn.Sequential(torch.nn.PReLU(),
                                       norms.get(norm_type)(bn_chan),
                                       torch.nn.Conv1d(bn_chan, n_out, 1, 1))

    def forward(self, inputs):
        bsz, spk, _ = inputs.shape
        inputs_flat = inputs.reshape(bsz * spk, -1)
        feats = 10 * torch.log10(self.mels(inputs_flat) + 1e-8)
        _, n_mels, _ = feats.shape
        if self.n_out == 2:
            #assert bsz == 2
            feats = feats.reshape(bsz, spk * n_mels, -1)
        
        logits = self.out(self.net(self.bn(feats)))

        if self.n_out == 1:
            logits = logits.reshape(bsz, spk, -1)

        return logits

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(TCN, self).train(mode)
        if self.freeze_bn:
            print("Freezing Mean/Var of BatchNorm2D.")
            if self.freeze_bn:
                print("Freezing Weight/Bias of BatchNorm2D.")
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.freeze_bn:
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False




