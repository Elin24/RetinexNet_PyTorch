import torch
import torch.nn as nn
import torch.nn.functional as F


def gradient(input_tensor, direction, Ichannel):
    h, w = input_tensor.size()[2], input_tensor.size()[3]

    smooth_kernel_x = torch.reshape(torch.Tensor([[0., 0.], [-1., 1.]]), (1, 1, 2, 2))
    smooth_kernel_y = smooth_kernel_x.permute(0, 1, 3, 2)

    assert direction in ['x', 'y']
    device = input_tensor.get_device()
    if direction == "x":
        kernel = smooth_kernel_x.expand(Ichannel, 1, 2, 2).to(device)
    else:
        kernel = smooth_kernel_y.expand(Ichannel, 1, 2, 2).to(device)

    out = F.conv2d(input_tensor, kernel, padding=(1, 1), groups=Ichannel)
    out = torch.abs(out[:, :, 0:h, 0:w])

    return out


def ave_gradient(input_tensor, direction, Ichannel):
    return (F.avg_pool2d(gradient(input_tensor, direction, Ichannel), 3, stride=1, padding=1))


def smooth(input_l, input_r, Ichannel):
    if Ichannel == 1:
        device = input_l.get_device()
        rgb_weights = torch.Tensor([0.2989, 0.5870, 0.1140]).reshape(1, -1, 1, 1).to(device)
        input_r = torch.sum(input_r * rgb_weights, dim=1).unsqueeze(1)

    return torch.mean(
        gradient(input_l, 'x', Ichannel) * torch.exp(-10 * ave_gradient(input_r, 'x', Ichannel)) +
        gradient(input_l, 'y', Ichannel) * torch.exp(-10 * ave_gradient(input_r, 'y', Ichannel))
    )


class DecomLoss(nn.Module):

    def __init__(self, Ichannel):
        super(DecomLoss, self).__init__()
        assert Ichannel in (1, 3), "Illumation channel is not available"
        self.Ichannel = Ichannel

    def forward(self, r_low, l_low, r_high, l_high, input_low, input_high):
        l_low_3 = l_low if self.Ichannel == 3 else torch.cat((l_low, l_low, l_low), 1)
        l_high_3 = l_high if self.Ichannel == 3 else torch.cat((l_high, l_high, l_high), 1)

        recon_loss_low = torch.mean(torch.abs(r_low * l_low_3 - input_low))
        recon_loss_high = torch.mean(torch.abs(r_high * l_high_3 - input_high))
        recon_loss_mutal_low = torch.mean(torch.abs(r_high * l_low_3 - input_low))
        recon_loss_mutal_high = torch.mean(torch.abs(r_low * l_high_3 - input_high))
        equal_r_loss = torch.mean(torch.abs(r_low - r_high))
        
        ismooth_loss_low = smooth(l_low, r_low, self.Ichannel)
        ismooth_loss_high = smooth(l_high, r_high, self.Ichannel)

        return \
            recon_loss_low + recon_loss_high +\
            0.001*recon_loss_mutal_low + 0.001*recon_loss_mutal_high + \
            0.1*ismooth_loss_low + 0.1*ismooth_loss_high + \
            0.01*equal_r_loss


class RelightLoss(nn.Module):

    def __init__(self, Ichannel):
        super(RelightLoss, self).__init__()
        assert Ichannel in (1, 3), "Illumation channel is not available"
        self.Ichannel = Ichannel

    def forward(self, l_delta, r_low, input_high):
        l_delta_3 = l_delta if self.Ichannel == 3 else torch.cat((l_delta, l_delta, l_delta), 1)
        relight_loss = torch.mean(torch.abs(r_low * l_delta_3 - input_high))
        ismooth_loss_delta = smooth(l_delta, r_low, self.Ichannel)
        return relight_loss + ismooth_loss_delta# * (4 - self.Ichannel)


if __name__ == '__main__':
    tensor = torch.rand(1, 300, 400, 1)
    out_data = smooth(tensor, torch.rand(1, 300, 400, 3), 1)
    print(out_data)
