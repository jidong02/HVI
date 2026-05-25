import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips as _lpips_pkg
from loss.vgg_arch import VGGFeatureExtractor, Registry
from loss.loss_utils import *


_reduction_modes = ['none', 'mean', 'sum']

class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)
        
        
        
class EdgeLoss(nn.Module):
    def __init__(self,loss_weight=1.0, reduction='mean'):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1).cuda()

        self.weight = loss_weight
        
    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)
        down        = filtered[:,:,::2,::2]
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4
        filtered    = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = mse_loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss*self.weight


class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculting losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=True,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'mse':
            self.criterion = torch.nn.MSELoss(reduction='mean')
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss




class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True,weight=1.):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        self.weight = weight

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return (1. - map_ssim(img1, img2, window, self.window_size, channel, self.size_average)) * self.weight



# ============== NEW: LAB Color Loss ==============
def rgb_to_lab(rgb):
    """
    可微的 RGB → LAB 转换。
    输入: (B, 3, H, W), [0, 1] 范围
    输出: (B, 3, H, W), L 大致 [0, 100], a/b 大致 [-128, 127]
    """
    eps = 1e-8
    # sRGB → linear RGB
    mask = (rgb > 0.04045).float()
    linear = mask * (((rgb + 0.055) / 1.055).clamp(min=eps)) ** 2.4 + (1 - mask) * (rgb / 12.92)
    
    # linear RGB → XYZ (D65)
    M = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], device=rgb.device, dtype=rgb.dtype)
    # (B, 3, H, W) → (B, H, W, 3)
    linear_hw3 = linear.permute(0, 2, 3, 1)
    xyz = linear_hw3 @ M.T  # (B, H, W, 3)
    
    # XYZ → LAB
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    xyz_n = xyz / torch.tensor([Xn, Yn, Zn], device=rgb.device, dtype=rgb.dtype)
    
    delta = 6.0 / 29.0
    mask_f = (xyz_n > delta ** 3).float()
    f = mask_f * (xyz_n.clamp(min=eps)) ** (1.0/3.0) + (1 - mask_f) * (xyz_n / (3 * delta ** 2) + 4.0/29.0)
    
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    
    lab = torch.stack([L, a, b], dim=-1)              # (B, H, W, 3)
    return lab.permute(0, 3, 1, 2)                    # (B, 3, H, W)


class LABLoss(nn.Module):
    """
    LAB 色彩空间的 L1 损失，重点关注 a/b 色度通道。
    """
    def __init__(self, loss_weight=0.5, ab_weight=2.0):
        super().__init__()
        self.weight = loss_weight
        self.ab_weight = ab_weight  # a/b 通道权重更高
        self.l1 = nn.L1Loss()
    
    def forward(self, pred, target):
        pred = pred.clamp(0, 1)
        target = target.clamp(0, 1)
        lab_pred = rgb_to_lab(pred)
        lab_target = rgb_to_lab(target)
        # L 通道用标准 L1
        L_loss = self.l1(lab_pred[:, 0:1], lab_target[:, 0:1]) / 100.0
        # a/b 通道（色度）用加权 L1，对色偏更敏感
        ab_loss = self.l1(lab_pred[:, 1:], lab_target[:, 1:]) / 128.0
        return self.weight * (L_loss + self.ab_weight * ab_loss)

        # ============== NEW: LPIPS Loss ==============

class LPIPSLoss(nn.Module):
    """
    LPIPS 感知损失，AlexNet backbone (与 GuidedHybSensUIR TCSVT 2025 一致)
    
    输入: pred, target ∈ [0, 1], shape (B, 3, H, W)
    内部转换到 [-1, 1] 喂给 LPIPS 网络
    LPIPS 网络参数冻结，不会被训练
    """
    def __init__(self, loss_weight=0.5, net='alex'):
        super().__init__()
        self.weight = loss_weight
        self.lpips_net = _lpips_pkg.LPIPS(net=net, verbose=False)
        for param in self.lpips_net.parameters():
            param.requires_grad = False
        self.lpips_net.eval()
    
    def forward(self, pred, target):
        pred = pred.clamp(0, 1) * 2.0 - 1.0
        target = target.clamp(0, 1) * 2.0 - 1.0
        return self.weight * self.lpips_net(pred, target).mean()