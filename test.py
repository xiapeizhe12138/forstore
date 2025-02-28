import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

def calculate_ssim(pred, gt, data_range=1.0, window_size=11, sigma=1.5):
    """
    计算两个张量之间的结构相似性指数（SSIM）
    
    参数：
        pred: 预测图像张量 (C,H,W) 或 (B,C,H,W)
        gt: 真实图像张量 (shape必须与pred相同)
        data_range: 数据的取值范围（ToTensor转换的应为1.0）
        window_size: 高斯滤波器尺寸
        sigma: 高斯滤波器标准差
        
    返回：
        ssim_value: 标量值
    """
    # 维度校验
    assert pred.shape == gt.shape, f"形状不一致：pred {pred.shape}, gt {gt.shape}"
    
    # 自动扩展维度（支持单张和多张计算）
    if len(pred.shape) == 3:
        pred = pred.unsqueeze(0)  # (C,H,W) -> (1,C,H,W)
        gt = gt.unsqueeze(0)
    
    # 创建高斯窗口
    device = pred.device
    channel = pred.size(1)
    gaussian = _create_gaussian_window(window_size, sigma, channel).to(device)
    
    # 计算SSIM各分量
    mu_pred = F.conv2d(pred, gaussian, padding=window_size//2, groups=channel)
    mu_gt = F.conv2d(gt, gaussian, padding=window_size//2, groups=channel)
    
    mu_pred_sq = mu_pred.pow(2)
    mu_gt_sq = mu_gt.pow(2)
    mu_pred_gt = mu_pred * mu_gt
    
    sigma_pred_sq = F.conv2d(pred*pred, gaussian, padding=window_size//2, groups=channel) - mu_pred_sq
    sigma_gt_sq = F.conv2d(gt*gt, gaussian, padding=window_size//2, groups=channel) - mu_gt_sq
    sigma_pred_gt = F.conv2d(pred*gt, gaussian, padding=window_size//2, groups=channel) - mu_pred_gt
    
    # SSIM计算
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    ssim_map = ((2*mu_pred_gt + C1) * (2*sigma_pred_gt + C2)) / \
               ((mu_pred_sq + mu_gt_sq + C1) * (sigma_pred_sq + sigma_gt_sq + C2))
    
    return ssim_map.mean().item()

def _create_gaussian_window(window_size, sigma, channel):
    """创建高斯卷积核"""
    x = torch.arange(window_size).float() - window_size//2
    gauss = torch.exp(-x.pow(2)/(2*sigma**2))
    gauss /= gauss.sum()
    
    window = gauss.ger(gauss)  # 外积生成二维窗口
    window = window / window.sum()
    window = window.view(1, 1, window_size, window_size)
    return window.repeat(channel, 1, 1, 1)  # (C,1,H,W)

# 使用示例
if __name__ == "__main__":
    # 加载两张图像
    x1 = "path/to/image1.jpg"
    x2 = "path/to/image2.jpg"
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.convert('RGB') if isinstance(x, Image.Image) else x)
    ])
    
    pred = transform(Image.open(x1))
    gt = transform(Image.open(x2))
    
    # 计算SSIM
    ssim_value = calculate_ssim(pred, gt)
    print(f"SSIM: {ssim_value:.4f}")