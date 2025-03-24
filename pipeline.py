import torch
import cv2
import numpy as np
import math
from pathlib import Path
from bsr.degradations import circular_lowpass_kernel, random_mixed_kernels
from bsr.utils import img2tensor, tensor2img
from bsr.utils.img_process_util import filter2D
from bsr.transforms import augment

class RealESRGANProcessor:
    def __init__(self, opt, device='cuda'):
        self.opt = opt
        self.device = torch.device(device)
        
        # 初始化退化参数
        self._init_degradation_params()
        self.jpeger = DiffJPEG(differentiable=False).to(self.device)
        
    def _init_degradation_params(self):
        """从配置中初始化退化参数"""
        # 第一次退化参数
        self.kernel_list = self.opt['kernel_list']
        self.kernel_prob = self.opt['kernel_prob']
        self.blur_sigma = self.opt['blur_sigma']
        self.sinc_prob = self.opt['sinc_prob']
        
        # 第二次退化参数
        self.kernel_list2 = self.opt['kernel_list2']
        self.kernel_prob2 = self.opt['kernel_prob2']
        self.blur_sigma2 = self.opt['blur_sigma2']
        self.sinc_prob2 = self.opt['sinc_prob2']
        
        # 公共参数
        self.kernel_range = [2*v+1 for v in range(3, 11)]  # 7-21

    def _generate_kernel(self, stage=1):
        """生成退化核（支持两阶段退化）"""
        kernel_size = random.choice(self.kernel_range)
        
        if stage == 1:
            use_sinc = np.random.uniform() < self.sinc_prob
            params = (self.kernel_list, self.kernel_prob, self.blur_sigma)
        else:
            use_sinc = np.random.uniform() < self.sinc_prob2
            params = (self.kernel_list2, self.kernel_prob2, self.blur_sigma2)
            
        if use_sinc:
            omega_c = np.random.uniform(np.pi/5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size)
        else:
            kernel = random_mixed_kernels(
                kernel_types=params[0],
                kernel_prob=params[1],
                kernel_size=kernel_size,
                sigma_x=params[2],
                sigma_y=params[2],
                theta= np.random.uniform(-math.pi, math.pi)
            )
            
        # 填充到21x21
        pad_size = (21 - kernel_size) // 2
        return np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

    def process(self, img_path, save_path=None):
        """处理单张图像"""
        # 读取并预处理图像
        img = cv2.imread(img_path).astype(np.float32) / 255.
        img = augment(img, self.opt['use_hflip'], self.opt['use_rot'])
        img_tensor = img2tensor(img, bgr2rgb=True, float32=True).unsqueeze(0).to(self.device)
        
        # 生成退化核
        kernel1 = self._generate_kernel(stage=1)
        kernel2 = self._generate_kernel(stage=2)
        sinc_kernel = self._generate_sinc_kernel()
        
        # 转换为Tensor
        kernel1_t = torch.FloatTensor(kernel1).to(self.device)
        kernel2_t = torch.FloatTensor(kernel2).to(self.device)
        sinc_t = torch.FloatTensor(sinc_kernel).to(self.device)
        
        # 执行退化
        lr_tensor = self._degrade_process(img_tensor, kernel1_t, kernel2_t, sinc_t)
        
        # 转换回numpy图像
        lr_img = tensor2img(lr_tensor, rgb2bgr=True, out_type=np.uint8)
        
        if save_path:
            cv2.imwrite(save_path, lr_img)
            
        return lr_img

    def _degrade_process(self, img_tensor, kernel1, kernel2, sinc_kernel):
        """执行完整的退化流程"""
        # 第一次退化
        out = filter2D(img_tensor, kernel1)
        out = self._random_resize(out, stage=1)
        out = self._add_noise(out, stage=1)
        out = self._jpeg_compress(out, stage=1)
        
        # 第二次退化
        if np.random.uniform() < self.opt['second_blur_prob']:
            out = filter2D(out, kernel2)
            
        out = self._random_resize(out, stage=2)
        out = self._add_noise(out, stage=2)
        out = self._final_process(out, sinc_kernel)
        
        return out

    def _random_resize(self, img, stage):
        """随机缩放处理"""
        resize_range = self.opt[f'resize_range{"" if stage==1 else "2"}']
        scale = np.random.uniform(resize_range[0], resize_range[1])
        mode = random.choice(["area", "bilinear", "bicubic"])
        return F.interpolate(img, scale_factor=scale, mode=mode)

    def _add_noise(self, img, stage):
        """添加噪声"""
        if np.random.uniform() < self.opt[f'gaussian_noise_prob{"" if stage==1 else "2"}']:
            sigma = np.random.uniform(*self.opt[f'noise_range{"" if stage==1 else "2"}'])
            return random_add_gaussian_noise_pt(img, sigma_range=[sigma, sigma])
        else:
            scale = np.random.uniform(*self.opt[f'poisson_scale_range{"" if stage==1 else "2"}'])
            return random_add_poisson_noise_pt(img, scale_range=[scale, scale])

    def _jpeg_compress(self, img, stage):
        """JPEG压缩"""
        quality = np.random.uniform(*self.opt[f'jpeg_range{"" if stage==1 else "2"}'])
        return self.jpeger(torch.clamp(img, 0, 1), quality=quality)

    def _generate_sinc_kernel(self):
        """生成最终sinc核"""
        if np.random.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi/3, np.pi)
            return circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
        else:
            return np.eye(21)  # 脉冲核

class DiffJPEG:
    """简化的JPEG压缩模拟器(具体实现需参考原始代码)"""
    def __init__(self, differentiable=False):
        pass
    
    def __call__(self, img, quality):
        return img  # 实际应包含DCT变换和量化步骤

# 使用示例
if __name__ == "__main__":
    # 示例配置（需与训练配置一致）
    opt = {
        'kernel_list': ['iso', 'aniso'],
        'kernel_prob': [0.5, 0.5],
        'blur_sigma': [0.2, 3.0],
        'sinc_prob': 0.1,
        # ...其他参数...
    }
    
    processor = RealESRGANProcessor(opt, device='cuda')
    lr_img = processor.process(
        img_path='input.png',
        save_path='output_lr.png'
    )