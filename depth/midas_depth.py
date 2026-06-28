# depth/midas_depth.py
"""
تخمین عمق با مدل MiDAS (Monocular Depth Estimation)
منبع: https://github.com/isl-org/MiDaS
"""

import cv2
import torch
import numpy as np
from typing import Optional, Tuple, List


class MiDASDepthEstimator:
    """
    تخمین عمق از یک تصویر معمولی با MiDAS
    نیازی به سخت‌افزار خاصی ندارد
    """
    
    def __init__(self, model_type: str = "MiDaS_small"):
        """
        Args:
            model_type: "MiDaS_small" (سریع، ~5-10 FPS روی CPU)
                        "DPT_Large" (دقیق، نیاز به GPU)
        """
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"[MiDAS] Loading model on {self.device}...")
        
        try:
            self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
            self.midas.to(self.device)
            self.midas.eval()
            
            # transform برای پیش‌پردازش
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            
            if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform
            
            self.is_available = True
            print(f"[MiDAS] Model loaded successfully")
            
        except Exception as e:
            print(f"[MiDAS] Failed to load model: {e}")
            print(f"[MiDAS] Continuing without depth estimation...")
            self.is_available = False
    
    def estimate_depth_map(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        تخمین نقشه عمق برای کل تصویر
        
        Returns:
            depth_map: آرایه 2D با مقادیر 0-255 (روشن = نزدیک، تیره = دور)
        """
        if not self.is_available:
            return None
        
        # پیش‌پردازش
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (384, 384))  # MiDAS input size
        input_batch = self.transform(img).to(self.device)
        
        # پیش‌بینی
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        
        # نرمالایز به 0-255
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = depth_normalized.astype(np.uint8)
        
        return depth_normalized
    
    def get_face_distance(self, frame: np.ndarray, face_bbox: Tuple) -> Optional[float]:
        """
        تخمین فاصله صورت از دوربین
        
        Returns:
            فاصله نسبی (0-100)، هرچه بزرگتر = نزدیک‌تر
        """
        depth_map = self.estimate_depth_map(frame)
        
        if depth_map is None:
            return None
        
        x, y, w, h = face_bbox
        # اطمینان از محدوده معتبر
        x = max(0, x)
        y = max(0, y)
        w = min(frame.shape[1] - x, w)
        h = min(frame.shape[0] - y, h)
        
        # میانگین عمق در ناحیه صورت (با وزن مرکز)
        face_region = depth_map[y:y+h, x:x+w]
        
        # وزن دهی مرکزی (نقاط نزدیک به مرکز صورت اهمیت بیشتری دارند)
        h_center, w_center = face_region.shape
        y_center, x_center = h_center // 2, w_center // 2
        
        # ایجاد وزن گاوسی
        y_grid, x_grid = np.ogrid[:h_center, :w_center]
        y_dist = (y_grid - y_center) ** 2 / (2 * (h_center/4) ** 2)
        x_dist = (x_grid - x_center) ** 2 / (2 * (w_center/4) ** 2)
        gaussian_weights = np.exp(-(y_dist + x_dist))
        
        # میانگین وزنی
        weighted_depth = np.sum(face_region * gaussian_weights) / np.sum(gaussian_weights)
        
        return float(weighted_depth)
    
    def get_relative_depth(self, depth_map: np.ndarray) -> float:
        """دریافت عمق نسبی (برای مقایسه بین چهره‌ها)"""
        if depth_map is None:
            return 0.5
        return np.mean(depth_map) / 255.0