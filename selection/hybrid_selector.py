# selection/hybrid_selector.py
"""
انتخاب ترکیبی راننده با استفاده از:
- فاصله تخمینی (از MiDAS)
- موقعیت مکانی (چپ‌ترین/راست‌ترین)
- تاریخچه ردیابی
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
@dataclass
class FaceCandidate:
    """اطلاعات یک چهره کاندید"""
    index: int
    bbox: Tuple[int, int, int, int]
    distance_score: float   # 0-1 (نزدیک‌تر = بالاتر)
    position_score: float   # 0-1 (در محدوده راننده = بالاتر)
    tracking_score: float   # 0-1 (سازگاری با تاریخچه)
    total_score: float = 0.0
class HybridDriverSelector:
    """
    انتخاب راننده با ترکیب چندین معیار
    
    وزن‌ها:
    - distance_weight: 0.5 (فاصله)
    - position_weight: 0.3 (موقعیت)
    - tracking_weight: 0.2 (تاریخچه)
    """
    def __init__(self,
                 steering_side: str = "left",
                 frame_width: int = 640,
                 frame_height: int = 480,
                 distance_weight: float = 0.5,
                 position_weight: float = 0.3,
                 tracking_weight: float = 0.2):
        self.steering_side = steering_side
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.distance_weight = distance_weight
        self.position_weight = position_weight
        self.tracking_weight = tracking_weight
        # تاریخچه ردیابی
        self.last_driver_idx: Optional[int] = None
        self.last_driver_bbox: Optional[Tuple] = None
        self.tracking_history: List[Tuple] = []  # آخرین 10 موقعیت
        # محدوده راننده (از کالیبراسیون)
        self.driver_x_range: Tuple[float, float] = (0, frame_width)
        self.driver_y_range: Tuple[float, float] = (0, frame_height)
        self.is_calibrated = False
    
    def set_driver_range(self, x_range: Tuple[float, float], y_range: Tuple[float, float]):
        """
        تنظیم محدوده راننده (از کالیبراسیون)
        """
        self.driver_x_range = x_range
        self.driver_y_range = y_range
        self.is_calibrated = True
        print(f"[SELECTOR] Driver range set: X={x_range}, Y={y_range}")
    
    def select_driver(self,
                      faces: List[Tuple],
                      distances: List[Optional[float]] = None,
                      depth_map: np.ndarray = None) -> Optional[int]:
        """
        انتخاب راننده اصلی
        
        Args:
            faces: لیست bounding boxes (x, y, w, h)
            distances: لیست فاصله تخمینی هر چهره (اختیاری)
            depth_map: نقشه عمق از MiDAS (اختیاری)
        
        Returns:
            ایندکس راننده در لیست faces
        """
        if not faces:
            self._update_tracking(None, None)
            return None
        
        candidates = []
        
        for i, bbox in enumerate(faces):
            # معیار 1: فاصله (از MiDAS + اندازه)
            distance_score = self._calculate_distance_score(bbox, distances, depth_map, i)
            
            # معیار 2: موقعیت (در محدوده راننده)
            position_score = self._calculate_position_score(bbox)
            
            # معیار 3: تاریخچه (سازگاری با فریم‌های قبل)
            tracking_score = self._calculate_tracking_score(bbox)
            
            total_score = (
                distance_score * self.distance_weight +
                position_score * self.position_weight +
                tracking_score * self.tracking_weight
            )
            
            candidates.append(FaceCandidate(
                index=i,
                bbox=bbox,
                distance_score=distance_score,
                position_score=position_score,
                tracking_score=tracking_score,
                total_score=total_score
            ))
        
        # انتخاب بهترین کاندید
        best = max(candidates, key=lambda c: c.total_score)
        
        # آستانه حداقل امتیاز
        if best.total_score < 0.3:
            # امتیاز خیلی پایین، از تاریخچه استفاده کن
            if self.last_driver_idx is not None and self.last_driver_idx < len(faces):
                print(f"[SELECTOR] Low score ({best.total_score:.2f}), using last driver")
                return self.last_driver_idx
            return None
        
        # به‌روزرسانی تاریخچه
        self._update_tracking(best.index, best.bbox)
        
        return best.index
    def _calculate_distance_score(self,
                                   bbox: Tuple,
                                   distances: List[Optional[float]],
                                   depth_map: np.ndarray,
                                   face_idx: int) -> float:
        """
        محاسبه امتیاز فاصله
        نزدیک‌ترین چهره = امتیاز بیشتر
        """
        score = 0.5  # مقدار پیش‌فرض
        
        # روش 1: از distances محاسبه شده (MiDAS یا اندازه)
        if distances and face_idx < len(distances) and distances[face_idx] is not None:
            distance_val = distances[face_idx]
            # نرمالایز: فاصله کمتر = امتیاز بیشتر
            # فرض: فاصله معمولی بین 20-100
            norm_score = 1 - min(1.0, max(0, (distance_val - 20) / 100))
            score = max(score, norm_score)
        
        # روش 2: از نقشه عمق
        if depth_map is not None:
            x, y, w, h = bbox
            x = max(0, x)
            y = max(0, y)
            w = min(self.frame_width - x, w)
            h = min(self.frame_height - y, h)
            
            face_depth = np.mean(depth_map[y:y+h, x:x+w])
            depth_score = 1 - (face_depth / 255.0)  # روشن = نزدیک
            score = max(score, depth_score)
        
        # روش 3: از اندازه صورت (fallback)
        area = bbox[2] * bbox[3]
        max_area = self.frame_width * self.frame_height * 0.3  
        size_score = min(1.0, area / max_area)
        score = max(score, size_score * 0.7)
        
        return score
    
    def _calculate_position_score(self, bbox: Tuple) -> float:
        """
        محاسبه امتیاز موقعیت
        چهره در محدوده راننده = امتیاز بیشتر
        """
        x, y, w, h = bbox
        center_x = x + w / 2
        center_y = y + h / 2
        
        score = 0.0
        
        # امتیاز بر اساس محدوده X
        if self.driver_x_range[0] <= center_x <= self.driver_x_range[1]:
            score += 0.6
        else:
            # هرچه به محدوده نزدیک‌تر باشد
            dist_to_range = min(
                abs(center_x - self.driver_x_range[0]),
                abs(center_x - self.driver_x_range[1])
            )
            score += max(0, 0.6 * (1 - dist_to_range / 200))
        
        # امتیاز بر اساس محدوده Y
        if self.driver_y_range[0] <= center_y <= self.driver_y_range[1]:
            score += 0.4
        else:
            dist_to_range = min(
                abs(center_y - self.driver_y_range[0]),
                abs(center_y - self.driver_y_range[1])
            )
            score += max(0, 0.4 * (1 - dist_to_range / 150))
        
        return score
    
    def _calculate_tracking_score(self, bbox: Tuple) -> float:
        """
        محاسبه امتیاز سازگاری با تاریخچه
        چهره‌ای که در فریم‌های قبل راننده بوده = امتیاز بیشتر
        """
        if self.last_driver_bbox is None:
            return 0.5  # حد وسط
        
        x1, y1, w1, h1 = self.last_driver_bbox
        x2, y2, w2, h2 = bbox
        
        center1 = (x1 + w1/2, y1 + h1/2)
        center2 = (x2 + w2/2, y2 + h2/2)
        
        # فاصله اقلیدسی
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        # نرمالایز: فاصله 0 = امتیاز 1، فاصله > 150 = امتیاز 0
        score = max(0, 1 - distance / 150)
        
        return score
    
    def _update_tracking(self, idx: Optional[int], bbox: Optional[Tuple]):
        """
        به‌روزرسانی تاریخچه ردیابی
        """
        self.last_driver_idx = idx
        self.last_driver_bbox = bbox
        
        if bbox is not None:
            self.tracking_history.append(bbox)
            if len(self.tracking_history) > 10:
                self.tracking_history.pop(0)
    
    def get_statistics(self) -> Dict:
        """دریافت آمار انتخابگر"""
        return {
            'last_driver_idx': self.last_driver_idx,
            'tracking_history_len': len(self.tracking_history),
            'is_calibrated': self.is_calibrated,
            'driver_x_range': self.driver_x_range,
            'driver_y_range': self.driver_y_range
        }