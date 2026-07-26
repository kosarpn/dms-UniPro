# calibration/position_calibration.py

class PositionCalibration:
    """
    کالیبراسیون موقعیت راننده در تصویر
    
    کاربرد:
    1. تشخیص کدام چهره راننده است (نه سرنشین کناری/عقب)
    2. تشخیص تغییر وضعیت راننده (خم شدن، چرخیدن)
    3. تشخیص خروج راننده از محدوده طبیعی
    """
    
    def __init__(self, frame_width: int = 640, frame_height: int = 480):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.driver_x_range = (0, frame_width)   # محدوده طبیعی X
        self.driver_y_range = (0, frame_height)  # محدوده طبیعی Y
        self.driver_center_x = frame_width / 2
        self.driver_center_y = frame_height / 2
    def calibrate(self, face_bboxes: list):
        """
        کالیبراسیون موقعیت از چند فریم
        در طول کالیبراسیون (5-10 ثانیه)،
        موقعیت راننده را ذخیره و محدوده طبیعی را محاسبه می‌کند.
        """
        if not face_bboxes:
            return
        
        # محاسبه مرکز همه چهره‌های مشاهده شده
        centers_x = []
        centers_y = []
        
        for bbox in face_bboxes:
            x, y, w, h = bbox
            centers_x.append(x + w/2)
            centers_y.append(y + h/2)
        
        # محدوده طبیعی = میانگین ± 2 برابر انحراف معیار
        mean_x = np.mean(centers_x)
        std_x = np.std(centers_x)
        mean_y = np.mean(centers_y)
        std_y = np.std(centers_y)
        
        self.driver_center_x = mean_x
        self.driver_center_y = mean_y
        self.driver_x_range = (
            max(0, mean_x - 2 * std_x),
            min(self.frame_width, mean_x + 2 * std_x)
        )
        self.driver_y_range = (
            max(0, mean_y - 2 * std_y),
            min(self.frame_height, mean_y + 2 * std_y)
        )
        
        print(f"[PositionCalibration] Driver range: X={self.driver_x_range}, Y={self.driver_y_range}")
    
    def is_driver(self, face_bbox: tuple) -> bool:
        """
        بررسی آیا این چهره متعلق به راننده است
        
        Returns: True اگر در محدوده راننده باشد
        """
        x, y, w, h = face_bbox
        center_x = x + w/2
        center_y = y + h/2
        
        x_in_range = self.driver_x_range[0] <= center_x <= self.driver_x_range[1]
        y_in_range = self.driver_y_range[0] <= center_y <= self.driver_y_range[1]
        
        return x_in_range and y_in_range
    
    def get_position_score(self, face_bbox: tuple) -> float:
        """
        محاسبه امتیاز موقعیت (برای انتخاب راننده)
        
        Returns: 0-1 (1 = دقیقاً در مرکز محدوده راننده)
        """
        x, y, w, h = face_bbox
        center_x = x + w/2
        center_y = y + h/2
        
        # فاصله از مرکز محدوده راننده
        dist_x = abs(center_x - self.driver_center_x)
        dist_y = abs(center_y - self.driver_center_y)
        
        # حداکثر فاصله مجاز (نصف عرض محدوده)
        max_dist_x = (self.driver_x_range[1] - self.driver_x_range[0]) / 2
        max_dist_y = (self.driver_y_range[1] - self.driver_y_range[0]) / 2
        
        # نرمالایز: فاصله کمتر = امتیاز بیشتر
        score_x = 1 - min(1.0, dist_x / max_dist_x) if max_dist_x > 0 else 0.5
        score_y = 1 - min(1.0, dist_y / max_dist_y) if max_dist_y > 0 else 0.5
        
        # ترکیب امتیاز X و Y
        return (score_x + score_y) / 2