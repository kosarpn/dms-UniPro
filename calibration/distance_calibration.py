class DistanceCalibration:
    """
    کالیبراسیون فاصله برای تخمین دقیق نزدیک‌ترین چهره
    
    کاربرد:
    1. تشخیص راننده از سرنشین عقب (راننده نزدیک‌تر است)
    2. تشخیص خم شدن راننده به جلو/عقب
    3. تنظیم آستانه‌ها بر اساس فاصله
    """
    
    def __init__(self, reference_distance_cm: float = 50.0):
        self.reference_distance = reference_distance_cm
        self.reference_face_area = None      # اندازه صورت در فاصله مرجع
        self.reference_eye_distance = None   # فاصله دو چشم در فاصله مرجع
        
    def calibrate(self, face_bbox: tuple, eye_distance: float = None):
        """
        کالیبراسیون: ذخیره اندازه مرجع
        
        وقتی کاربر در فاصله استاندارد (مثلاً 50cm) نشسته،
        این تابع اندازه صورت را به عنوان مرجع ذخیره می‌کند.
        """
        x, y, w, h = face_bbox
        self.reference_face_area = w * h
        self.reference_eye_distance = eye_distance
        
        print(f"[DistanceCalibration] Reference set: area={self.reference_face_area}px at {self.reference_distance}cm")
    
    def estimate_distance(self, face_bbox: tuple) -> float:
        """
        تخمین فاصله فعلی بر اساس اندازه
        
        اصل: هرچه چهره کوچک‌تر دیده شود، دورتر است
        """
        if self.reference_face_area is None:
            return 50.0  # مقدار پیش‌فرض
        
        x, y, w, h = face_bbox
        current_area = w * h
        
        # رابطه معکوس: فاصله ∝ 1/√مساحت
        distance = self.reference_distance * (self.reference_face_area / current_area) ** 0.5
        return distance
    
    def get_distance_score(self, face_bbox: tuple) -> float:
        """
        محاسبه امتیاز نزدیکی (برای انتخاب راننده)
        
        Returns: 0-1 (1 = نزدیک‌ترین)
        """
        if self.reference_face_area is None:
            return 0.5
        
        x, y, w, h = face_bbox
        current_area = w * h
        
        # هرچه بزرگ‌تر = نزدیک‌تر = امتیاز بیشتر
        score = min(1.0, current_area / self.reference_face_area)
        return score