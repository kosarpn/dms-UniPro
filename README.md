# Driver Drowsiness Detection System

این پروژه یک سیستم تشخیص خواب‌آلودگی راننده (Driver Monitoring System) است که با استفاده از دوربین، وضعیت راننده را بررسی می‌کند و در صورت تشخیص خواب‌آلودگی هشدار می‌دهد.

در این پروژه بیشتر تمرکز روی استفاده از پردازش تصویر، استخراج ویژگی‌های چهره و بررسی چند روش مختلف برای تشخیص خواب‌آلودگی است.

## امکانات پروژه

- تشخیص چهره با استفاده از MediaPipe
- استفاده از YOLO به عنوان روش کمکی در تشخیص چهره
- انتخاب راننده در صورت وجود چند چهره در تصویر
- استفاده از اطلاعات فاصله، موقعیت و تاریخچه چهره برای انتخاب راننده
- استخراج ویژگی‌های مربوط به چشم، دهان و وضعیت سر
- محاسبه EAR برای بررسی وضعیت چشم‌ها
- محاسبه MAR برای بررسی وضعیت دهان و خمیازه
- تشخیص وضعیت سر با استفاده از Head Pose
- استفاده از MiDaS برای تخمین عمق نسبی تصویر
- کالیبراسیون اولیه برای هر راننده
- استفاده از EMA برای کاهش نوسانات ویژگی‌ها
- تشخیص خواب‌آلودگی به صورت Real-Time
- استفاده از روش Rule-Based
- آموزش و استفاده از SVM
- آموزش و استفاده از Random Forest
- ارزیابی مدل‌ها با Accuracy، Precision، Recall و F1-Score

---

# ساختار کلی پروژه

ساختار کلی پروژه به شکل زیر است:

```text
driver-drowsiness-detection/
│
├── alerts/
│   └── alert_manager.py
│
├── calibration/
│   └── calibration_manager.py
│
├── core/
│   ├── data_types.py
│   └── features_vector.py
│
├── detectors/
│   └── hybrid_detector.py
│
├── depth/
│   └── midas_depth.py
│
├── features/
│   ├── ear.py
│   ├── mar.py
│   ├── head_pose.py
│   └── extractor.py
│
├── ml/
│   ├── classifiers/
│   │   ├── base_classifier.py
│   │   ├── rule_based_classifier.py
│   │   ├── svm_classifier.py
│   │   └── rf_classifier.py
│   │
│   ├── dataset_builder.py
│   ├── build_dataset.py
│   ├── train_svm.py
│   ├── train_rf.py
│   └── evaluator.py
│
├── selection/
│   └── hybrid_selector.py
│
├── models/
│   ├── svm_model.pkl
│   ├── svm_scaler.pkl
│   └── rf_model.pkl
│
├── data/
│   ├── raw/
│   └── processed/
│
├── results/
│
├── main.py
├── requirements.txt
└── README.md
```

---

# روند کلی سیستم

در حالت Real-Time، روند کلی برنامه به این صورت است:

```text
Camera
   │
   ▼
Face Detection
   │
   ▼
Driver Selection
   │
   ├── Distance
   ├── Position
   └── Tracking
   │
   ▼
Feature Extraction
   │
   ├── EAR
   ├── MAR
   └── Head Pose
   │
   ▼
EMA Smoothing
   │
   ▼
Drowsiness Classifier
   │
   ├── Rule-Based
   ├── SVM
   └── Random Forest
   │
   ▼
Alert
```

یعنی ابتدا تصویر از دوربین دریافت می‌شود. بعد چهره یا چهره‌های موجود در تصویر پیدا می‌شوند. اگر بیشتر از یک چهره وجود داشته باشد، سیستم با استفاده از چند معیار سعی می‌کند چهره مربوط به راننده را انتخاب کند.

بعد از انتخاب راننده، ویژگی‌های مورد نیاز از چهره استخراج می‌شوند و در مرحله بعد برای تشخیص خواب‌آلودگی استفاده می‌شوند.

---

# تشخیص چهره

برای تشخیص چهره در پروژه از `MediaPipe Face Mesh` استفاده شده است.

MediaPipe علاوه بر تشخیص چهره، نقاط مختلفی از صورت را نیز در اختیار سیستم قرار می‌دهد. این نقاط برای محاسبه ویژگی‌هایی مثل EAR، MAR و Head Pose استفاده می‌شوند.

در پروژه، MediaPipe روش اصلی تشخیص چهره است و Hybrid Detector امکان استفاده از روش کمکی را نیز در نظر گرفته است.

---

# انتخاب راننده

اگر فقط یک نفر جلوی دوربین باشد، انتخاب راننده ساده است. اما اگر مثلاً راننده و سرنشین هر دو در تصویر دیده شوند، سیستم باید مشخص کند کدام چهره مربوط به راننده است.

برای این کار فایل:

```text
selection/hybrid_selector.py
```

استفاده می‌شود.

در این بخش سه معیار اصلی بررسی می‌شوند:

### 1. Distance

چهره‌ای که به دوربین نزدیک‌تر باشد معمولاً احتمال بیشتری دارد که مربوط به راننده باشد.

برای تخمین عمق از MiDaS استفاده شده و در صورت نبود اطلاعات مناسب، اندازه صورت نیز می‌تواند به عنوان معیار کمکی استفاده شود.

### 2. Position

موقعیت چهره در تصویر بررسی می‌شود.

محدوده‌ای که در مرحله Calibration برای راننده مشخص شده است، در این قسمت استفاده می‌شود.

### 3. Tracking

سیستم موقعیت راننده در فریم‌های قبلی را نیز در نظر می‌گیرد.

به عنوان مثال اگر در فریم قبلی یک چهره به عنوان راننده انتخاب شده باشد، در فریم بعدی چهره‌ای که نزدیک‌تر به موقعیت قبلی راننده باشد امتیاز بیشتری دریافت می‌کند.

در نهایت این سه معیار با وزن‌های مشخص ترکیب می‌شوند و چهره‌ای که امتیاز بیشتری داشته باشد به عنوان راننده انتخاب می‌شود.

---

# استخراج ویژگی‌ها

بعد از انتخاب راننده، ویژگی‌های مورد نیاز از landmarks چهره استخراج می‌شوند.

این قسمت در پوشه:

```text
features/
```

قرار دارد.

ویژگی‌های اصلی شامل موارد زیر هستند:

```text
EAR
MAR
Pitch
Yaw
Roll
```

همچنین تعدادی ویژگی زمانی مانند:

```text
Blink Rate
PERCLOS
Closed Duration
Yawn Duration
Head Down Frames
```

در مسیر Real-Time سیستم محاسبه می‌شوند.

---

# EAR

EAR یا Eye Aspect Ratio برای بررسی وضعیت باز یا بسته بودن چشم استفاده می‌شود.

به طور کلی وقتی چشم باز است مقدار EAR در یک محدوده مشخص قرار دارد و وقتی چشم بسته می‌شود مقدار آن کاهش پیدا می‌کند.

بنابراین سیستم می‌تواند با بررسی EAR و همچنین مدت زمان بسته بودن چشم، احتمال خواب‌آلودگی را بررسی کند.

---

# MAR

MAR یا Mouth Aspect Ratio برای بررسی باز شدن دهان استفاده می‌شود.

این ویژگی بیشتر برای تشخیص خمیازه مورد استفاده قرار می‌گیرد.

به تنهایی بالا بودن MAR به معنی خواب‌آلودگی نیست، بنابراین در سیستم Real-Time مدت زمان باز بودن دهان نیز در نظر گرفته می‌شود.

---

# Head Pose

در این بخش وضعیت سر با سه زاویه اصلی بررسی می‌شود:

```text
Pitch
Yaw
Roll
```

برای مثال Pitch می‌تواند برای بررسی پایین بودن سر و Yaw برای بررسی چرخش سر به چپ یا راست استفاده شود.

این ویژگی‌ها در کنار EAR و MAR برای تشخیص بهتر وضعیت راننده استفاده می‌شوند.

---

# MiDaS

فایل مربوط به MiDaS در مسیر زیر قرار دارد:

```text
depth/midas_depth.py
```

MiDaS برای تخمین عمق نسبی تصویر استفاده می‌شود.

در پروژه از این اطلاعات بیشتر برای کمک به انتخاب راننده استفاده شده است.

روند کلی آن به شکل زیر است:

```text
Frame
  │
  ▼
MiDaS
  │
  ▼
Depth Map
  │
  ▼
Face Depth
  │
  ▼
Distance Score
```

نکته مهم این است که این بخش با مختصات `Z` مربوط به landmarks مدیپایپ یکی نیست. در پروژه برای انتخاب راننده از تخمین عمق MiDaS استفاده شده است.

---

# Calibration

در پروژه برای هر راننده یک مرحله کالیبراسیون در نظر گرفته شده است.

کالیبراسیون در فایل:

```text
calibration/calibration_manager.py
```

مدیریت می‌شود.

در این مرحله اطلاعاتی مثل مقدار معمول EAR و محدوده قرارگیری راننده در تصویر مشخص می‌شود.

هدف این است که سیستم فقط از یک مقدار ثابت برای همه افراد استفاده نکند و بتواند بعضی از پارامترها را با توجه به شرایط راننده تنظیم کند.

---

# EMA Smoothing

مقادیر ویژگی‌هایی مثل EAR ممکن است بین فریم‌های مختلف کمی تغییر کنند.

این تغییرات همیشه به معنی تغییر واقعی وضعیت راننده نیستند و ممکن است به دلیل نویز تشخیص landmarks باشند.

برای کاهش این نوسانات از EMA یا Exponential Moving Average استفاده شده است.

به صورت ساده:

```text
Raw Feature
     │
     ▼
EMA
     │
     ▼
Smoothed Feature
     │
     ▼
Classifier
```

بنابراین classifier به جای اینکه مستقیماً روی مقدار خام هر فریم تصمیم بگیرد، از مقدار هموارشده استفاده می‌کند.

---

# بخش Machine Learning

پوشه `ml` مربوط به قسمت یادگیری ماشین پروژه است.

در این قسمت سه روش اصلی بررسی شده‌اند:

```text
Rule-Based
SVM
Random Forest
```

---

# Dataset

برای آموزش مدل‌های SVM و Random Forest ابتدا تصاویر دیتاست پردازش می‌شوند.

تصاویر ابتدا وارد MediaPipe می‌شوند و بعد از تشخیص landmarks، ویژگی‌های مورد نیاز از آنها استخراج می‌شود.

در نهایت این ویژگی‌ها در یک فایل CSV ذخیره می‌شوند.

روند ساخت Dataset:

```text
Images
   │
   ▼
Face Detection
   │
   ▼
Landmarks
   │
   ▼
Feature Extraction
   │
   ▼
EAR / MAR / Head Pose
   │
   ▼
dataset.csv
```

فایل اصلی دیتاست پردازش‌شده:

```text
data/processed/dataset.csv
```

است.

---

# Dataset Builder

دو فایل مربوط به ساخت دیتاست در پروژه وجود دارد:

```text
dataset_builder.py
build_dataset.py
```

`dataset_builder.py` کلاس `DatasetBuilder` را تعریف می‌کند.

این کلاس کارهایی مثل خواندن تصویر، تشخیص چهره، استخراج ویژگی و ذخیره نمونه‌ها را انجام می‌دهد.

در مقابل، `build_dataset.py` بیشتر نقش اجرای این کلاس را دارد.

یعنی به زبان ساده:

```text
dataset_builder.py
        │
        │  ابزار ساخت دیتاست
        ▼
DatasetBuilder

build_dataset.py
        │
        │  اجرای DatasetBuilder
        ▼
dataset.csv
```

---

# ویژگی‌های استفاده‌شده در مدل‌های Machine Learning

برای آموزش SVM و Random Forest از 8 ویژگی زیر استفاده شده است:

```text
1. ear
2. mar
3. pitch
4. yaw
5. roll
6. is_head_down
7. is_head_left
8. is_head_right
```

این ویژگی‌ها به صورت عددی در `dataset.csv` ذخیره می‌شوند.

مدل‌های SVM و Random Forest مستقیماً با تصویر کار نمی‌کنند.

روند کار به صورت زیر است:

```text
Image
  │
  ▼
Face Landmarks
  │
  ▼
Feature Extraction
  │
  ▼
8 Numerical Features
  │
  ▼
SVM / Random Forest
  │
  ▼
Drowsiness Prediction
```

---

# Rule-Based Classifier

روش Rule-Based بر اساس قوانین و آستانه‌هایی که از قبل مشخص شده‌اند تصمیم‌گیری می‌کند.

برای مثال:

EAR پایین
PERCLOS بالا
مدت بسته بودن چشم زیاد
خمیازه
Blink Rate بالا
سر پایین

هر کدام از این شرایط می‌توانند مقداری امتیاز به وضعیت خواب‌آلودگی اضافه کنند.

در نهایت مجموع امتیازها بررسی می‌شود.

برای مثال در کد فعلی:

Low EAR → 25
High PERCLOS → 25
Closed Duration → 20
High Blink Rate → 10
Yawning → 10
Head Down → 10

اگر امتیاز از آستانه مشخص‌شده بیشتر شود، راننده خواب‌آلود در نظر گرفته می‌شود.

Rule-Based برخلاف SVM و Random Forest نیاز به آموزش ندارد.

---

# SVM

در پروژه از SVM با Kernel نوع RBF استفاده شده است.

SVM با استفاده از نمونه‌های آموزشی یاد می‌گیرد که بین دو کلاس:

```text
Not Drowsy
Drowsy
```

مرز مناسبی پیدا کند.

قبل از آموزش SVM، ویژگی‌ها با `StandardScaler` استاندارد می‌شوند.

روند آموزش:

```text
dataset.csv
    │
    ▼
Select Features
    │
    ▼
Train / Test Split
    │
    ▼
StandardScaler
    │
    ▼
SVM
    │
    ▼
Trained Model
```

---

# Random Forest

Random Forest از چندین Decision Tree تشکیل شده است.

در این پروژه از 100 درخت استفاده شده است.

هر درخت روی ویژگی‌های ورودی تصمیم‌گیری می‌کند و در نهایت نتیجه درخت‌ها با هم ترکیب می‌شود.

برخلاف SVM، برای Random Forest نیازی به StandardScaler وجود ندارد.

---

# آموزش مدل‌ها

فایل‌های آموزش مدل‌ها عبارت‌اند از:

```text
ml/train_svm.py
ml/train_rf.py
```

## آموزش SVM

برای آموزش SVM:

```bash
python ml/train_svm.py
```

در این مرحله Dataset خوانده می‌شود، ویژگی‌ها جدا می‌شوند، داده به Train و Test تقسیم می‌شود، ویژگی‌ها استاندارد می‌شوند و سپس مدل آموزش داده می‌شود.

در پایان دو فایل ذخیره می‌شود:

```text
models/
├── svm_model.pkl
└── svm_scaler.pkl
```

`svm_model.pkl` خود مدل آموزش‌دیده SVM است.

`svm_scaler.pkl` اطلاعات مربوط به استانداردسازی ویژگی‌ها را نگه می‌دارد تا هنگام استفاده از مدل روی داده جدید، همان تبدیل دوباره انجام شود.

## آموزش Random Forest

برای آموزش Random Forest:

```bash
python ml/train_rf.py
```

مدل آموزش‌دیده در این مسیر ذخیره می‌شود:

```text
models/rf_model.pkl
```

Random Forest به scaler نیاز ندارد.

---

# Classifier ها

فایل‌های زیر مربوط به استفاده از مدل‌ها در برنامه اصلی هستند:

```text
ml/classifiers/
│
├── base_classifier.py
├── rule_based_classifier.py
├── svm_classifier.py
└── rf_classifier.py
```

`BaseClassifier` یک interface مشترک برای classifierها تعریف می‌کند.

`RuleBasedClassifier` قوانین تشخیص را اجرا می‌کند.

`SVMClassifier` مدل ذخیره‌شده SVM را load کرده و برای داده جدید prediction انجام می‌دهد.

`RFClassifier` نیز مدل ذخیره‌شده Random Forest را load کرده و prediction انجام می‌دهد.

---

# تفاوت Trainer و Classifier

Trainer و Classifier یک کار انجام نمی‌دهند.

Trainer برای مرحله آموزش و ارزیابی مدل استفاده می‌شود.

مثلاً:

```text
train_svm.py
train_rf.py
```

مدل را با Dataset آموزش می‌دهند.

اما Classifier در زمان اجرای سیستم از مدل آموزش‌دیده استفاده می‌کند.

مثلاً:

```text
svm_classifier.py
rf_classifier.py
```

به صورت ساده:

```text
Training:

Dataset
   ↓
Trainer
   ↓
Trained Model
   ↓
.pkl


Real-Time:

Camera
   ↓
Features
   ↓
Classifier
   ↓
Prediction
```

---

# Offline و Real-Time

در پروژه دو حالت اصلی وجود دارد.

## Real-Time

در حالت Real-Time اطلاعات مستقیماً از دوربین دریافت می‌شود.

برای مثال:

```text
Camera Frame
   ↓
Detection
   ↓
Feature Extraction
   ↓
Classifier
   ↓
Prediction
```

در این قسمت متد اصلی classifier:

```python
predict()
```

است.

## Offline

در حالت Offline، داده‌ها از فایل CSV خوانده می‌شوند و مدل‌ها روی مجموعه‌ای از داده‌های از قبل آماده‌شده آزمایش می‌شوند.

در این حالت از:

```python
predict_batch()
```

استفاده شده است.

هدف این قسمت بیشتر بررسی عملکرد مدل‌ها روی یک مجموعه داده مشخص است.

---

# Evaluator

فایل:

```text
ml/evaluator.py
```

برای مقایسه سه روش استفاده می‌شود:

```text
Rule-Based
SVM
Random Forest
```

ابتدا Dataset خوانده می‌شود و همان تقسیم Train/Test مورد استفاده در آموزش دوباره ایجاد می‌شود.

سپس مدل‌ها روی داده‌های Test prediction انجام می‌دهند.

در نهایت معیارهای زیر برای هر مدل محاسبه می‌شوند:

```text
Accuracy
Precision
Recall
F1-Score
```

نتایج نیز در فایل:

```text
results/evaluation_results.csv
```

ذخیره می‌شوند.

---

# معیارهای ارزیابی

### Accuracy

نشان می‌دهد چه درصدی از کل پیش‌بینی‌ها درست بوده‌اند.

### Precision

از بین نمونه‌هایی که مدل آنها را خواب‌آلود تشخیص داده، چند مورد واقعاً خواب‌آلود بوده‌اند.

### Recall

از بین نمونه‌های واقعاً خواب‌آلود، چند مورد توسط مدل درست تشخیص داده شده‌اند.

### F1-Score

ترکیبی از Precision و Recall است و برای مقایسه مدل‌ها استفاده می‌شود.

---

# اجرای پروژه

ابتدا وابستگی‌های پروژه را نصب کنید:

```bash
pip install -r requirements.txt
```

برای ساخت Dataset:

```bash
python ml/build_dataset.py
```

برای آموزش SVM:

```bash
python ml/train_svm.py
```

برای آموزش Random Forest:

```bash
python ml/train_rf.py
```

برای ارزیابی مدل‌ها:

```bash
python ml/evaluator.py
```

برای اجرای سیستم Real-Time:

```bash
python main.py
```

---

# جریان کامل پروژه

در حالت کلی پروژه از دو بخش تشکیل شده است.

## بخش آموزش و ارزیابی

```text
Raw Images
    │
    ▼
Dataset Builder
    │
    ▼
dataset.csv
    │
    ├───────────────┐
    ▼               ▼
Train SVM       Train RF
    │               │
    ▼               ▼
SVM Model        RF Model
    │               │
    └───────┬───────┘
            ▼
        Evaluator
            │
            ▼
Accuracy / Precision
Recall / F1
```

## بخش Real-Time

```text
Camera
   │
   ▼
Face Detection
   │
   ▼
Driver Selection
   │
   ▼
Feature Extraction
   │
   ▼
EMA Smoothing
   │
   ▼
Classifier
   │
   ▼
Drowsiness Detection
   │
   ▼
Alert
```

---

# نکته درباره مدل‌ها

در این پروژه مدل‌های SVM و Random Forest از تصاویر خام مستقیماً یاد نمی‌گیرند.

تصاویر ابتدا به ویژگی‌های عددی تبدیل می‌شوند.

برای مثال:

```text
EAR = 0.21
MAR = 0.31
Pitch = 15
Yaw = -5
Roll = 2
...
```

این مقادیر وارد مدل می‌شوند و مدل بر اساس نمونه‌های آموزشی یاد می‌گیرد که این ترکیب ویژگی‌ها بیشتر مربوط به حالت خواب‌آلود یا هوشیار است.

---

# هدف پروژه

هدف اصلی پروژه این است که با استفاده از اطلاعات قابل استخراج از چهره راننده، وضعیت خواب‌آلودگی او را در زمان واقعی تشخیص دهد.

همچنین با استفاده از سه روش Rule-Based، SVM و Random Forest امکان مقایسه روش‌های مختلف تشخیص فراهم شده است.

در بخش Offline عملکرد مدل‌ها با معیارهای استاندارد یادگیری ماشین بررسی می‌شود و در بخش Real-Time مدل‌ها در شرایط اجرای واقعی سیستم مورد استفاده قرار می‌گیرند.
