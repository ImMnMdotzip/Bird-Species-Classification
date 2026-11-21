# 🦜 ระบบจำแนกสายพันธุ์นกด้วยโครงข่ายประสาทเทียมแบบสไปกิง (Spiking Neural Networks)

[English](README.md) | [ภาษาไทย](README_TH.md)

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.8-EE4C2C?logo=pytorch&logoColor=white)
![snnTorch](https://img.shields.io/badge/snnTorch-SNN-blue)
![Accuracy](https://img.shields.io/badge/Validation%20Accuracy-89.5%25-brightgreen)

## ภาพรวมโปรเจกต์ (Overview)

โปรเจกต์นี้เป็นการพัฒนาระบบจำแนกสายพันธุ์นกอัตโนมัติโดยใช้ **Spiking Neural Network (SNN)** พัฒนาด้วย **PyTorch** และ **snnTorch** สามารถจำแนกนกได้ **7 สายพันธุ์** ที่ความละเอียดภาพ **128 × 128 พิกเซล** และทำค่าความแม่นยำรวมบนชุดข้อมูลตรวจสอบ (Validation Accuracy) ได้สูงถึง **89.5%** หลังจากการเทรน 30 รอบ (Epochs)

ระบบถูกออกแบบให้ประมวลผลบนเครื่อง Local ผ่าน **Visual Studio Code (VS Code)** โดยใช้การ์ดจอประมวลผล (CUDA 12.8) ร่วมกับการดึงและซิงก์ชุดข้อมูลภาพจากคลาวด์อัตโนมัติผ่าน **Google Drive API**

---

## วัตถุประสงค์ (Objectives)

- พัฒนาและเทรนโมเดลโครงข่ายประสาทเทียมแบบสไปกิง (SNN) สำหรับงานจำแนกสายพันธุ์นกหลายคลาส (Multi-class Classification)
- เชื่อมต่อและดึงชุดข้อมูลภาพจาก Google Drive มายังสภาพแวดล้อมการทำงานในเครื่องอัตโนมัติด้วย Google Drive API
- ใช้ความสามารถในการประมวลผลแบบขนานของ GPU (NVIDIA Blackwell / CUDA 12.8) เพื่อเร่งความเร็วในการเรียนรู้แบบ Surrogate Gradient
- บันทึกน้ำหนักโมเดลที่เทรนสำเร็จ (`snn_birds_best.pt`) พร้อมฟังก์ชันทำนายผล (`predict()`) สำหรับทดสอบกับภาพนกใหม่

---

## เทคโนโลยีและเครื่องมือที่ใช้ (Technologies)

- **ภาษาและสภาพแวดล้อม:** Python 3.12, Visual Studio Code, Jupyter Notebook, `uv` Package Manager
- **Deep Learning:** PyTorch (`cu128`), snnTorch, Torchvision
- **การจัดการข้อมูลและแสดงผล:** NumPy, Pillow (PIL), Matplotlib, scikit-learn
- **การเชื่อมต่อคลาวด์:** Google Drive API (`google-api-python-client`, `google-auth-oauthlib`)
- **ฮาร์ดแวร์ประมวลผล:** NVIDIA GeForce RTX 5060 Ti (CUDA 12.8 / `sm_120`)

---

## โครงสร้างโปรเจกต์ (Project Structure)

```text
Bird-Species-Classification/
│
├── .vscode/
│   └── settings.json
├── outputs/
│   └── snn_birds_best.pt                   # ไฟล์น้ำหนักโมเดล SNN ที่ดีที่สุด (ความแม่นยำ 89.5%)
├── test_images/
│   ├── test_kingfisher.jpg                 # ภาพทดสอบตัวอย่าง (นกกระเต็นน้อยธรรมดา)
│   └── test_peacock.png                    # ภาพทดสอบตัวอย่าง (นกยูงอินเดีย)
├── .gitignore
├── SNN_Birds_VSCode_GoogleDrive_API.ipynb  # ไฟล์หลักสำหรับเทรน ประเมินผล และทดสอบทำนายภาพ
├── README.md                               # เอกสารอธิบายโปรเจกต์ (ภาษาอังกฤษ)
└── README_TH.md                            # เอกสารอธิบายโปรเจกต์ (ภาษาไทย)
```

> **หมายเหตุ:** สภาพแวดล้อมเสมือน (`.venv/`), ไฟล์รูปภาพดิบ (`data/Image_Dataset/`) และไฟล์รหัสผ่าน OAuth (`credentials.json` และ `token.json`) ถูกซ่อนไว้ด้วย `.gitignore` เพื่อความปลอดภัย

---

## ชุดข้อมูล (Dataset)

ชุดข้อมูลประกอบด้วยภาพสี RGB ของนก **7 สายพันธุ์** (สายพันธุ์ละ 240 ภาพในชุด Validation รวมทั้งหมด 1,680 ภาพ) ปรับขนาดเป็น **128 × 128 พิกเซล**:

1. `Common-Kingfisher` (นกกระเต็นน้อยธรรมดา)
2. `Common-Myna` (นกเอี้ยงสาริกา)
3. `House-Crow` (อีกาบ้าน)
4. `Indian-Peacock` (นกยูงอินเดีย)
5. `Indian-Pitta` (นกแต้วแล้วธรรมดา)
6. `Ruddy-Shelduck` (เป็ดพม่า)
7. `Sarus-Crane` (นกกระเรียนไทย)

ขณะรันโค้ด ตัว Notebook จะยืนยันตัวตนผ่าน OAuth 2.0 (`credentials.json`) และดาวน์โหลดชุดข้อมูลจาก Google Drive มาเก็บไว้ใน `data/Image_Dataset/` โดยอัตโนมัติ

---

## สถาปัตยกรรมโมเดลและขั้นตอนการทำงาน (Methodology)

- **การสกัดคุณลักษณะภาพ (Feature Extraction):** ใช้ชั้น Convolutional (`Conv2d`, Batch Normalization, Pooling) เพื่อดึงลักษณะเด่นเชิงพื้นที่ของภาพ
- **พลวัตของเซลล์ประสาทสไปกิง (Spiking Dynamics):** ใช้เซลล์ประสาทแบบ Leaky Integrate-and-Fire (**LIF**) ทำงานตามลำดับช่วงเวลา (Time Steps)
- **กลยุทธ์การเทรน:** ใช้ Surrogate Gradient Descent ร่วมกับตัวปรับค่า **Adam** และฟังก์ชันสูญเสีย Cross-Entropy เป็นจำนวน **30 Epochs**

### ขั้นตอนการทำงาน (Workflow Pipeline)

1. ยืนยันตัวตนกับ Google Drive API และดาวน์โหลด/ตรวจสอบชุดข้อมูลใน `data/Image_Dataset/`
2. ปรับขนาดภาพ (`128 × 128`), แปลงเป็น Tensor และทำ Normalization
3. แบ่งข้อมูลออกเป็นชุดฝึกสอน (Training Set) และชุดตรวจสอบ (Validation Set)
4. เทรนโมเดล SNN บนการ์ดจอ (`cuda`) และบันทึกน้ำหนักที่ดีที่สุดลงไฟล์ `outputs/snn_birds_best.pt`
5. ประเมินประสิทธิภาพรายสายพันธุ์ด้วยค่า Precision, Recall, F1-score และ Confusion Matrix
6. ทดสอบทำนายผล Top-$k$ กับภาพใหม่ในโฟลเดอร์ `test_images/` ผ่านฟังก์ชัน `predict()`

---

## ผลการทดลอง (Results)

หลังจากการเทรนครบ 30 Epochs โมเดล SNN สามารถทำค่าความแม่นยำรวมได้ **89.5%** จากภาพทดสอบทั้งหมด 1,680 ภาพ:

| สายพันธุ์นก (Bird Species) | Precision | Recall | F1-Score | จำนวนภาพ (Support) |
| :--- | :---: | :---: | :---: | :---: |
| **Common-Kingfisher** | 0.892 | 0.933 | 0.912 | 240 |
| **Common-Myna** | 0.918 | 0.842 | 0.878 | 240 |
| **House-Crow** | 0.842 | 0.821 | 0.831 | 240 |
| **Indian-Peacock** | 0.920 | 0.863 | 0.890 | 240 |
| **Indian-Pitta** | 0.904 | 0.938 | 0.920 | 240 |
| **Ruddy-Shelduck** | 0.970 | 0.933 | 0.951 | 240 |
| **Sarus-Crane** | 0.833 | 0.938 | 0.882 | 240 |
| **ค่าเฉลี่ยรวม (Overall / Macro Avg)** | **0.897** | **0.895** | **0.895** | **1680** |

---

## วิธีการติดตั้งและใช้งาน (Getting Started)

1. **โคลนโปรเจกต์ลงเครื่อง:**
   ```bash
   git clone [https://github.com/ImMnMdotzip/Bird-Species-Classification.git](https://github.com/ImMnMdotzip/Bird-Species-Classification.git)
   cd Bird-Species-Classification
   ```

2. **สร้าง Environment และติดตั้งไลบรารี (ด้วย `uv`):**
   ```bash
   uv venv --python 3.12
   uv pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu128](https://download.pytorch.org/whl/cu128)
   uv pip install snntorch matplotlib scikit-learn pillow google-api-python-client google-auth-httplib2 google-auth-oauthlib ipykernel
   ```

3. **การรันทดสอบหรือเทรนโมเดลใหม่:**
   - **ทดสอบทำนายผลทันที (Quick Inference):** เปิดไฟล์ `SNN_Birds_VSCode_GoogleDrive_API.ipynb` โหลดน้ำหนักโมเดลที่เทรนแล้วจาก `outputs/snn_birds_best.pt` และรันคำสั่ง `predict()` กับภาพใน `test_images/`
   - **เทรนโมเดลใหม่ทั้งหมด (Full Training):** นำไฟล์ `credentials.json` จาก Google Cloud OAuth มาวางไว้ที่โฟลเดอร์หลักของโปรเจกต์ แล้วสั่งรันทุก Cell ตั้งแต่ต้นจนจบ

---

## ผู้จัดทำ (Author)

**ImMnMdotzip**
