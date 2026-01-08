# 🎭 Face Reconstruction Under Mask - Deep Learning Project

> **Deep Learning Course Project - Semester 20251**

**Objective**: Reconstruct masked facial regions using deep generative models.

---

## I. 👥 Team Members

| No. | Full Name        | Student ID |
| :-: | ---------------- | :--------: |
|  1  | Phan Đình Trung  |  20230093  |
|  2  | Phạm Văn Vũ Hoàn |  20235497  |
|  3  | Phan Hải Nguyên  |  20235540  |
|  4  | Lê Hoàng Nam     |  20235536  |
|  5  | Nguyễn Trung Hải |  20235495  |

---

## II. 📊 Datasets

### 2.1. Training Dataset

- **CelebA-HQ Resized 256×256**: [Kaggle Link](https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256)
- Total images: 30,000 high-quality face images
- Resolution: 256 × 256 pixels

### 2.2. Data Preprocessing

- Applied **OpenCV DNN Face Detector** for face detection
- Generated synthetic black rectangular masks covering the lower face region (nose, mouth, chin)
- Mask region: Lower 45% of the face bounding box (from 55% to 100% of face height)
- Dataset structure:
  - `without_mask/`: Original images (ground truth)
  - `with_mask/`: Masked images (input)

### 2.3. Cross-Dataset Evaluation & Validation

- **Flickr-Faces-HQ (FFHQ)**: [Kaggle Link](https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq)
- Used to evaluate model generalization on a different dataset

---

## III. 🔬 Member Contribution

### 1. GAN framework with convolutional networks and PatchGAN discriminator:

- **Phan Dinh Trung:** Use Double Convolutional Blocks for Generator
- **Le Hoang Nam:** Use Partial Convolution for Generator
- **Nguyen Trung Hai:** Use Facial Landmarks Prior for deeper facial analysis

### 2. GAN framework + Transformer: Phan Hai Nguyen

### 3. Diffusion models: Pham Van Vu Hoan

---

## IV. 📄 Documents

- 📑 **PDF Report**: [Link to Report](https://drive.google.com/...)
- 📊 **Presentation Slide**: [Link to Slide](https://drive.google.com/...)

---

_Last updated: January 2026_
