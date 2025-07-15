
# handspiral-cleaner

A lightweight data preprocessing pipeline for digitized spiral handwriting samples.

This project focuses on cleaning and preparing spiral-based handwriting datasets for downstream analysis, including feature extraction and model training tasks. It handles common preprocessing steps such as noise removal, format conversion, and structural validation.

## 🔍 Use Case

Originally built as part of a larger multimodal data project involving handwriting samples, this module helps in preprocessing data from pen tablet recordings — especially spirals drawn by users in controlled environments.

## 📁 Dataset

This repo expects input similar in structure to datasets derived from digitized handwriting tests (e.g. .png or .jpg images of spiral drawings).  
It works best with grayscale image data and can be extended to support additional modalities.

> ⚠️ *Note:* Dataset not included due to license restrictions. Placeholder directory structure is provided.

## ⚙️ Features

- Batch cleaning of handwritten spiral images
- Resize & normalization
- Binarization and denoising
- File renaming and I/O utilities
- Extendable for further feature extraction

## 🛠️ Technologies Used

- Python
- OpenCV
- NumPy
- Pandas (optional logging and data structure support)

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/yourusername/handspiral-cleaner.git
cd handspiral-cleaner
pip install -r requirements.txt
