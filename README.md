# VisionShape
This project uses computer vision techniques to detect and analyze geometric shapes from images. It identifies object contours, classifies shapes, and extracts key features such as area and perimeter through an interactive dashboard.

Below is a **clean, well-structured, GitHub-submission–ready `README.md`** tailored **exactly to your current project code and features**.
You can **copy-paste this directly** into a `README.md` file in your repository.

---

# 📐 Geometric Shape & Contour Analyzer

### *Computer Vision DA1 Project*

## 📌 Overview

The **Geometric Shape & Contour Analyzer** is a computer vision–based application that detects, classifies, and analyzes geometric shapes from images. Using OpenCV-based image processing techniques, the system extracts object contours, identifies shape types, and computes geometric properties through an interactive Streamlit dashboard.

This project is developed as part of **Computer Vision DA1** and demonstrates practical applications of image preprocessing, contour detection, and feature extraction.

---

## 🎯 Features

* Upload images containing geometric shapes
* Automatic contour detection using OpenCV
* Shape classification:

  * Triangle
  * Square
  * Rectangle
  * Circle / Oval
  * Polygon
* Extraction of geometric features:

  * Area
  * Perimeter
  * Number of vertices
* Detection of **most frequently occurring shape**
* Interactive visualization:

  * Annotated contour image
  * Binary threshold image
* Shape distribution analysis
* Download options:

  * Annotated contour image
  * Binary image
  * CSV feature report
* Clean, light-themed professional dashboard UI

---

## 🛠️ Technologies Used

* **Python**
* **OpenCV** – Image processing and contour detection
* **NumPy** – Numerical computations
* **Pillow (PIL)** – Image handling
* **Pandas** – Data analysis
* **Streamlit** – Interactive web dashboard

---

## 📂 Project Structure

```
├── app.py          # Main Streamlit application
├── README.md       # Project documentation
├── requirements.txt
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/geometric-shape-analyzer.git
cd geometric-shape-analyzer
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt**

```
streamlit
opencv-python
numpy
pillow
pandas
```

### 3️⃣ Run the Application

```bash
streamlit run app.py
```

---

## 🖼️ How the System Works

1. User uploads an image through the dashboard
2. Image preprocessing:

   * Grayscale conversion
   * Gaussian blurring
   * Binary thresholding
3. Contours are detected using OpenCV
4. Shapes are classified based on polygon approximation
5. Geometric features are calculated and displayed
6. Results are visualized and made available for download

---

## 📊 Outputs

* Annotated image with detected contours
* Binary threshold image
* Shape distribution chart
* Tabular feature data
* Downloadable CSV report

---

## 🎓 Applications

* Computer Vision laboratory experiments
* DA / Mini Project submissions
* Image processing demonstrations
* Educational and research purposes

---

## 👤 Author

**Akhileshwar Singh**
Computer Vision DA1

---

## 📜 License

This project is intended for **educational and academic use only**.

---

Just tell me 👍
