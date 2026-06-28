&#x09;			🩸 **AnemoScan – AI-Based Early Detection of Anemia from Nail and Conjunctiva Images**



* **AnemoScan** is a graduation project that combines **deep learning** and **explainable AI** to detect anemia from **smartphone‑captured images** of the **conjunctiva** and **fingernails.** The system provides a **non‑invasive, low‑cost screening tool with interpretable heatmaps (Grad‑CAM)** and **personalized nutritional advice.**


🎯 **Key Features**



**1. Two‑stage AI pipeline**



* **YOLOv8 object detection** → localizes the **region of interest (ROI)** in the uploaded image.  
* **MobileNetV2 binary classifier** → predicts **Anemic / Non‑Anemic** from the cropped **ROI.**



**2.** **Explainable AI (XAI)** – **Grad‑CAM heatmaps** highlight the image regions that influenced the prediction.  

**3.** **Nutritional guidance** – **Automatic, context‑aware advice** based on the prediction result.  

**4.** **Interactive web app** – **Built with Streamlit,** supports **image upload, live preview,** and **downloadable PDF reports.**  

**5.** **Lightweight** – **Models are small (best.pt \~6 MB, .h5 models \~9 MB each)** and run on a **standard CPU.**



🧠 **Model Architecture**



|      **Component**        |           **Model / Approach**             |                 **Output**                    |

|---------------------- |----------------------------------------|-------------------------------------------|

| **ROI Detection**         | **YOLOv8n (fine‑tuned on custom dataset)** | **Bounding box + class (conjunctiva / nail)** |

| **Anemia Classification** | **MobileNetV2 (transfer learning)**        | **Binary: Anemic / Non‑Anemic**               |

| **Image Type Classifier** | **MobileNetV2 (binary)**                   | **Conjunctiva / Nail (fallback)**             |

| **Explainability**        | **Grad‑CAM (last convolutional layer)**    | **Heatmap overlay on the ROI**                |



📁 **Project Structure**



**AnemoScan/**

**├── App/**

**│ ├── app.py # Main Streamlit application**

**│ └── logo.png # Sidebar logo**

**├── Data/**

**│ ├── Conjunctiva/ # Raw \& split images (Anemic / Non‑Anemic)**

**│ ├── Nails/ # Raw \& split images (Anemic / Non‑Anemic)**

**│ ├── Type\_Classifier/ # Combined dataset for type classifier**

**│ └── Detection/ # YOLO dataset + training results**

**├── Models/**

**│ ├── best.pt # YOLOv8 trained weights**

**│ ├── best\_initial\_model\_conjunctiva.h5**

**│ ├── best\_initial\_model\_nails.h5**

**│ └── type\_classifier\_model.h5**

**├── Notebooks/**

**│ ├── 01\_Data\_Preparation\_Conjunctiva.ipynb**

**│ ├── 01\_Data\_Preparation\_Nails.ipynb**

**│ ├── 02\_Model\_Conjunctiva.ipynb**

**│ ├── 02\_Model\_Nails.ipynb**

**│ ├── 03\_XAI\_Analysis.ipynb**

**│ ├── 04\_Type\_Classifier.ipynb**

**│ └── 05\_ROI\_Detection\_YOLO.ipynb**

**├── Results/ # Saved plots, metrics, confusion matrices**

**├── Utils/**

**│ ├── init.py**

**│ ├── gradcam.py # Grad‑CAM functions**

**│ ├── nutrition.py # Nutrition advice generator**

**│ ├── pdf\_generator.py # PDF report creation (reportlab)**

**│ ├── preprocess.py # Image cropping \& normalization**

**│ └── roi\_detection.py # YOLO detector wrapper**

**├── README.md**

**└── requirements.txt**





🚀 **Installation \& Setup**



1\. **Clone the repository**

bash:

**git clone https://github.com/your-username/AnemoScan.git**

**cd AnemoScan/App**





**2. Create a virtual environment (recommended)**

bash:

**python -m venv venv**

**source venv/bin/activate      # Linux / Mac**

**venv\\Scripts\\activate         # Windows**





**3. Install dependencies**

bash:

**pip install -r ../requirements.txt**





**4. Download the pre-trained models**

Place the following files inside the **Models/** folder **(relative to App/):**



* **best.pt (YOLOv8)**
* **best\_initial\_model\_conjunctiva.h5**
* **best\_initial\_model\_nails.h5**
* **type\_classifier\_model.h5**



**Note:**

* **Models are not included in the repository due to size limits.**
* **You can retrain them using the notebooks, or request the pre‑trained weights from the author.**





**5. Run the Streamlit App**

bash:

**streamlit run app.py**



**Note:**

* **The app will open in your default browser at http://localhost:8501.**



🧪 **How to Use**



1. **Upload an image** – JPG, JPEG, or PNG (clear, well‑lit conjunctiva or nail).



**2. Wait for YOLO detection** – The system will crop the region of interest and show it in an expander.



**3. Click “Predict”** – The anemia classifier runs on the cropped ROI.



**4. Explore results** –



* **Results tab** – diagnosis, confidence bars.
* **Grad‑CAM tab** – heatmap overlay, downloadable.
* **Nutrition tab** – dietary recommendations.
* **Download tab** – full PDF report (including images and advice).



📊 **Performance (Test Set)**



|          **Model**                 |           **Accuracy**             |       **mAP50-95 (YOLO)**       |         **AUC**         |

|--------------------------------|--------------------------------|-----------------------------|---------------------|

| **Conjunctiva classifier**         |            **70.00%**              |               **-**             |       **78.60%**        |

| **Nails classifier**               |            **70.00%**              |               **-**             |       **75.20%**        |

| **YOLOv8 ROI detection**           |              **-**                 |             **76.12%**          |       **100.0%**        |

| **Type classifier (Conj. / Nail)** |            **99.80%**              |               **-**             |       **100.0%**        |



**Note:**

* **Detailed metrics are available in the Results/ folder after training.**



**🔧 Customization \& Retraining**



All training notebooks are provided in the **Notebooks/** directory:



**01\_\* – data preparation, renaming, splitting, augmentation.**



**02\_\* – MobileNetV2 training (initial + fine‑tuning).**



**03\_XAI\_Analysis – Grad‑CAM visualization on test samples.**



**04\_Type\_Classifier – training the auxiliary type classifier.**



**05\_ROI\_Detection\_YOLO – YOLOv8 fine‑tuning, evaluation, export.**



**Note:**

* **Update the paths in the notebooks to match your dataset location before running.**



**📜 License \& Disclaimer**



* **This project is for educational and research purposes only. It is not a medical device and must not replace professional medical consultation. The authors assume no liability for misuse or misinterpretation of the results.**



**👥 Contributors**

&#x20;                        **Eng. Abdulrhman Osama – Final Year Project, TM471, supervised by Dr. Ramadan Babers.**



**🙏 Acknowledgements**



* **Ultralytics YOLOv8**
* **TensorFlow / Keras**
* **Streamlit**
* **OpenCV**
* **ReportLab**



&#x20;                                                **Made with ❤️ for accessible healthcare AI.**

