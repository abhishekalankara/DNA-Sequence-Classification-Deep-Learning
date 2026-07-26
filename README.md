# DNA Sequence Classification using Deep Learning

A deep learning framework for multi-class DNA sequence classification using **CNN, DeepSEA, DeepBind, CNN-LSTM, and CNN-Random Forest** architectures. This project compares multiple deep learning models for genomic disease prediction, with the **CNN-LSTM hybrid model achieving the highest classification accuracy of 94%**.

---

## Table of Contents

- Overview
- Research Paper
- Project Highlights
- Dataset
- Project Workflow
- Deep Learning Models
- Results
- Performance Evaluation
- Technologies Used
- Repository Structure
- Applications
- Future Work
- Author
- Acknowledgements
- License

---

## Overview

DNA sequence classification is an important application of artificial intelligence in bioinformatics, enabling automated identification of disease-associated genomic patterns.

This project presents a comparative study of multiple deep learning architectures for multi-class DNA sequence classification.

The workflow includes:

- DNA sequence preprocessing
- One-Hot Encoding
- Model development
- Model training
- Performance evaluation
- Result visualization

Five deep learning architectures were implemented and compared:

- 1D CNN
- DeepSEA
- DeepBind
- CNN-LSTM
- CNN-Random Forest

Among these models, the **CNN-LSTM hybrid model achieved the highest classification accuracy of 94%**, demonstrating its ability to capture both local sequence features and long-range dependencies in genomic data.

---

## Research Paper

This project is associated with the research paper:

**Convolution Meets the Genome: Advancing and Evaluating Deep Learning Models**

Presented at the **International Conference on Machine Learning and Data Engineering (ICMLDE 2025).**

---

## Project Highlights

- Multi-class DNA sequence classification
- Disease prediction using genomic sequences
- One-Hot Encoding for DNA representation
- Comparative analysis of five deep learning architectures
- CNN-LSTM Hybrid Model
- CNN-Random Forest Hybrid Model
- DeepSEA Architecture
- DeepBind Architecture
- Performance comparison across multiple models
- Model evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix
- Training and validation visualization

---

## Dataset

The project uses a DNA sequence dataset containing genomic sequences belonging to five disease categories.

### Disease Categories

| Class | Disease |
|------:|---------------------------|
| 0 | Breast Cancer |
| 1 | Alzheimer's Disease |
| 2 | Diabetes Type 2 |
| 3 | Colorectal Cancer |
| 4 | Hepatitis B Infection |

Each DNA sequence consists of nucleotide bases:

- A
- T
- G
- C

### Data Preprocessing

- Missing value removal
- Sequence padding
- One-Hot Encoding
- Train-Test Split
- Class balancing
- Data normalization

---

## Project Workflow

```
DNA Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Sequence Padding
      │
      ▼
One-Hot Encoding
      │
      ▼
Model Development
      │
      ▼
Model Training
      │
      ▼
Performance Evaluation
      │
      ▼
Result Visualization
```

---

## Deep Learning Models

### 1. 1D CNN

A lightweight convolutional neural network that extracts local sequence features using Conv1D layers followed by pooling and dense layers.

---

### 2. DeepSEA

A deep convolutional architecture inspired by genomic sequence learning that captures hierarchical biological features.

---

### 3. DeepBind

A convolution-based model designed for sequence motif detection using multiple convolution filters.

---

### 4. CNN-LSTM

The best-performing model in this study.

#### Architecture

```
Input
   │
Conv1D
   │
MaxPooling
   │
Bidirectional LSTM
   │
Dropout
   │
Dense Layer
   │
Softmax
```

#### Advantages

- Learns local sequence features
- Captures long-term dependencies
- Improved classification performance
- Highest overall accuracy

---

### 5. CNN-Random Forest

A hybrid approach where CNN extracts deep features while Random Forest performs the final classification.

---

# Results

## Model Performance

| Model | Accuracy |
|----------------------|----------|
| CNN-LSTM | **94.0%** |
| CNN-Random Forest | **92.0%** |
| DeepBind | **88.3%** |
| DeepSEA | **85.2%** |
| 1D CNN | **85.0%** |

---

## Training Performance

The CNN-LSTM model was trained using TensorFlow/Keras with the Adam optimizer. Training performance was monitored using training accuracy, validation accuracy, training loss, and validation loss across multiple epochs.

![Training Results](results/training_results.png)

---

## Confusion Matrix

The confusion matrix illustrates the classification performance of the CNN-LSTM model across all disease classes.

![Confusion Matrix](results/confusion_matrix.png)

---

## Classification Metrics

The following visualization presents Precision, Recall, F1-Score, and Support for each disease class.

![Classification Metrics](results/classification_metrics.png)

---

## Performance Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Support
- Confusion Matrix
- Classification Report

The CNN-LSTM model demonstrated the best overall performance with balanced precision, recall, and F1-score across all disease classes.

---

## Technologies Used

### Programming Language

- Python

### Frameworks

- TensorFlow
- Keras

### Libraries

- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

### Development Environment

- Google Colab
- Jupyter Notebook

---

## Repository Structure

```
DNA-Sequence-Classification-Deep-Learning/
│
├── results/
│   ├── classification_metrics.png
│   ├── confusion_matrix.png
│   └── training_results.png
│
├── src/
│   ├── preprocessing.py
│   ├── cnn_model.py
│   ├── deepsea_model.py
│   ├── deepbind_model.py
│   ├── cnn_lstm_model.py
│   ├── cnn_random_forest.py
│   ├── evaluation.py
│   └── train.py
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Applications

- Disease Prediction
- Bioinformatics
- Genomic Data Analysis
- Precision Medicine
- Healthcare Artificial Intelligence
- Biomedical Research
- DNA Sequence Classification

---

## Future Work

- Transformer-based DNA models (DNABERT)
- Explainable AI (XAI)
- Attention-based sequence models
- Larger genomic datasets
- Multi-omics data integration
- Real-time genomic disease prediction

---

## References

- TensorFlow
- Keras
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## Author

**Abhishek Alankara**

B.Tech – Electronics and Communication Engineering

SRM University-AP

**LinkedIn:** https://www.linkedin.com/in/abhishekalankara/

**GitHub:** https://github.com/abhishekalankara

---

## Acknowledgements

I would like to express my sincere gratitude to the faculty members and mentors of the Department of Electronics and Communication Engineering, SRM University-AP, for their continuous guidance and support throughout this project. I also acknowledge the open-source community and the developers of TensorFlow, Keras, Scikit-learn, NumPy, Pandas, Matplotlib, and Seaborn for providing the tools that made this work possible.

---

## License

This project is licensed under the **MIT License**.

---

If you found this project useful, consider giving this repository a **Star**.
