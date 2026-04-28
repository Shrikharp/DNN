# Alzheimer’s Disease Diagnosis using Graph Neural Networks (GCN vs GAT)

## Project Overview

This project focuses on detecting Alzheimer’s Disease using machine learning, specifically Graph Neural Networks (GNNs). The goal is to compare two models:

* Graph Convolutional Network (GCN) – baseline model
* Graph Attention Network (GAT) – improved model with attention

The project is inspired by research on interpretable AI models for medical diagnosis.

---

## Objectives

* Detect Alzheimer’s Disease using structured data
* Convert tabular data into graph format
* Compare GCN and GAT performance
* Analyze improvements in accuracy and interpretability

---

## Dataset

We use the OASIS Cross-Sectional Dataset.

File used:
data/oasis_cross-sectional.csv

The dataset includes:

* Demographic details (age, gender)
* Clinical measurements
* Cognitive test scores

---

## Project Structure

project/
│
├── main.py
├── requirements.txt
│
├── src/
│   ├── data_loader.py
│   ├── graph_builder.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
│
└── data/
└── oasis_cross-sectional.csv

---

## Installation & Setup

1. Clone the repository:
   git clone <your-repo-link>
   cd <repo-name>

2. Create virtual environment:
   python3 -m venv .venv
   source .venv/bin/activate

3. Install dependencies:
   pip install -r requirements.txt

4. Run the project:
   python main.py

---

## Methodology

Step 1: Data Preprocessing

* Clean dataset
* Handle missing values
* Normalize features

Step 2: Graph Construction

* Each row → Node
* Similarity between rows → Edge
* Cosine similarity used

Step 3: Model Training

* Train GCN model
* Train GAT model
* Use cross-entropy loss

Step 4: Evaluation

* Measure accuracy
* Compare models

---

## Results

GCN Accuracy: 92.05%
GAT Accuracy: 93.18%

Observations:

* GAT performs better than GCN
* Attention mechanism improves learning
* Both models perform well for classification

---

## Visualizations

The project generates:

* Loss vs Epoch graph
* Accuracy comparison graph

These help understand:

* Training performance
* Model comparison

---

## Comparison (Important)

GCN:

* Simpler model
* Faster training
* Less interpretable

GAT:

* Uses attention mechanism
* Better performance
* More interpretable

---

## Research Connection

Paper 1 (C2C-7):

* Uses GCN for Alzheimer’s diagnosis

Paper 2 (C2C-8):

* Uses GKAN (Graph + Kolmogorov-Arnold Networks)
* Focus on interpretability

Our work:

* Implements GCN baseline
* Uses GAT as an improved model
* Demonstrates better accuracy

---

## Limitations

* Small dataset size
* Graph threshold sensitivity
* GKAN not fully implemented

---

## Future Work

* Implement full GKAN model
* Use larger datasets (ADNI)
* Add MRI/fMRI data
* Improve interpretability

---

## Technologies Used

* Python
* PyTorch
* PyTorch Geometric
* NumPy
* Pandas
* Matplotlib

---

## Contributors

* Your Name
* Teammate Name

---

## License

This project is for academic use only.

---