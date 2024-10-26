# Handwritten Character Recognition Project

This project implements a handwritten character recognition system using Support Vector Machines (SVM) and Random Forest classifiers. It includes dimensionality reduction via PCA and out-of-distribution detection capabilities. The system was tested on a dataset of handwritten digits (0-9) and letters (A-F).

This was a school project for the course [Inf264](https://www4.uib.no/emner/INF264)  focused on machine learning classifier performance and handling real-world challenges like class imbalance and corrupt data.

## How to Run the Code

Ensure you have Python installed. Install required dependencies:

```bash
pip install -r requirements.txt
```

To reproduce the experiments:
1. Run the classification experiments in `Problem 1`
2. Test dimensionality reduction in `Problem 2`
3. Evaluate out-of-distribution detection in `Problem 3`

Note: Set the random seed to 0 to reproduce the exact results mentioned in the report.

## Report Highlights

- Comprehensive analysis of handwritten character dataset
- Implementation of SMOTE for handling class imbalance
- Comparison between SVM and Random Forest classifiers
- PCA dimensionality reduction study (10-200 components)
- Novel approach to detecting out-of-distribution images
- Performance optimization reducing computation time from 13min to 43s

### Key Results

- Final classifier: SVM with PCA (100 components)
  - 93% weighted F1-score
  - Perfect detection of empty labels
  - Significantly improved minority class performance
- Successfully identified 75/85 corrupt images

[View the full report (PDF)](Report.pdf) for detailed methodology and results analysis.

## Authors
* Elias Hovdenes
* Magnus Brørby
