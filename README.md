# breast-cancer-classification
Breast Cancer Diagnosis Prediction using K-Nearest Neighbors (KNN) and K-Means Clustering 

## Objective
The goal of this project is to classify breast cancer diagnoses into two categories — malignant (M) and benign (B) — using machine learning techniques, specifically K-Nearest Neighbors (KNN) and K-Means clustering.

## Dataset
The dataset contains various features extracted from breast cancer biopsy images. The target column is `diagnosis`, where:
- M = Malignant  
- B = Benign

## Data Preprocessing
Before training the models, the following preprocessing steps were applied:
1. Removed unnecessary columns: `id` and `Unnamed: 32`.  
2. Encoded the diagnosis column: M → 1, B → 0.  
3. Scaled all features using Min-Max normalization to bring values into the range [0, 1].  
4. Split the dataset into training (80%) and testing (20%) sets.

## Methods

### K-Means Clustering
- Applied K-Means clustering with 2 clusters to see how the data naturally groups.  
- Compared the clusters with the actual diagnosis labels to get an idea of how well the data separates.

### K-Nearest Neighbors (KNN)
- Trained a KNN classifier with k=5 on the training data.  
- Evaluated the model using standard metrics: Accuracy, Precision, Recall, and F1-score.

## Results
After training and testing the KNN model, we achieved the following metrics on the test set:

- **Accuracy:** 96.5%  
- **Precision:** 95.3%  
- **Recall:** 95.3%  
- **F1-score:** 95.3%  

These results indicate that the KNN model performs very well in distinguishing between malignant and benign cases.

## Conclusion
- The KNN classifier achieved high accuracy and balanced precision and recall, making it a reliable choice for this dataset.  
- K-Means clustering roughly separated the two classes, but KNN provided more precise classification results.

## How to Run
1. Open the notebook `breast_cancer_knn_kmeans.ipynb` in Google Colab or Jupyter Notebook.  
2. Run all cells in order to reproduce the preprocessing, K-Means clustering, KNN training, and evaluation results.
