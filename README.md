
# Grapevine Leaf Image Analysis using PCA, Clustering, and CNN

**Author:** Muskan Dhingra  
**Course:** Applied Machine Learning  
**University:** Indiana University Bloomington  

---

## Dataset
**Dataset:** *Grapevine Leaves Image Dataset*  
**Classes:** 4 leaf categories (A, B, D, N), identified by the first letter of each image filename.  
**Format:** `.jpg` or `.png` images stored in folders, loaded and preprocessed using OpenCV or PIL.

---

##  Assignment Overview

This project explores **dimensionality reduction**, **visualization**, **clustering**, and **deep learning** techniques on a grapevine leaf image dataset. The tasks include:

1. **PCA for Dimensionality Reduction**
2. **Reconstruction of Images After PCA**
3. **2D Visualizations with PCA, t-SNE, LLE, MDS**
4. **K-Means and EM Clustering**
5. **Synthetic Image Generation using GMM**
6. **Feedforward Neural Network Classification**

---

## Task Breakdown

### 1. PCA Analysis
- PCA applied to reduce dimensions while preserving **95% of variance**.
- **89 components** were needed to retain 95% of the information.
- Cumulative explained variance plotted to choose optimal `n_components`.

### 2.  Image Reconstruction
- Top 10 original grayscale images visualized along with their PCA reconstructions.
- The results showed high-quality reconstructions despite reduced dimensionality.

### 3. 2D Visualizations
- Applied PCA, t-SNE, LLE, and MDS to project high-dimensional images into 2D.
- Visualizations were color-coded based on image categories.
- Added image thumbnails directly on scatter plots for interpretability.
- **Preferred Visualization:** t-SNE showed best clustering with non-linear separability.

### 4. K-Means Clustering
- PCA reduced data to 2D before clustering.
- Elbow and Silhouette methods used to determine optimal number of clusters.
- **Optimal Clusters:** 3 or 4 depending on the method.
- Clustering visualized with **decision boundaries** and **label annotations**.

### 5. EM Clustering (Gaussian Mixture Models)
- EM performed on PCA-reduced data.
- Optimal number of components determined using **Bayesian Information Criterion (BIC)**.
- Probabilistic clusters visualized with Gaussian decision boundaries.

### 6. Leaf Generation with GMM
- Used `gmm.sample(20)` to synthesize new leaf images.
- Reconstructed from PCA space to image space with `inverse_transform()`.
- Images visualized after reshaping to original (20x20) format.

### 7. Feedforward Neural Network (CNN)
- CNN trained using Keras with 1 Conv2D and 2 Dense layers.
- Trained on 64x64 RGB images.
- **Training Accuracy:** ~93.28%  
- **Validation Accuracy:** ~43.5%  
- **Total Parameters:** 1,969,348  
- **Bias Parameters:** 100  
- **Training Time:** ~23.2 seconds (20 epochs)

### Training Curves
- Plots for training & validation loss/accuracy across epochs were generated.
- Demonstrated overfitting (high training accuracy, moderate validation accuracy).

---

## Model Summary (Keras)

```
Total Parameters: 1,969,348  
Trainable Parameters: 1,969,348  
Bias Parameters: 100
```

---

##  Dependencies

```bash
numpy
matplotlib
opencv-python
scikit-learn
scikit-image
tensorflow
PIL (Pillow)
```

---

##  How to Run

1. Place dataset under correct paths:  
   `Grapevine_Leaves_Image_Dataset/train` and `test`

2. Run the `.py` script in a Python environment or Jupyter Notebook.

---

##  References

- Kaggle  
- *Hands-On Machine Learning* by Aurélien Géron  
- Official [GitHub Repo](https://github.com/ageron/handson-ml3/blob/main/08_dimensionality_reduction.ipynb)  
- Course material and Colab notebooks
