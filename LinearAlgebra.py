# Unit-I Linear Algebra: Introduction to Vector space, Subspace, Linear Independence andDependence, Basis and Dimensions, Convex set, Rank of a matrix, System of linear equations,Orthogonal bases, Projection, Gram-Schmidt orthogonality process, Linear Mappings, Kernel andImage space of a linear map, Matrix associated with linear map, Eigen values and Eigen vectors,PCA, SVD, Applications in Data Reduction, Text Analysis and Image Processing.

pip install numpy scipy sympy matplotlib scikit-learn

import numpy as np
import sympy as sp
from sympy import Matrix
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
import requests
from io import BytesIO

print("=== Unit-I: Linear Algebra in Python ===")

# 1. Vector Space and Subspace
print("\n--- Vector Space and Subspace ---")
V = np.array([[1, 2], [2, 4]])
print("Vectors:\n", V)

# Check if second is scalar multiple of first
print("Linearly Dependent:", np.linalg.matrix_rank(V) < V.shape[0])

# 2. Basis and Dimension
print("\n--- Basis and Dimension ---")
A = np.array([[1, 2], [3, 6]])
rank = np.linalg.matrix_rank(A)
print("Rank (dimension of column space):", rank)

# 3. Convex Set (2 points and all points between them)
print("\n--- Convex Set ---")
x = np.array([1, 2])
y = np.array([4, 6])
for alpha in np.linspace(0, 1, 5):
    z = alpha * x + (1 - alpha) * y
    print(f"Point for alpha={alpha:.2f}:", z)

# 4. System of Linear Equations
print("\n--- System of Linear Equations ---")
A = np.array([[2, 1], [1, 3]])
b = np.array([8, 13])
x = np.linalg.solve(A, b)
print("Solution to Ax = b:", x)

# 5. Orthogonal Bases and Projection
print("\n--- Orthogonal Projection ---")
u = np.array([3, 4])
v = np.array([4, 0])
proj = np.dot(u, v) / np.dot(v, v) * v
print("Projection of u on v:", proj)

# 6. Gram-Schmidt Process
print("\n--- Gram-Schmidt Orthogonalization ---")
def gram_schmidt(X):
    Q = []
    for x in X:
        for q in Q:
            x = x - np.dot(x, q) * q
        x = x / np.linalg.norm(x)
        Q.append(x)
    return np.array(Q)

X = np.array([[3, 1], [2, 2]], dtype=float)
Q = gram_schmidt(X)
print("Orthogonal basis Q:\n", Q)

# 7. Linear Map, Kernel, Image
print("\n--- Linear Map, Kernel and Image ---")
M = Matrix([[1, 2], [2, 4]])
ker = M.nullspace()
print("Kernel (nullspace):", ker)
img = M.columnspace()
print("Image space (column space):", img)

# 8. Matrix Associated with Linear Map
print("\nMatrix of linear map f(x) = Ax")
A = np.array([[1, 2], [3, 4]])
print("Matrix A:\n", A)

# 9. Eigenvalues and Eigenvectors
print("\n--- Eigenvalues and Eigenvectors ---")
eigvals, eigvecs = np.linalg.eig(A)
print("Eigenvalues:", eigvals)
print("Eigenvectors:\n", eigvecs)

# 10. PCA
print("\n--- PCA ---")
X = np.random.rand(100, 5)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print("Reduced Data Shape (PCA):", X_pca.shape)

# 11. SVD
print("\n--- SVD ---")
U, S, VT = np.linalg.svd(A)
print("U:\n", U)
print("S:", S)
print("VT:\n", VT)

# 12. Text Analysis using PCA
print("\n--- Text Analysis (TF-IDF + PCA) ---")
docs = ["data science is amazing", "machine learning is part of data science", "text analysis with vector space"]
tfidf = TfidfVectorizer()
X_text = tfidf.fit_transform(docs).toarray()
pca_text = PCA(n_components=2)
reduced_text = pca_text.fit_transform(X_text)
print("TF-IDF reduced with PCA:", reduced_text)

# 13. Image Compression using SVD
print("\n--- Image Processing using SVD ---")
# Load sample image (grayscale)
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/24/Lenna.png/256px-Lenna.png"
img = Image.open(BytesIO(requests.get(url).content)).convert("L")
img_arr = np.array(img)

# Perform SVD
U, S, VT = np.linalg.svd(img_arr, full_matrices=False)

# Keep only top-k singular values
k = 50
compressed = (U[:, :k] @ np.diag(S[:k]) @ VT[:k, :])

# Show original and compressed image
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img_arr, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f"SVD Compressed (k={k})")
plt.imshow(compressed, cmap='gray')
plt.axis('off')
plt.show()
