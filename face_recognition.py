import os
import cv2
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# ==========================================
# 1. Dataset Loading and Preprocessing
# ==========================================
def load_dataset(base_path, target_size=(64, 64)):
    """
    Loads face images from subdirectories in base_path, converts them to grayscale,
    and resizes them to target_size.
    """
    face_db = []
    labels = []
    class_names = []
    
    # List all subdirectories (classes/identities)
    subdirs = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    
    for label_idx, subdir in enumerate(subdirs):
        class_names.append(subdir)
        subdir_path = os.path.join(base_path, subdir)
        
        # Read all images in this subdirectory
        for img_name in os.listdir(subdir_path):
            img_path = os.path.join(subdir_path, img_name)
            # Read in grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            # Resize image to target dimensions (m x n)
            img_resized = cv2.resize(img, target_size)
            
            # Flatten image to a vector of size (m*n) and normalize to [0, 1]
            img_vector = img_resized.flatten().astype(np.float32) / 255.0
            
            face_db.append(img_vector)
            labels.append(label_idx)
            
    return np.array(face_db).T, np.array(labels), class_names  # shape: (mn, p) for face_db

# ==========================================
# 2. PCA (Eigenfaces) Module
# ==========================================
class EigenfacePCA:
    def __init__(self, k=10):
        self.k = k
        self.mean_face = None
        self.eigenfaces = None
        self.signatures = None
        self.eigenvalues = None
        self.eigenvectors = None
        
    def fit(self, face_db):
        """
        Fits PCA on training database face_db of shape (mn, p)
        """
        mn, p = face_db.shape
        self.k = min(self.k, p)
        
        # Step 2: Mean Calculation
        self.mean_face = np.mean(face_db, axis=1, keepdims=True)  # (mn, 1)
        
        # Step 3: Mean Zero (centering)
        A = face_db - self.mean_face  # (mn, p)
        
        # Step 4: Surrogate Covariance Matrix C = A^T * A
        C = A.T @ A  # (p, p)
        
        # Step 5: Eigenvalue and Eigenvector decomposition
        # scipy.linalg.eigh is optimized for symmetric matrices
        eigenvalues, eigenvectors = scipy.linalg.eigh(C)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[idx]
        self.eigenvectors = eigenvectors[:, idx]
        
        # Step 6: Select the top k directions (Feature vectors)
        V_k = self.eigenvectors[:, :self.k]  # (p, k)
        
        # Step 7: Generating Eigenfaces
        # Eigenfaces = V_k^T * A^T
        self.eigenfaces = V_k.T @ A.T  # (k, mn)
        
        # Normalize eigenfaces to have unit norm
        norms = np.linalg.norm(self.eigenfaces, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        self.eigenfaces = self.eigenfaces / norms
        
        # Step 8: Generate Signature of Each Training Face
        self.signatures = self.eigenfaces @ A  # (k, p)
        
        return self.signatures
    
    def project(self, face_img):
        """
        Projects a test face_img of shape (mn, 1) or (mn, num_imgs) to face space.
        """
        # Step 2: Subtract Mean
        face_aligned = face_img - self.mean_face
        # Step 3: Project
        projected = self.eigenfaces @ face_aligned  # (k, num_imgs)
        return projected

    def reconstruct(self, projected_signatures):
        """
        Reconstructs mean-aligned faces from projected signatures.
        Shape: (k, num_imgs) -> (mn, num_imgs)
        """
        # I_rec = U^T * Omega_test + Mean
        return self.eigenfaces.T @ projected_signatures + self.mean_face

# ==========================================
# 3. Custom NumPy ANN (Backpropagation)
# ==========================================
class NumPyANN:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=1e-4):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.wd = weight_decay
        
        # He (Kaiming) initialization for weights
        self.W1 = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((hidden_dim, 1))
        self.W2 = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((output_dim, 1))
        
        # Adam optimizer states (first and second moments)
        self.m_W1, self.v_W1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        self.m_b1, self.v_b1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        self.m_W2, self.v_W2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        self.m_b2, self.v_b2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        
        self.t = 0  # time step for bias correction

    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, x * alpha)
        
    def leaky_relu_derivative(self, x, alpha=0.01):
        dx = np.ones_like(x)
        dx[x <= 0] = alpha
        return dx
        
    def softmax(self, x):
        # Numeric stability shift
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
        
    def forward(self, X):
        """
        X shape: (input_dim, batch_size)
        """
        self.Z1 = self.W1 @ X + self.b1
        self.A1 = self.leaky_relu(self.Z1)
        self.Z2 = self.W2 @ self.A1 + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2
        
    def backward(self, X, Y, A2):
        """
        Y shape: (output_dim, batch_size) - one-hot encoded targets
        """
        batch_size = X.shape[1]
        
        # Output layer error
        dZ2 = A2 - Y  # (output_dim, batch_size)
        # Add L2 regularization gradient
        dW2 = (dZ2 @ self.A1.T) / batch_size + self.wd * self.W2
        db2 = np.sum(dZ2, axis=1, keepdims=True) / batch_size
        
        # Hidden layer error
        dZ1 = (self.W2.T @ dZ2) * self.leaky_relu_derivative(self.Z1)
        dW1 = (dZ1 @ X.T) / batch_size + self.wd * self.W1
        db1 = np.sum(dZ1, axis=1, keepdims=True) / batch_size
        
        return dW1, db1, dW2, db2
        
    def train(self, X, Y, epochs=1000, batch_size=32):
        """
        X: (input_dim, n_samples)
        Y: (output_dim, n_samples) - one-hot encoded
        """
        n_samples = X.shape[1]
        
        for epoch in range(epochs):
            # Shuffle training data
            permutation = np.random.permutation(n_samples)
            X_shuffled = X[:, permutation]
            Y_shuffled = Y[:, permutation]
            
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[:, i:i+batch_size]
                Y_batch = Y_shuffled[:, i:i+batch_size]
                
                # Forward Pass
                A2 = self.forward(X_batch)
                
                # Backward Pass
                dW1, db1, dW2, db2 = self.backward(X_batch, Y_batch, A2)
                
                # Increment time step
                self.t += 1
                
                # Update weights and biases using Adam optimizer
                # Update W1
                self.m_W1 = self.beta1 * self.m_W1 + (1.0 - self.beta1) * dW1
                self.v_W1 = self.beta2 * self.v_W1 + (1.0 - self.beta2) * (dW1 ** 2)
                m_W1_corrected = self.m_W1 / (1.0 - self.beta1 ** self.t)
                v_W1_corrected = self.v_W1 / (1.0 - self.beta2 ** self.t)
                self.W1 -= self.lr * m_W1_corrected / (np.sqrt(v_W1_corrected) + self.epsilon)
                
                # Update b1
                self.m_b1 = self.beta1 * self.m_b1 + (1.0 - self.beta1) * db1
                self.v_b1 = self.beta2 * self.v_b1 + (1.0 - self.beta2) * (db1 ** 2)
                m_b1_corrected = self.m_b1 / (1.0 - self.beta1 ** self.t)
                v_b1_corrected = self.v_b1 / (1.0 - self.beta2 ** self.t)
                self.b1 -= self.lr * m_b1_corrected / (np.sqrt(v_b1_corrected) + self.epsilon)
                
                # Update W2
                self.m_W2 = self.beta1 * self.m_W2 + (1.0 - self.beta1) * dW2
                self.v_W2 = self.beta2 * self.v_W2 + (1.0 - self.beta2) * (dW2 ** 2)
                m_W2_corrected = self.m_W2 / (1.0 - self.beta1 ** self.t)
                v_W2_corrected = self.v_W2 / (1.0 - self.beta2 ** self.t)
                self.W2 -= self.lr * m_W2_corrected / (np.sqrt(v_W2_corrected) + self.epsilon)
                
                # Update b2
                self.m_b2 = self.beta1 * self.m_b2 + (1.0 - self.beta1) * db2
                self.v_b2 = self.beta2 * self.v_b2 + (1.0 - self.beta2) * (db2 ** 2)
                m_b2_corrected = self.m_b2 / (1.0 - self.beta1 ** self.t)
                v_b2_corrected = self.v_b2 / (1.0 - self.beta2 ** self.t)
                self.b2 -= self.lr * m_b2_corrected / (np.sqrt(v_b2_corrected) + self.epsilon)
                
    def predict(self, X):
        probs = self.forward(X)
        preds = np.argmax(probs, axis=0)
        return preds, probs

# Helper to one-hot encode
def to_one_hot(labels, num_classes):
    one_hot = np.zeros((num_classes, len(labels)))
    for idx, val in enumerate(labels):
        one_hot[val, idx] = 1.0
    return one_hot

# ==========================================
# 4. Main Evaluation Pipeline
# ==========================================
def main():
    dataset_path = "dataset/dataset/faces"
    print("="*60)
    print("PCA + ANN Face Recognition Pipeline")
    print("="*60)
    
    # Load all face vectors
    print("Loading face dataset...")
    face_db, labels, class_names = load_dataset(dataset_path, target_size=(64, 64))
    mn, p_total = face_db.shape
    print(f"Total loaded images: {p_total}")
    print(f"Image flattened dimensions (mn): {mn} ({int(np.sqrt(mn))}x{int(np.sqrt(mn))})")
    print(f"Classes found: {class_names}")
    
    # Determine Enrolled vs. Imposter Subjects
    # We will exclude some classes to serve as imposters (who are not in the training set)
    # Let's say Farhan and Ileana are imposters, others are enrolled
    imposter_names = ["Farhan", "Ileana"]
    enrolled_names = [name for name in class_names if name not in imposter_names]
    
    print(f"\nEnrolled Subjects ({len(enrolled_names)}): {enrolled_names}")
    print(f"Imposter Subjects ({len(imposter_names)}): {imposter_names}")
    
    # Map old class indices to names, and create new enrolled class index mapping
    enrolled_indices = [class_names.index(name) for name in enrolled_names]
    imposter_indices = [class_names.index(name) for name in imposter_names]
    
    new_label_map = {old_idx: new_idx for new_idx, old_idx in enumerate(enrolled_indices)}
    
    # Separate data into Enrolled and Imposter
    enrolled_mask = np.isin(labels, enrolled_indices)
    imposter_mask = np.isin(labels, imposter_indices)
    
    enrolled_db = face_db[:, enrolled_mask]
    enrolled_labels = labels[enrolled_mask]
    # Map enrolled labels to consecutive 0..C-1
    enrolled_labels_mapped = np.array([new_label_map[l] for l in enrolled_labels])
    
    imposter_db = face_db[:, imposter_mask]
    
    # Split enrolled data: 60% train, 40% test
    # We do a stratified split so that each enrolled person has 60% training images
    num_enrolled_classes = len(enrolled_names)
    train_indices = []
    test_indices = []
    
    for c in range(num_enrolled_classes):
        class_mask = np.where(enrolled_labels_mapped == c)[0]
        # Shuffle class indices
        np.random.shuffle(class_mask)
        split_point = int(0.6 * len(class_mask))
        train_indices.extend(class_mask[:split_point])
        test_indices.extend(class_mask[split_point:])
        
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    
    train_db = enrolled_db[:, train_indices]
    train_labels = enrolled_labels_mapped[train_indices]
    
    test_db = enrolled_db[:, test_indices]
    test_labels = enrolled_labels_mapped[test_indices]
    
    print(f"\nTrain set shape: {train_db.shape} (enrolled faces only)")
    print(f"Test set (enrolled) shape: {test_db.shape}")
    print(f"Test set (imposters) shape: {imposter_db.shape}")
    
    # We will sweep different k values and track:
    # 1. Classification accuracy on Enrolled test set
    # 2. Imposter detection accuracy on Imposter test set
    k_values = [2, 5, 10, 15, 20, 30, 40, 50, 75, 100]
    enrolled_accuracies = []
    imposter_detection_rates = []
    overall_accuracies = []
    
    for k in k_values:
        print(f"\n--- Running Experiment for k = {k} ---")
        
        # 1. Fit PCA on training database
        pca = EigenfacePCA(k=k)
        train_signatures_raw = pca.fit(train_db)  # (k, p_train)
        
        # Standardize signatures for ANN to prevent gradient explosion/saturation
        sig_mean = np.mean(train_signatures_raw, axis=1, keepdims=True)
        sig_std = np.std(train_signatures_raw, axis=1, keepdims=True) + 1e-8
        train_signatures = (train_signatures_raw - sig_mean) / sig_std
        
        # Convert training labels to one-hot for ANN
        Y_train_onehot = to_one_hot(train_labels, num_enrolled_classes)
        
        # 2. Initialize and train NumPy ANN
        # Input features: k, Hidden: 128, Output: num_enrolled_classes
        ann = NumPyANN(input_dim=k, hidden_dim=128, output_dim=num_enrolled_classes, learning_rate=0.003)
        ann.train(train_signatures, Y_train_onehot, epochs=1500, batch_size=16)
        
        # 3. Test on Enrolled Test Set
        test_signatures_raw = pca.project(test_db)
        test_signatures = (test_signatures_raw - sig_mean) / sig_std
        test_preds, test_probs = ann.predict(test_signatures)
        
        # Calculate standard classification accuracy on enrolled test set
        enrolled_acc = np.mean(test_preds == test_labels) * 100
        enrolled_accuracies.append(enrolled_acc)
        
        # 4. Imposter Detection thresholding
        # We define a threshold using PCA reconstruction error of enrolled training images.
        # If the reconstruction error of a test face is significantly higher than training faces,
        # or if the neural network confidence is low, it is an imposter.
        
        # Calculate reconstruction error for training images (using RAW signatures)
        train_reconstruction = pca.reconstruct(train_signatures_raw)
        train_rec_errors = np.linalg.norm(train_db - train_reconstruction, axis=0)
        # Set threshold at 95th percentile of training reconstruction errors
        reconstruction_threshold = np.percentile(train_rec_errors, 95)
        
        # Let's also check distance to nearest training signature in face space
        # Distances between each train signature and all other train signatures
        train_dists = []
        for i in range(train_signatures_raw.shape[1]):
            diffs = train_signatures_raw - train_signatures_raw[:, i:i+1]
            dists = np.linalg.norm(diffs, axis=0)
            # Find the minimum non-zero distance (distance to nearest neighbor)
            dists[i] = np.inf
            train_dists.append(np.min(dists))
        # Distance threshold at 95th percentile of nearest training neighbors
        distance_threshold = np.percentile(train_dists, 95) * 2.0  # scaling factor
        
        # Evaluate Enrolled Test set for "False Alarm" (mistakenly flagged as imposter)
        test_reconstruction = pca.reconstruct(test_signatures_raw)
        test_rec_errors = np.linalg.norm(test_db - test_reconstruction, axis=0)
        
        test_min_dists = []
        for i in range(test_signatures_raw.shape[1]):
            dists = np.linalg.norm(train_signatures_raw - test_signatures_raw[:, i:i+1], axis=0)
            test_min_dists.append(np.min(dists))
        test_min_dists = np.array(test_min_dists)
        
        # Max probability from ANN
        test_max_probs = np.max(test_probs, axis=0)
        
        # We classify as imposter ("not enrolled") if:
        # - Reconstruction error exceeds reconstruction_threshold, OR
        # - Distance to nearest training face exceeds distance_threshold, OR
        # - ANN output probability is lower than a confidence threshold (e.g. 0.45)
        enrolled_is_imposter_pred = (
            (test_rec_errors > reconstruction_threshold) | 
            (test_min_dists > distance_threshold) |
            (test_max_probs < 0.45)
        )
        
        # Evaluate Imposter Test set
        imposter_signatures_raw = pca.project(imposter_db)
        imposter_signatures = (imposter_signatures_raw - sig_mean) / sig_std
        
        imposter_reconstruction = pca.reconstruct(imposter_signatures_raw)
        imposter_rec_errors = np.linalg.norm(imposter_db - imposter_reconstruction, axis=0)
        
        imposter_min_dists = []
        for i in range(imposter_signatures_raw.shape[1]):
            dists = np.linalg.norm(train_signatures_raw - imposter_signatures_raw[:, i:i+1], axis=0)
            imposter_min_dists.append(np.min(dists))
        imposter_min_dists = np.array(imposter_min_dists)
        
        _, imposter_probs = ann.predict(imposter_signatures)
        imposter_max_probs = np.max(imposter_probs, axis=0)
        
        imposter_is_imposter_pred = (
            (imposter_rec_errors > reconstruction_threshold) | 
            (imposter_min_dists > distance_threshold) |
            (imposter_max_probs < 0.45)
        )
        
        # Calculate rates
        imposter_detection_rate = np.mean(imposter_is_imposter_pred) * 100
        imposter_detection_rates.append(imposter_detection_rate)
        
        # Combined score:
        # Enrolled images must be correctly classified (and NOT flagged as imposter)
        # Imposters must be flagged as imposter.
        correct_enrolled = np.sum((test_preds == test_labels) & (~enrolled_is_imposter_pred))
        correct_imposters = np.sum(imposter_is_imposter_pred)
        
        total_test_samples = len(test_labels) + imposter_db.shape[1]
        overall_acc = (correct_enrolled + correct_imposters) / total_test_samples * 100
        overall_accuracies.append(overall_acc)
        
        print(f"  Enrolled Face Classification Accuracy: {enrolled_acc:.2f}%")
        print(f"  Imposter Detection Rate (True Negative): {imposter_detection_rate:.2f}%")
        print(f"  Overall Recognition System Accuracy (Enrolled + Imposters): {overall_acc:.2f}%")
        
    # ==========================================
    # 5. Plotting and Visualizations
    # ==========================================
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, enrolled_accuracies, marker='o', linewidth=2, label='Enrolled Classification Accuracy')
    plt.plot(k_values, imposter_detection_rates, marker='s', linewidth=2, label='Imposter Detection Rate')
    plt.plot(k_values, overall_accuracies, marker='^', linewidth=2, label='Overall System Accuracy')
    plt.title('Face Recognition System Performance vs. Number of Eigenfaces (k)', fontsize=14)
    plt.xlabel('Number of Selected Eigenvectors (k)', fontsize=12)
    plt.ylabel('Performance (%)', fontsize=12)
    plt.xticks(k_values)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    
    # Save the plot
    plot_path = "accuracy_vs_k.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved performance plot to: {os.path.abspath(plot_path)}")

    # Let's save some sample eigenfaces for visualization
    # Re-train with k=10 for visualization
    pca_vis = EigenfacePCA(k=10)
    pca_vis.fit(train_db)
    
    plt.figure(figsize=(12, 6))
    for idx in range(min(5, pca_vis.k)):
        eigenface = pca_vis.eigenfaces[idx].reshape((64, 64))
        plt.subplot(1, 5, idx + 1)
        # Using grayscale colormap to view actual face structures
        plt.imshow(eigenface, cmap='gray')
        plt.title(f'Eigenface {idx+1}')
        plt.axis('off')
    plt.suptitle('Top 5 Generated Eigenfaces (k Direction Projections)', fontsize=14)
    vis_path = "eigenfaces_visualization.png"
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved eigenfaces visualization to: {os.path.abspath(vis_path)}")
    print("="*60)

if __name__ == "__main__":
    main()
