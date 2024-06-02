from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
from scipy.spatial import distance
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import itertools

# Load breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Splitting the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply PCA
pca_2 = PCA(n_components=2)
pca_5 = PCA(n_components=5)
pca_10 = PCA(n_components=10)

X_train_pca_2 = pca_2.fit_transform(X_train)
X_train_pca_5 = pca_5.fit_transform(X_train)
X_train_pca_10 = pca_10.fit_transform(X_train)

# Apply LDA
lda = LDA()
X_train_lda = lda.fit_transform(X_train, y_train)

def mahalanobis_classifier(X_train, y_train, X_test):
    if X_train.ndim == 1:
        X_train = X_train[:, np.newaxis]
    if X_test.ndim == 1:
        X_test = X_test[:, np.newaxis]
    mean_class0 = np.mean(X_train[y_train == 0], axis=0)
    mean_class1 = np.mean(X_train[y_train == 1], axis=0)
    if X_train.shape[1] == 1:
        distances_class0 = np.abs(X_test - mean_class0)
        distances_class1 = np.abs(X_test - mean_class1)
    else:
        cov_matrix = np.cov(X_train, rowvar=False)
        inv_cov_matrix = np.linalg.pinv(cov_matrix)
        distances_class0 = [distance.mahalanobis(x, mean_class0, inv_cov_matrix) for x in X_test]
        distances_class1 = [distance.mahalanobis(x, mean_class1, inv_cov_matrix) for x in X_test]
    y_pred = np.where(np.array(distances_class0) < np.array(distances_class1), 0, 1)
    
    return y_pred

X_test_pca_2 = pca_2.transform(X_test)
X_test_pca_5 = pca_5.transform(X_test)
X_test_pca_10 = pca_10.transform(X_test)

X_test_lda = lda.transform(X_test)

y_pred_pca_2 = mahalanobis_classifier(X_train_pca_2, y_train, X_test_pca_2)
y_pred_pca_5 = mahalanobis_classifier(X_train_pca_5, y_train, X_test_pca_5)
y_pred_pca_10 = mahalanobis_classifier(X_train_pca_10, y_train, X_test_pca_10)
y_pred_lda = mahalanobis_classifier(X_train_lda, y_train, X_test_lda)

# plot a confusion matrix 
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['0', '1'], rotation=45)
    plt.yticks(tick_marks, ['0', '1'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()

# Compute confusion matrices
confusion_pca_2 = confusion_matrix(y_test, y_pred_pca_2)
confusion_pca_5 = confusion_matrix(y_test, y_pred_pca_5)
confusion_pca_10 = confusion_matrix(y_test, y_pred_pca_10)
confusion_lda = confusion_matrix(y_test, y_pred_lda)

# Normalize
def normalize_confusion_matrix(cm):
    return cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

normalized_confusion_pca_2 = normalize_confusion_matrix(confusion_pca_2)
normalized_confusion_pca_5 = normalize_confusion_matrix(confusion_pca_5)
normalized_confusion_pca_10 = normalize_confusion_matrix(confusion_pca_10)
normalized_confusion_lda = normalize_confusion_matrix(confusion_lda)

# Plot the normalized confusion matrices
plot_confusion_matrix(normalized_confusion_pca_2, title="PCA (2 components)")
plot_confusion_matrix(normalized_confusion_pca_5, title="PCA (5 components)")
plot_confusion_matrix(normalized_confusion_pca_10, title="PCA (10 components)")
plot_confusion_matrix(normalized_confusion_lda, title="LDA")
plt.show()

# Compute ROC curve and ROC area for each PCA and LDA
fpr_pca_2, tpr_pca_2, _ = roc_curve(y_test, y_pred_pca_2)
roc_auc_pca_2 = auc(fpr_pca_2, tpr_pca_2)
fpr_pca_5, tpr_pca_5, _ = roc_curve(y_test, y_pred_pca_5)
roc_auc_pca_5 = auc(fpr_pca_5, tpr_pca_5)
fpr_pca_10, tpr_pca_10, _ = roc_curve(y_test, y_pred_pca_10)
roc_auc_pca_10 = auc(fpr_pca_10, tpr_pca_10)
fpr_lda, tpr_lda, _ = roc_curve(y_test, y_pred_lda)
roc_auc_lda = auc(fpr_lda, tpr_lda)

# Plotting the ROC curve
plt.figure(figsize=(10, 8))
plt.plot(fpr_pca_2, tpr_pca_2, color='blue', lw=2, label='PCA (2 components) (area = %0.2f)' % roc_auc_pca_2)
plt.plot(fpr_pca_5, tpr_pca_5, color='green', lw=2, label='PCA (5 components) (area = %0.2f)' % roc_auc_pca_5)
plt.plot(fpr_pca_10, tpr_pca_10, color='yellow', lw=2, label='PCA (10 components) (area = %0.2f)' % roc_auc_pca_10)
plt.plot(fpr_lda, tpr_lda, color='red', lw=2, label='LDA (area = %0.2f)' % roc_auc_lda)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc="lower right")
plt.show()
