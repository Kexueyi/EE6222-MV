import torch
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

def to_cpu(tensor):
    return tensor.to('cpu') if tensor.is_cuda else tensor

train_features = torch.load('train_features.pt')
train_labels = torch.load('train_labels.pt')
val_features = torch.load('val_features.pt')
val_labels = torch.load('val_labels.pt')

train_features = to_cpu(train_features).reshape(150, -1)
train_labels = to_cpu(train_labels)
val_features = to_cpu(val_features).reshape(96, -1)
val_labels = to_cpu(val_labels)

svm = SVC()

svm.fit(train_features.numpy(), train_labels.numpy())

train_preds = svm.predict(train_features.numpy())
val_preds = svm.predict(val_features.numpy())

# evaluation
conf_mat = confusion_matrix(val_labels.numpy(), val_preds)
sns.heatmap(conf_mat, annot=True, fmt='d')
plt.savefig('pics/svm_evalu.png')

print(classification_report(val_labels.numpy(), val_preds))
