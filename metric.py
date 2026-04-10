import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_recall_curve, auc, confusion_matrix
)
from torch_geometric.data import Batch

# Tox21 endpoint names
tox21_tasks = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]

def evaluate_model(model, test_dataset, device="cpu"):
    model.eval()

    y_true = []
    y_pred = []

    for data in test_dataset:
        g = data
        batch = Batch.from_data_list([g]).to(device)

        with torch.no_grad():
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        y_pred.append(probs)
        y_true.append(data.y.cpu().numpy()[0])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    results = {}

    for i, task in enumerate(tox21_tasks):

        true = y_true[:, i]
        pred = y_pred[:, i]

        # Skip tasks where all labels are missing
        if np.all(true == 0) or np.all(true == 1):
            continue

        # Calculate ROC-AUC
        try:
            roc = roc_auc_score(true, pred)
        except:
            roc = None

        # PR-AUC (better for imbalanced assays)
        try:
            precision, recall, _ = precision_recall_curve(true, pred)
            pr_auc = auc(recall, precision)
        except:
            pr_auc = None

        # Convert probability → binary using threshold=0.5
        bin_pred = (pred > 0.5).astype(int)

        acc = accuracy_score(true, bin_pred)
        f1 = f1_score(true, bin_pred)

        cm = confusion_matrix(true, bin_pred)

        results[task] = {
            "ROC-AUC": roc,
            "PR-AUC": pr_auc,
            "Accuracy": acc,
            "F1-score": f1,
            "Confusion Matrix": cm
        }

    return results


# Run evaluation
device = "cpu"
model = load_model(device)
metrics = evaluate_model(model, test_dataset, device)

# Print nicely
for task, res in metrics.items():
    print(f"\n===== {task} =====")
    for k, v in res.items():
        print(f"{k}: {v}")
