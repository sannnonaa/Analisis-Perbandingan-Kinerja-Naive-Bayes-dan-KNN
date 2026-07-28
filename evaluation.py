import numpy as np

def confusion_matrix_manual(y_true, y_pred, labels=None):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))

    label_to_index = {label: i for i, label in enumerate(labels)}

    cm = np.zeros((len(labels), len(labels)), dtype=int)

    for yt, yp in zip(y_true, y_pred):
        i = label_to_index[yt]
        j = label_to_index[yp]
        cm[i, j] += 1

    return cm, labels


# -----------------------------------------------------
# Accuracy

def accuracy_score_manual(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) == 0:
        return 0.0

    return np.sum(y_true == y_pred) / len(y_true)


# -----------------------------------------------------
# Precision, Recall, F1 per class + macro

def classification_report_manual(y_true, y_pred, labels=None):

    cm, labels = confusion_matrix_manual(y_true, y_pred, labels)

    n_class = len(labels)

    precision = {}
    recall = {}
    f1 = {}
    support = {}

    for i in range(n_class):

        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP

        if TP + FP == 0:
            prec = 0.0
        else:
            prec = TP / (TP + FP)

        if TP + FN == 0:
            rec = 0.0
        else:
            rec = TP / (TP + FN)

        if prec + rec == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * prec * rec / (prec + rec)

        precision[labels[i]] = prec
        recall[labels[i]] = rec
        f1[labels[i]] = f1_score
        support[labels[i]] = cm[i, :].sum()

    macro_precision = np.mean(list(precision.values()))
    macro_recall = np.mean(list(recall.values()))
    macro_f1 = np.mean(list(f1.values()))

    report = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "macro_avg": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1
        }
    }

    return report, cm, labels


def print_classification_report(report):

    print("Class\t\tPrecision\tRecall\t\tF1-Score\tSupport")

    for label in report["precision"].keys():
        print(
            f"{label}\t\t"
            f"{report['precision'][label]:.4f}\t\t"
            f"{report['recall'][label]:.4f}\t\t"
            f"{report['f1'][label]:.4f}\t\t"
            f"{report['support'][label]}"
        )

    print("\nMacro Average")
    print(
        f"Precision : {report['macro_avg']['precision']:.4f}\n"
        f"Recall    : {report['macro_avg']['recall']:.4f}\n"
        f"F1-Score  : {report['macro_avg']['f1']:.4f}"
    )


def weighted_f1(report: dict) -> float:
    """
    Menghitung Weighted F1-Score dari classification report manual.

    Weighted F1 = Σ (support_i / total_support) × F1_i
    Berbeda dengan Macro F1 yang memberikan bobot sama ke setiap kelas,
    Weighted F1 memberikan bobot proporsional terhadap jumlah sampel per kelas.
    Lebih representatif pada dataset tidak seimbang (imbalanced).
    """
    f1_scores = report.get("f1", {})
    supports  = report.get("support", {})

    total_support = sum(supports.values())
    if total_support == 0:
        return 0.0

    weighted = sum(
        f1_scores.get(label, 0.0) * supports.get(label, 0)
        for label in f1_scores
    )
    return weighted / total_support