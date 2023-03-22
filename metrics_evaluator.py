import numpy as np
from sklearn.metrics import average_precision_score

class MetricsEvaluator: 
    def __init__(self, gt_data, pred_data):
        self.gt_data = gt_data.cpu().numpy()
        self.pred_data = pred_data.cpu().numpy()

    def f1_score(self):
        # Compute true positives, false positives, and false negatives
        true_positives = np.sum(np.logical_and(self.gt_data == 1, self.pred_data == 1))
        false_positives = np.sum(np.logical_and(self.gt_data == 0, self.pred_data == 1))
        false_negatives = np.sum(np.logical_and(self.gt_data == 1, self.pred_data == 0))

        # Compute precision, recall, and F1 score
        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

        # Compute F1 score for overlay area
        overlay_f1_score = 2 * true_positives / (np.sum(self.gt_data) + np.sum(self.pred_data) + 1e-10)

        return precision, recall, f1, overlay_f1_score
    
    def dice_coefficient(self):
        # Compute true positives, false positives, and false negatives
        true_positives = np.sum(np.logical_and(self.gt_data == 1, self.pred_data == 1))
        false_positives = np.sum(np.logical_and(self.gt_data == 0, self.pred_data == 1))
        false_negatives = np.sum(np.logical_and(self.gt_data == 1, self.pred_data == 0))

        # Compute Dice coefficient for overlay area
        dice_coefficient = 2 * true_positives / (2 * true_positives + false_positives + false_negatives)

        return dice_coefficient
    
    def average_precision(self):
        y_true = self.gt_data.flatten()
        y_scores = self.pred_data.flatten()
        nan_mask = np.logical_or(np.isnan(y_true), np.isnan(y_scores))
        y_true = y_true[~nan_mask]
        y_scores = y_scores[~nan_mask]
        ap = average_precision_score(y_true, y_scores)

        return ap
