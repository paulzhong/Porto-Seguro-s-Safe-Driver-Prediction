"""
Scoring Metric

Submissions are evaluated using the Normalized Gini Coefficient.

During scoring, observations are sorted from the largest to the smallest
predictions. Predictions are only used for ordering observations; therefore,
the relative magnitude of the predictions are not used during scoring.
The scoring algorithm then compares the cumulative proportion of positive
class observations to a theoretical uniform proportion.

The Gini Coefficient ranges from approximately 0 for random guessing,
to approximately 0.5 for a perfect score. The theoretical maximum for the
discrete calculation is (1 - frac_pos) / 2.

The Normalized Gini Coefficient adjusts the score by the theoretical
maximum so that the maximum score is 1.
"""


def auc_to_gini_norm(auc_score): #AucToGiniNorm(auc_score):
    return 2*auc_score-1
