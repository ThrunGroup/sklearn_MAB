import numpy as np

from .splitting import SplitInfo

class CandidateContainer:
    def __init__(self, histograms, n_samples, missing_go_to_left, is_categorical,
                 sum_gradient, sum_hessian, node):
        n_features = histograms.shape[0]
        n_bins = histograms.shape[1]


        self.candidate_stats = np.zeros((n_features, n_bins, 2)) #(mean, confidence_bound) pair array of size (n_features, n_bins)

        sum_gradient_left, sum_hessian_left = 0., 0.
        n_samples_left = 0

        for feature_idx in range(n_features):
            for bin_idx in range(n_bins):

                n_samples_left += histograms[feature_idx, bin_idx].count
                sum_gradient_left += histograms[feature_idx, bin_idx].sum_gradient
                sum_hessian_left += histograms[feature_idx, bin_idx].sum_hessian

                n_samples_right = n_samples - n_samples_left
                sum_gradient_right = sum_gradient - sum_gradient_left
                sum_hessian_right = sum_hessian - sum_hessian_left

                gain = _split_gain(sum_gradient_left, sum_hessian_left,
                                   sum_gradient_right, sum_hessian_right,
                                   loss_current_node,
                                   monotonic_cst,
                                   lower_bound,
                                   upper_bound,
                                   self.l2_regularization)

                value_left =

                candidate = SplitInfo(gain, feature_idx, bin_idx,
                                      missing_go_to_left, sum_gradient_left, sum_hessian_left,
                                      sum_gradient_right, sum_hessian_right, n_samples_left,
                                      n_samples_right, value_left, value_right,
                                      is_categorical, left_cat_bitset)


        self.best_candidate = self.candidates[0]
        self.candidate_n = n_features * n_bins #works where missing doesn't exist