"""This module contains routines for building histograms."""

# Author: Nicolas Hug

import numpy as np

from .splitting import SplitInfo

cimport cython
from cython.parallel import prange

import numpy as np

from .common import HISTOGRAM_DTYPE
from .common cimport hist_struct
from .common cimport X_BINNED_DTYPE_C
from .common cimport G_H_DTYPE_C

from libc.stdio cimport printf


# Notes:
# - IN views are read-only, OUT views are write-only
# - In a lot of functions here, we pass feature_idx and the whole 2d
#   histograms arrays instead of just histograms[feature_idx]. This is because
#   Cython generated C code will have strange Python interactions (likely
#   related to the GIL release and the custom histogram dtype) when using 1d
#   histogram arrays that come from 2d arrays.
# - The for loops are un-wrapped, for example:
#
#   for i in range(n):
#       array[i] = i
#
#   will become
#
#   for i in range(n // 4):
#       array[i] = i
#       array[i + 1] = i + 1
#       array[i + 2] = i + 2
#       array[i + 3] = i + 3
#
#   This is to hint gcc that it can auto-vectorize these 4 operations and
#   perform them all at once.


@cython.final
cdef class MABHistogramBuilder:
    """A Histogram builder... used to build histograms.

    A histogram is an array with n_bins entries of type HISTOGRAM_DTYPE. Each
    feature has its own histogram. A histogram contains the sum of gradients
    and hessians of all the samples belonging to each bin.

    There are different ways to build a histogram:
    - by subtraction: hist(child) = hist(parent) - hist(sibling)
    - from scratch. In this case we have routines that update the hessians
      or not (not useful when hessians are constant for some losses e.g.
      least squares). Also, there's a special case for the root which
      contains all the samples, leading to some possible optimizations.
      Overall all the implementations look the same, and are optimized for
      cache hit.

    Parameters
    ----------
    X_binned : ndarray of int, shape (n_samples, n_features)
        The binned input samples. Must be Fortran-aligned.
    n_bins : int
        The total number of bins, including the bin for missing values. Used
        to define the shape of the histograms.
    gradients : ndarray, shape (n_samples,)
        The gradients of each training sample. Those are the gradients of the
        loss w.r.t the predictions, evaluated at iteration i - 1.
    hessians : ndarray, shape (n_samples,)
        The hessians of each training sample. Those are the hessians of the
        loss w.r.t the predictions, evaluated at iteration i - 1.
    hessians_are_constant : bool
        Whether hessians are constant.
    """
    cdef public:
        #const X_BINNED_DTYPE_C [::1, :] X_binned
        unsigned int n_features
        unsigned int n_bins
        G_H_DTYPE_C [::1] gradients
        G_H_DTYPE_C [::1] hessians
        G_H_DTYPE_C [::1] ordered_gradients
        G_H_DTYPE_C [::1] ordered_hessians
        unsigned char hessians_are_constant
        int n_threads

    def __init__(self, const X_BINNED_DTYPE_C [::1, :] X_binned,
                 unsigned int n_bins, G_H_DTYPE_C [::1] gradients,
                 G_H_DTYPE_C [::1] hessians,
                 unsigned char hessians_are_constant,
                 unsigned int n_training_data,
                 int n_threads):

        self.X_binned = X_binned
        self.n_features = X_binned.shape[1]
        # Note: all histograms will have <n_bins> bins, but some of the
        # bins may be unused if a feature has a small number of unique values.
        self.n_bins = n_bins
        self.gradients = np.zeros((n_training_data))
        self.hessians = np.zeros((n_training_data))
        # for root node, gradients and hessians are already ordered
        self.ordered_gradients = gradients.copy()
        self.ordered_hessians = hessians.copy()
        self.hessians_are_constant = hessians_are_constant
        self.n_threads = n_threads

    def set_gradient(
        HistogramBuilder self,
        const unsigned int sample_indices,
        G_H_DTYPE_C [::1] gradient  #TODO: set appropriate type shape:(sample_indices.shape[0])
    ):
        pass
        #TODO: implement

    def set_hessian(
        HistogramBuilder self,
        const unsigned int sample_indices,
        G_H_DTYPE_C [::1] hessian  #TODO: set appropriate type shape:(sample_indices.shape[0])
    ):
        pass
        #TODO: implement

    def compute_histograms_brute(
        HistogramBuilder self,
        const unsigned int [::1] sample_indices,       # IN
        const unsigned int [:] allowed_features=None,
        const unsigned int n_features
    ):
        """Compute the histograms of the node by scanning through all the data.

        For a given feature, the complexity is O(n_samples)

        Parameters
        ----------
        sample_indices : array of int, shape (n_samples_at_node,)
            The indices of the samples at the node to split.

        allowed_features : None or ndarray, dtype=np.uint32
            Indices of the features that are allowed by interaction constraints to be
            split.

        Returns
        -------
        histograms : ndarray of HISTOGRAM_DTYPE, shape (n_features, n_bins)
            The computed histograms of the current node.
        """
        cdef:
            int n_samples
            int feature_idx
            int f_idx
            int i
            # need local views to avoid python interactions
            unsigned char hessians_are_constant = self.hessians_are_constant
            int n_allowed_features = n_features
            G_H_DTYPE_C [::1] ordered_gradients = self.ordered_gradients
            G_H_DTYPE_C [::1] gradients = self.gradients
            G_H_DTYPE_C [::1] ordered_hessians = self.ordered_hessians
            G_H_DTYPE_C [::1] hessians = self.hessians
            # Histograms will be initialized to zero later within a prange
            hist_struct [:, ::1] histograms = np.empty(
                shape=(self.n_features, self.n_bins),
                dtype=HISTOGRAM_DTYPE
            )
            bint has_interaction_cst = allowed_features is not None
            int n_threads = self.n_threads

        if has_interaction_cst:
            n_allowed_features = allowed_features.shape[0]

        with nogil:
            n_samples = sample_indices.shape[0]

            # Populate ordered_gradients and ordered_hessians. (Already done
            # for root) Ordering the gradients and hessians helps to improve
            # cache hit.
            if sample_indices.shape[0] != gradients.shape[0]:
                if hessians_are_constant:
                    for i in prange(n_samples, schedule='static',
                                    num_threads=n_threads):
                        ordered_gradients[i] = gradients[sample_indices[i]]
                else:
                    for i in prange(n_samples, schedule='static',
                                    num_threads=n_threads):
                        ordered_gradients[i] = gradients[sample_indices[i]]
                        ordered_hessians[i] = hessians[sample_indices[i]]

            # Compute histogram of each feature
            for f_idx in prange(
                n_allowed_features, schedule='static', num_threads=n_threads
            ):
                if has_interaction_cst:
                    feature_idx = allowed_features[f_idx]
                else:
                    feature_idx = f_idx

                self._compute_histogram_brute_single_feature(
                    feature_idx, sample_indices, histograms, True
                )

        return histograms

    cdef void _compute_histogram_brute_single_feature(
            HistogramBuilder self,
            const int feature_idx,
            const unsigned int [::1] sample_indices,  # IN
            hist_struct [:, ::1] histograms,
            const bint is_initial=True) noexcept nogil:  # OUT
        """Compute the histogram for a given feature."""

        cdef:
            unsigned int n_samples = sample_indices.shape[0]
            const X_BINNED_DTYPE_C [::1] X_binned = \
                self.X_binned[:, feature_idx]
            unsigned int root_node = X_binned.shape[0] == n_samples
            G_H_DTYPE_C [::1] ordered_gradients = \
                self.ordered_gradients[:n_samples]
            G_H_DTYPE_C [::1] ordered_hessians = \
                self.ordered_hessians[:n_samples]
            unsigned char hessians_are_constant = \
                self.hessians_are_constant
            unsigned int bin_idx = 0

        if is_initial:
            for bin_idx in range(self.n_bins):
                histograms[feature_idx, bin_idx].sum_gradients = 0.
                histograms[feature_idx, bin_idx].sum_hessians = 0.
                histograms[feature_idx, bin_idx].count = 0

        if root_node:
            if hessians_are_constant:
                _build_histogram_root_no_hessian(feature_idx, X_binned,
                                                 ordered_gradients,
                                                 histograms)
            else:
                _build_histogram_root(feature_idx, X_binned,
                                      ordered_gradients, ordered_hessians,
                                      histograms)
        else:
            if hessians_are_constant:
                _build_histogram_no_hessian(feature_idx,
                                            sample_indices, X_binned,
                                            ordered_gradients, histograms)
            else:
                _build_histogram(feature_idx, sample_indices,
                                 X_binned, ordered_gradients,
                                 ordered_hessians, histograms)

    def compute_histograms_subtraction(
        HistogramBuilder self,
        hist_struct [:, ::1] parent_histograms,        # IN
        hist_struct [:, ::1] sibling_histograms,       # IN
        const unsigned int [:] allowed_features=None,  # IN
    ):
        """Compute the histograms of the node using the subtraction trick.

        hist(parent) = hist(left_child) + hist(right_child)

        For a given feature, the complexity is O(n_bins). This is much more
        efficient than compute_histograms_brute, but it's only possible for one
        of the siblings.

        Parameters
        ----------
        parent_histograms : ndarray of HISTOGRAM_DTYPE, \
                shape (n_features, n_bins)
            The histograms of the parent.
        sibling_histograms : ndarray of HISTOGRAM_DTYPE, \
                shape (n_features, n_bins)
            The histograms of the sibling.
        allowed_features : None or ndarray, dtype=np.uint32
            Indices of the features that are allowed by interaction constraints to be
            split.

        Returns
        -------
        histograms : ndarray of HISTOGRAM_DTYPE, shape(n_features, n_bins)
            The computed histograms of the current node.
        """

        cdef:
            int feature_idx
            int f_idx
            int n_allowed_features = self.n_features
            hist_struct [:, ::1] histograms = np.empty(
                shape=(self.n_features, self.n_bins),
                dtype=HISTOGRAM_DTYPE
            )
            bint has_interaction_cst = allowed_features is not None
            int n_threads = self.n_threads

        if has_interaction_cst:
            n_allowed_features = allowed_features.shape[0]

        # Compute histogram of each feature
        for f_idx in prange(n_allowed_features, schedule='static', nogil=True,
                            num_threads=n_threads):
            if has_interaction_cst:
                feature_idx = allowed_features[f_idx]
            else:
                feature_idx = f_idx

            _subtract_histograms(
                feature_idx,
                self.n_bins,
                parent_histograms,
                sibling_histograms,
                histograms,
            )
        return histograms

    #super naive: assumes "no intersection"(?)
    def update_histograms(
        HistogramBuilder self,
        hist_struct [:, ::1] histograms,
        const unsigned int [::1] sample_indices,
        const unsigned int [:] allowed_features=None
    ):
        cdef:
            int n_samples
            int feature_idx
            int f_idx
            int i
            # need local views to avoid python interactions
            unsigned char hessians_are_constant = self.hessians_are_constant
            int n_allowed_features = self.n_features
            G_H_DTYPE_C [::1] ordered_gradients = self.ordered_gradients
            G_H_DTYPE_C [::1] gradients = self.gradients
            G_H_DTYPE_C [::1] ordered_hessians = self.ordered_hessians
            G_H_DTYPE_C [::1] hessians = self.hessians
            bint has_interaction_cst = allowed_features is not None
            int n_threads = self.n_threads
            hist_struct [:, ::1] new_histograms = histograms

            int n_hist_data = 0

        print(has_interaction_cst)

        if has_interaction_cst:
            n_allowed_features = allowed_features.shape[0]

        with nogil:
            # Populate ordered_gradients and ordered_hessians. (Already done
            # for root) Ordering the gradients and hessians helps to improve
            # cache hit.
            if sample_indices.shape[0] != gradients.shape[0]:
                if hessians_are_constant:
                    for i in prange(n_samples, schedule='static',
                                    num_threads=n_threads):
                        ordered_gradients[i] = gradients[sample_indices[i]]
                else:
                    for i in prange(n_samples, schedule='static',
                                    num_threads=n_threads):
                        ordered_gradients[i] = gradients[sample_indices[i]]
                        ordered_hessians[i] = hessians[sample_indices[i]]

            # Compute histogram of each feature
            for f_idx in prange(
                n_allowed_features, schedule='static', num_threads=n_threads
            ):
                if has_interaction_cst:
                    feature_idx = allowed_features[f_idx]
                else:
                    feature_idx = f_idx

                self._compute_histogram_brute_single_feature(
                    feature_idx, sample_indices, new_histograms, False
                )

        return new_histograms

cpdef void _build_histogram_naive(
        const int feature_idx,
        unsigned int [:] sample_indices,  # IN
        X_BINNED_DTYPE_C [:] binned_feature,  # IN
        G_H_DTYPE_C [:] ordered_gradients,  # IN
        G_H_DTYPE_C [:] ordered_hessians,  # IN
        hist_struct [:, :] out) noexcept nogil:  # OUT
    """Build histogram in a naive way, without optimizing for cache hit.

    Used in tests to compare with the optimized version."""
    cdef:
        unsigned int i
        unsigned int n_samples = sample_indices.shape[0]
        unsigned int sample_idx
        unsigned int bin_idx

    for i in range(n_samples):
        sample_idx = sample_indices[i]
        bin_idx = binned_feature[sample_idx]
        out[feature_idx, bin_idx].sum_gradients += ordered_gradients[i]
        out[feature_idx, bin_idx].sum_hessians += ordered_hessians[i]
        out[feature_idx, bin_idx].count += 1


cpdef void _subtract_histograms(
        const int feature_idx,
        unsigned int n_bins,
        hist_struct [:, ::1] hist_a,  # IN
        hist_struct [:, ::1] hist_b,  # IN
        hist_struct [:, ::1] out) noexcept nogil:  # OUT
    """compute (hist_a - hist_b) in out"""
    cdef:
        unsigned int i = 0
    for i in range(n_bins):
        out[feature_idx, i].sum_gradients = (
            hist_a[feature_idx, i].sum_gradients -
            hist_b[feature_idx, i].sum_gradients
        )
        out[feature_idx, i].sum_hessians = (
            hist_a[feature_idx, i].sum_hessians -
            hist_b[feature_idx, i].sum_hessians
        )
        out[feature_idx, i].count = (
            hist_a[feature_idx, i].count -
            hist_b[feature_idx, i].count
        )


cpdef void _build_histogram(
        const int feature_idx,
        const unsigned int [::1] sample_indices,  # IN
        const X_BINNED_DTYPE_C [::1] binned_feature,  # IN
        const G_H_DTYPE_C [::1] ordered_gradients,  # IN
        const G_H_DTYPE_C [::1] ordered_hessians,  # IN
        hist_struct [:, ::1] out) noexcept nogil:  # OUT
    """Return histogram for a given feature."""
    cdef:
        unsigned int i = 0
        unsigned int n_node_samples = sample_indices.shape[0]
        unsigned int unrolled_upper = (n_node_samples // 4) * 4

        unsigned int bin_0
        unsigned int bin_1
        unsigned int bin_2
        unsigned int bin_3
        unsigned int bin_idx

    for i in range(0, unrolled_upper, 4):
        bin_0 = binned_feature[sample_indices[i]]
        bin_1 = binned_feature[sample_indices[i + 1]]
        bin_2 = binned_feature[sample_indices[i + 2]]
        bin_3 = binned_feature[sample_indices[i + 3]]

        out[feature_idx, bin_0].sum_gradients += ordered_gradients[i]
        out[feature_idx, bin_1].sum_gradients += ordered_gradients[i + 1]
        out[feature_idx, bin_2].sum_gradients += ordered_gradients[i + 2]
        out[feature_idx, bin_3].sum_gradients += ordered_gradients[i + 3]

        out[feature_idx, bin_0].sum_hessians += ordered_hessians[i]
        out[feature_idx, bin_1].sum_hessians += ordered_hessians[i + 1]
        out[feature_idx, bin_2].sum_hessians += ordered_hessians[i + 2]
        out[feature_idx, bin_3].sum_hessians += ordered_hessians[i + 3]

        out[feature_idx, bin_0].count += 1
        out[feature_idx, bin_1].count += 1
        out[feature_idx, bin_2].count += 1
        out[feature_idx, bin_3].count += 1

    for i in range(unrolled_upper, n_node_samples):
        bin_idx = binned_feature[sample_indices[i]]
        out[feature_idx, bin_idx].sum_gradients += ordered_gradients[i]
        out[feature_idx, bin_idx].sum_hessians += ordered_hessians[i]
        out[feature_idx, bin_idx].count += 1


cpdef void _build_histogram_no_hessian(
        const int feature_idx,
        const unsigned int [::1] sample_indices,  # IN
        const X_BINNED_DTYPE_C [::1] binned_feature,  # IN
        const G_H_DTYPE_C [::1] ordered_gradients,  # IN
        hist_struct [:, ::1] out) noexcept nogil:  # OUT
    """Return histogram for a given feature, not updating hessians.

    Used when the hessians of the loss are constant (typically LS loss).
    """

    cdef:
        unsigned int i = 0
        unsigned int n_node_samples = sample_indices.shape[0]
        unsigned int unrolled_upper = (n_node_samples // 4) * 4

        unsigned int bin_0
        unsigned int bin_1
        unsigned int bin_2
        unsigned int bin_3
        unsigned int bin_idx

    for i in range(0, unrolled_upper, 4):
        bin_0 = binned_feature[sample_indices[i]]
        bin_1 = binned_feature[sample_indices[i + 1]]
        bin_2 = binned_feature[sample_indices[i + 2]]
        bin_3 = binned_feature[sample_indices[i + 3]]

        out[feature_idx, bin_0].sum_gradients += ordered_gradients[i]
        out[feature_idx, bin_1].sum_gradients += ordered_gradients[i + 1]
        out[feature_idx, bin_2].sum_gradients += ordered_gradients[i + 2]
        out[feature_idx, bin_3].sum_gradients += ordered_gradients[i + 3]

        out[feature_idx, bin_0].count += 1
        out[feature_idx, bin_1].count += 1
        out[feature_idx, bin_2].count += 1
        out[feature_idx, bin_3].count += 1

    for i in range(unrolled_upper, n_node_samples):
        bin_idx = binned_feature[sample_indices[i]]
        out[feature_idx, bin_idx].sum_gradients += ordered_gradients[i]
        out[feature_idx, bin_idx].count += 1


cpdef void _build_histogram_root(
        const int feature_idx,
        const X_BINNED_DTYPE_C [::1] binned_feature,  # IN
        const G_H_DTYPE_C [::1] all_gradients,  # IN
        const G_H_DTYPE_C [::1] all_hessians,  # IN
        hist_struct [:, ::1] out) noexcept nogil:  # OUT
    """Compute histogram of the root node.

    Unlike other nodes, the root node has to find the split among *all* the
    samples from the training set. binned_feature and all_gradients /
    all_hessians already have a consistent ordering.
    """

    cdef:
        unsigned int i = 0
        unsigned int n_samples = binned_feature.shape[0]
        unsigned int unrolled_upper = (n_samples // 4) * 4

        unsigned int bin_0
        unsigned int bin_1
        unsigned int bin_2
        unsigned int bin_3
        unsigned int bin_idx

    for i in range(0, unrolled_upper, 4):

        bin_0 = binned_feature[i]
        bin_1 = binned_feature[i + 1]
        bin_2 = binned_feature[i + 2]
        bin_3 = binned_feature[i + 3]

        out[feature_idx, bin_0].sum_gradients += all_gradients[i]
        out[feature_idx, bin_1].sum_gradients += all_gradients[i + 1]
        out[feature_idx, bin_2].sum_gradients += all_gradients[i + 2]
        out[feature_idx, bin_3].sum_gradients += all_gradients[i + 3]

        out[feature_idx, bin_0].sum_hessians += all_hessians[i]
        out[feature_idx, bin_1].sum_hessians += all_hessians[i + 1]
        out[feature_idx, bin_2].sum_hessians += all_hessians[i + 2]
        out[feature_idx, bin_3].sum_hessians += all_hessians[i + 3]

        out[feature_idx, bin_0].count += 1
        out[feature_idx, bin_1].count += 1
        out[feature_idx, bin_2].count += 1
        out[feature_idx, bin_3].count += 1

    for i in range(unrolled_upper, n_samples):
        bin_idx = binned_feature[i]
        out[feature_idx, bin_idx].sum_gradients += all_gradients[i]
        out[feature_idx, bin_idx].sum_hessians += all_hessians[i]
        out[feature_idx, bin_idx].count += 1

cpdef void _build_histogram_root_no_hessian(
        const int feature_idx,
        const X_BINNED_DTYPE_C [::1] binned_feature,  # IN
        const G_H_DTYPE_C [::1] all_gradients,  # IN
        hist_struct [:, ::1] out) noexcept nogil:  # OUT
    """Compute histogram of the root node, not updating hessians.

    Used when the hessians of the loss are constant (typically LS loss).
    """

    cdef:
        unsigned int i = 0
        unsigned int n_samples = binned_feature.shape[0]
        unsigned int unrolled_upper = (n_samples // 4) * 4

        unsigned int bin_0
        unsigned int bin_1
        unsigned int bin_2
        unsigned int bin_3
        unsigned int bin_idx

    for i in range(0, unrolled_upper, 4):
        bin_0 = binned_feature[i]
        bin_1 = binned_feature[i + 1]
        bin_2 = binned_feature[i + 2]
        bin_3 = binned_feature[i + 3]

        out[feature_idx, bin_0].sum_gradients += all_gradients[i]
        out[feature_idx, bin_1].sum_gradients += all_gradients[i + 1]
        out[feature_idx, bin_2].sum_gradients += all_gradients[i + 2]
        out[feature_idx, bin_3].sum_gradients += all_gradients[i + 3]

        out[feature_idx, bin_0].count += 1
        out[feature_idx, bin_1].count += 1
        out[feature_idx, bin_2].count += 1
        out[feature_idx, bin_3].count += 1

    for i in range(unrolled_upper, n_samples):
        bin_idx = binned_feature[i]
        out[feature_idx, bin_idx].sum_gradients += all_gradients[i]
        out[feature_idx, bin_idx].count += 1

"""
This module contains the TreeGrower class.

TreeGrower builds a regression tree fitting a Newton-Raphson step, based on
the gradients and hessians of the training data.
"""
# Author: Nicolas Hug

import numbers
from heapq import heappop, heappush
from timeit import default_timer as time

import numpy as np

from sklearn.utils._openmp_helpers import _openmp_effective_n_threads

from ._bitset import set_raw_bitset_from_binned_bitset
from .common import (
    PREDICTOR_RECORD_DTYPE,
    X_BITSET_INNER_DTYPE,
    Y_DTYPE,
    MonotonicConstraint,
)
from .histogram import HistogramBuilder
from .predictor import TreePredictor
from .splitting import Splitter, SplitInfo
from .utils import sum_parallel

EPS = np.finfo(Y_DTYPE).eps  # to avoid zero division errors

class TreeNode:
    """Tree Node class used in TreeGrower.

    This isn't used for prediction purposes, only for training (see
    TreePredictor).

    Parameters
    ----------
    depth : int
        The depth of the node, i.e. its distance from the root.
    sample_indices : ndarray of shape (n_samples_at_node,), dtype=np.uint
        The indices of the samples at the node.
    sum_gradients : float
        The sum of the gradients of the samples at the node.
    sum_hessians : float
        The sum of the hessians of the samples at the node.

    Attributes
    ----------
    depth : int
        The depth of the node, i.e. its distance from the root.
    sample_indices : ndarray of shape (n_samples_at_node,), dtype=np.uint
        The indices of the samples at the node.
    sum_gradients : float
        The sum of the gradients of the samples at the node.
    sum_hessians : float
        The sum of the hessians of the samples at the node.
    split_info : SplitInfo or None
        The result of the split evaluation.
    is_leaf : bool
        True if node is a leaf
    left_child : TreeNode or None
        The left child of the node. None for leaves.
    right_child : TreeNode or None
        The right child of the node. None for leaves.
    value : float or None
        The value of the leaf, as computed in finalize_leaf(). None for
        non-leaf nodes.
    partition_start : int
        start position of the node's sample_indices in splitter.partition.
    partition_stop : int
        stop position of the node's sample_indices in splitter.partition.
    allowed_features : None or ndarray, dtype=int
        Indices of features allowed to split for children.
    interaction_cst_indices : None or list of ints
        Indices of the interaction sets that have to be applied on splits of
        child nodes. The fewer sets the stronger the constraint as fewer sets
        contain fewer features.
    children_lower_bound : float
    children_upper_bound : float
    """

    split_info = None
    left_child = None
    right_child = None
    histograms = None

    # start and stop indices of the node in the splitter.partition
    # array. Concretely,
    # self.sample_indices = view(self.splitter.partition[start:stop])
    # Please see the comments about splitter.partition and
    # splitter.split_indices for more info about this design.
    # These 2 attributes are only used in _update_raw_prediction, because we
    # need to iterate over the leaves and I don't know how to efficiently
    # store the sample_indices views because they're all of different sizes.
    partition_start = 0
    partition_stop = 0

    def __init__(self, depth, sample_indices, sum_gradients, sum_hessians, value=None):
        self.depth = depth
        self.sample_indices = sample_indices
        self.n_samples = sample_indices.shape[0]
        self.sum_gradients = sum_gradients
        self.sum_hessians = sum_hessians
        self.value = value
        self.is_leaf = False
        self.allowed_features = None
        self.interaction_cst_indices = None
        self.set_children_bounds(float("-inf"), float("+inf"))

        self.n_used_samples

    def set_children_bounds(self, lower, upper):
        """Set children values bounds to respect monotonic constraints."""

        # These are bounds for the node's *children* values, not the node's
        # value. The bounds are used in the splitter when considering potential
        # left and right child.
        self.children_lower_bound = lower
        self.children_upper_bound = upper

    def __lt__(self, other_node):
        """Comparison for priority queue.

        Nodes with high gain are higher priority than nodes with low gain.

        heapq.heappush only need the '<' operator.
        heapq.heappop take the smallest item first (smaller is higher
        priority).

        Parameters
        ----------
        other_node : TreeNode
            The node to compare with.
        """
        return self.split_info.gain > other_node.split_info.gain


class TreeGrower:
    """Tree grower class used to build a tree.

    The tree is fitted to predict the values of a Newton-Raphson step. The
    splits are considered in a best-first fashion, and the quality of a
    split is defined in splitting._split_gain.

    Parameters
    ----------
    X_binned : ndarray of shape (n_samples, n_features), dtype=np.uint8
        The binned input samples. Must be Fortran-aligned.
    gradients : ndarray of shape (n_samples,)
        The gradients of each training sample. Those are the gradients of the
        loss w.r.t the predictions, evaluated at iteration ``i - 1``.
    hessians : ndarray of shape (n_samples,)
        The hessians of each training sample. Those are the hessians of the
        loss w.r.t the predictions, evaluated at iteration ``i - 1``.
    max_leaf_nodes : int, default=None
        The maximum number of leaves for each tree. If None, there is no
        maximum limit.
    max_depth : int, default=None
        The maximum depth of each tree. The depth of a tree is the number of
        edges to go from the root to the deepest leaf.
        Depth isn't constrained by default.
    min_samples_leaf : int, default=20
        The minimum number of samples per leaf.
    min_gain_to_split : float, default=0.
        The minimum gain needed to split a node. Splits with lower gain will
        be ignored.
    n_bins : int, default=256
        The total number of bins, including the bin for missing values. Used
        to define the shape of the histograms.
    n_bins_non_missing : ndarray, dtype=np.uint32, default=None
        For each feature, gives the number of bins actually used for
        non-missing values. For features with a lot of unique values, this
        is equal to ``n_bins - 1``. If it's an int, all features are
        considered to have the same number of bins. If None, all features
        are considered to have ``n_bins - 1`` bins.
    has_missing_values : bool or ndarray, dtype=bool, default=False
        Whether each feature contains missing values (in the training data).
        If it's a bool, the same value is used for all features.
    is_categorical : ndarray of bool of shape (n_features,), default=None
        Indicates categorical features.
    monotonic_cst : array-like of int of shape (n_features,), dtype=int, default=None
        Indicates the monotonic constraint to enforce on each feature.
          - 1: monotonic increase
          - 0: no constraint
          - -1: monotonic decrease

        Read more in the :ref:`User Guide <monotonic_cst_gbdt>`.
    interaction_cst : list of sets of integers, default=None
        List of interaction constraints.
    l2_regularization : float, default=0.
        The L2 regularization parameter.
    min_hessian_to_split : float, default=1e-3
        The minimum sum of hessians needed in each node. Splits that result in
        at least one child having a sum of hessians less than
        ``min_hessian_to_split`` are discarded.
    shrinkage : float, default=1.
        The shrinkage parameter to apply to the leaves values, also known as
        learning rate.
    n_threads : int, default=None
        Number of OpenMP threads to use. `_openmp_effective_n_threads` is called
        to determine the effective number of threads use, which takes cgroups CPU
        quotes into account. See the docstring of `_openmp_effective_n_threads`
        for details.

    Attributes
    ----------
    histogram_builder : HistogramBuilder
    splitter : Splitter
    root : TreeNode
    finalized_leaves : list of TreeNode
    splittable_nodes : list of TreeNode
    missing_values_bin_idx : int
        Equals n_bins - 1
    n_categorical_splits : int
    n_features : int
    n_nodes : int
    total_find_split_time : float
        Time spent finding the best splits
    total_compute_hist_time : float
        Time spent computing histograms
    total_apply_split_time : float
        Time spent splitting nodes
    with_monotonic_cst : bool
        Whether there are monotonic constraints that apply. False iff monotonic_cst is
        None.
    """

    def __init__(
        self,
        X_binned,
        gradients,
        hessians,
        max_leaf_nodes=None,
        max_depth=None,
        min_samples_leaf=20,
        min_gain_to_split=0.0,
        n_bins=256,
        n_bins_non_missing=None,
        has_missing_values=False,
        is_categorical=None,
        monotonic_cst=None,
        interaction_cst=None,
        l2_regularization=0.0,
        min_hessian_to_split=1e-3,
        shrinkage=1.0,
        n_threads=None,
    ):
        self._validate_parameters(
            X_binned,
            min_gain_to_split,
            min_hessian_to_split,
        )
        n_threads = _openmp_effective_n_threads(n_threads)

        if n_bins_non_missing is None:
            n_bins_non_missing = n_bins - 1

        if isinstance(n_bins_non_missing, numbers.Integral):
            n_bins_non_missing = np.array(
                [n_bins_non_missing] * X_binned.shape[1], dtype=np.uint32
            )
        else:
            n_bins_non_missing = np.asarray(n_bins_non_missing, dtype=np.uint32)

        if isinstance(has_missing_values, bool):
            has_missing_values = [has_missing_values] * X_binned.shape[1]
        has_missing_values = np.asarray(has_missing_values, dtype=np.uint8)

        # `monotonic_cst` validation is done in _validate_monotonic_cst
        # at the estimator level and therefore the following should not be
        # needed when using the public API.
        if monotonic_cst is None:
            monotonic_cst = np.full(
                shape=X_binned.shape[1],
                fill_value=MonotonicConstraint.NO_CST,
                dtype=np.int8,
            )
        else:
            monotonic_cst = np.asarray(monotonic_cst, dtype=np.int8)
        self.with_monotonic_cst = np.any(monotonic_cst != MonotonicConstraint.NO_CST)

        if is_categorical is None:
            is_categorical = np.zeros(shape=X_binned.shape[1], dtype=np.uint8)
        else:
            is_categorical = np.asarray(is_categorical, dtype=np.uint8)

        if np.any(
            np.logical_and(
                is_categorical == 1, monotonic_cst != MonotonicConstraint.NO_CST
            )
        ):
            raise ValueError("Categorical features cannot have monotonic constraints.")

        hessians_are_constant = hessians.shape[0] == 1
        self.histogram_builder = HistogramBuilder(
            X_binned, n_bins, gradients, hessians, hessians_are_constant, n_threads
        )
        missing_values_bin_idx = n_bins - 1
        self.splitter = Splitter(
            X_binned,
            n_bins_non_missing,
            missing_values_bin_idx,
            has_missing_values,
            is_categorical,
            monotonic_cst,
            l2_regularization,
            min_hessian_to_split,
            min_samples_leaf,
            min_gain_to_split,
            hessians_are_constant,
            n_threads,
        )
        self.n_bins_non_missing = n_bins_non_missing
        self.missing_values_bin_idx = missing_values_bin_idx
        self.max_leaf_nodes = max_leaf_nodes
        self.has_missing_values = has_missing_values
        self.monotonic_cst = monotonic_cst
        self.interaction_cst = interaction_cst
        self.is_categorical = is_categorical
        self.l2_regularization = l2_regularization
        self.n_features = X_binned.shape[1]
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.X_binned = X_binned
        self.min_gain_to_split = min_gain_to_split
        self.shrinkage = shrinkage
        self.n_threads = n_threads
        self.splittable_nodes = []
        self.finalized_leaves = []
        self.total_find_split_time = 0.0  # time spent finding the best splits
        self.total_compute_hist_time = 0.0  # time spent computing histograms
        self.total_apply_split_time = 0.0  # time spent splitting nodes
        self.n_categorical_splits = 0
        self._intilialize_root(gradients, hessians, hessians_are_constant)
        self.n_nodes = 1

    def _validate_parameters(
        self,
        X_binned,
        min_gain_to_split,
        min_hessian_to_split,
    ):
        """Validate parameters passed to __init__.

        Also validate parameters passed to splitter.
        """
        if X_binned.dtype != np.uint8:
            raise NotImplementedError("X_binned must be of type uint8.")
        if not X_binned.flags.f_contiguous:
            raise ValueError(
                "X_binned should be passed as Fortran contiguous "
                "array for maximum efficiency."
            )
        if min_gain_to_split < 0:
            raise ValueError(
                "min_gain_to_split={} must be positive.".format(min_gain_to_split)
            )
        if min_hessian_to_split < 0:
            raise ValueError(
                "min_hessian_to_split={} must be positive.".format(min_hessian_to_split)
            )

    def grow(self):
        """Grow the tree, from root to leaves."""
        while self.splittable_nodes:
            #self.split_mab()
            self.split_next()

        self._apply_shrinkage()

    def _apply_shrinkage(self):
        """Multiply leaves values by shrinkage parameter.

        This must be done at the very end of the growing process. If this were
        done during the growing process e.g. in finalize_leaf(), then a leaf
        would be shrunk but its sibling would potentially not be (if it's a
        non-leaf), which would lead to a wrong computation of the 'middle'
        value needed to enforce the monotonic constraints.
        """
        for leaf in self.finalized_leaves:
            leaf.value *= self.shrinkage

    def _intilialize_root(self, gradients, hessians, hessians_are_constant):
        """Initialize root node and finalize it if needed."""
        n_samples = self.X_binned.shape[0]
        depth = 0
        sum_gradients = sum_parallel(gradients, self.n_threads)
        if self.histogram_builder.hessians_are_constant:
            sum_hessians = hessians[0] * n_samples
        else:
            sum_hessians = sum_parallel(hessians, self.n_threads)
        self.root = TreeNode(
            depth=depth,
            sample_indices=self.splitter.partition,
            sum_gradients=sum_gradients,
            sum_hessians=sum_hessians,
            value=0,
        )

        self.root.partition_start = 0
        self.root.partition_stop = n_samples

        if self.root.n_samples < 2 * self.min_samples_leaf:
            # Do not even bother computing any splitting statistics.
            self._finalize_leaf(self.root)
            return
        if sum_hessians < self.splitter.min_hessian_to_split:
            self._finalize_leaf(self.root)
            return

        if self.interaction_cst is not None:
            self.root.interaction_cst_indices = range(len(self.interaction_cst))
            allowed_features = set().union(*self.interaction_cst)
            self.root.allowed_features = np.fromiter(
                allowed_features, dtype=np.uint32, count=len(allowed_features)
            )

        tic = time()
        self.root.histograms = self.histogram_builder.compute_histograms_brute(
            self.root.sample_indices[:1000], self.root.allowed_features
        )

        print(self.root.sample_indices.shape)

        n_hist_data = 0

        for histogram in self.root.histograms:
            for bin in histogram:
                n_hist_data += bin['count']

        print("before update")
        print(n_hist_data)

        self.root.histograms = self.histogram_builder.update_histogram(self.root.histograms, self.root.sample_indices[1000:], self.root.allowed_features)

        n_hist_data = 0

        for histogram in self.root.histograms:
            for bin in histogram:
                n_hist_data += bin['count']

        print("after update")
        print(n_hist_data)

        self.total_compute_hist_time += time() - tic

        tic = time()
        self._compute_best_split_and_push(self.root)
        self.total_find_split_time += time() - tic

    def _compute_best_split_and_push(self, node):
        """Compute the best possible split (SplitInfo) of a given node.

        Also push it in the heap of splittable nodes if gain isn't zero.
        The gain of a node is 0 if either all the leaves are pure
        (best gain = 0), or if no split would satisfy the constraints,
        (min_hessians_to_split, min_gain_to_split, min_samples_leaf)
        """

        """
        node.split_info = self.splitter.find_node_split(
            n_samples=node.n_samples,
            histograms=node.histograms,
            sum_gradients=node.sum_gradients,
            sum_hessians=node.sum_hessians,
            value=node.value,
            lower_bound=node.children_lower_bound,
            upper_bound=node.children_upper_bound,
            allowed_features=node.allowed_features,
        )
        """

        candidates = CandidateContainer(node)

        while candidates.n_candidates > n_epsilon_candidates and node.n_used_samples < node.n_samples:
            loss_current_node = _loss_from_value(node.value, node.sum_gradients)

            #mean update
            for i in range(candidiates.n_features):
                fidx = candidates.valid_features[i]

                sum_gradient_left, sum_hessian_left = 0., 0.
                n_samples_left = 0

                for bidx in range(candidates.n_valid_bins[fidx] - 1):
                    n_samples_left += node.histograms[fidx, bidx].count
                    sum_gradient_left += node.histograms[fidx, bidx].gradient

                    if self.hessians_are_constant:
                        sum_hessian_left += node.histograms[feature_idx, bin_idx].count
                    else:
                        sum_hessian_left += \
                            node.histograms[fidx, bidx].hessian
                        sum_hessian_right = node.sum_hessians - sum_hessian_left

                    n_samples_right = node.n_used_samples - n_samples_left
                    sum_gradient_right = node.sum_gradients - sum_gradient_right

                    candidates.candidate_stats[fidx, bidx, 0] = _split_gain(sum_gradient_left, sum_hessian_left,
                                                                            sum_gradient_right, sum_hessian_right,
                                                                            loss_current_node,
                                                                            self.monotonic_cst,
                                                                            node.children_lower_bound,
                                                                            node.children_upper_bound,
                                                                            self.l2_regularization)


            #update confidence bound
            #TODO: for each candidate in candidates, update confidence bound

            candidates.update_filter_candidates()

            #batch_indices = sample()
            #TODO: implement sample()

            #binned_batch = bin_samples(batch_indices(maybe sth else)) -> where and how to use?

            #get_gradient(batch_indices)

            #histogram_builder.set_gradient(gradient, batch_indices)

            histogram_builder.update_histograms(node.histograms, batch_indices)


        #if not decided
        #TODO: compute_exact()

        #TODO: node.split_info = SplitInfo(...)



        if node.split_info.gain <= 0:  # no valid split
            self._finalize_leaf(node)
        else:
            heappush(self.splittable_nodes, node)

    def split_next(self):
        """Split the node with highest potential gain.

        Returns
        -------
        left : TreeNode
            The resulting left child.
        right : TreeNode
            The resulting right child.
        """
        # Consider the node with the highest loss reduction (a.k.a. gain)
        node = heappop(self.splittable_nodes)
        tic = time()
        (
            sample_indices_left,
            sample_indices_right,
            right_child_pos,
        ) = self.splitter.split_indices(node.split_info, node.sample_indices)
        self.total_apply_split_time += time() - tic

        depth = node.depth + 1
        n_leaf_nodes = len(self.finalized_leaves) + len(self.splittable_nodes)
        n_leaf_nodes += 2

        left_child_node = TreeNode(
            depth,
            sample_indices_left,
            node.split_info.sum_gradient_left,
            node.split_info.sum_hessian_left,
            value=node.split_info.value_left,
        )
        right_child_node = TreeNode(
            depth,
            sample_indices_right,
            node.split_info.sum_gradient_right,
            node.split_info.sum_hessian_right,
            value=node.split_info.value_right,
        )

        node.right_child = right_child_node
        node.left_child = left_child_node

        # set start and stop indices
        left_child_node.partition_start = node.partition_start
        left_child_node.partition_stop = node.partition_start + right_child_pos
        right_child_node.partition_start = left_child_node.partition_stop
        right_child_node.partition_stop = node.partition_stop

        # set interaction constraints (the indices of the constraints sets)
        if self.interaction_cst is not None:
            # Calculate allowed_features and interaction_cst_indices only once. Child
            # nodes inherit them before they get split.
            (
                left_child_node.allowed_features,
                left_child_node.interaction_cst_indices,
            ) = self._compute_interactions(node)
            right_child_node.interaction_cst_indices = (
                left_child_node.interaction_cst_indices
            )
            right_child_node.allowed_features = left_child_node.allowed_features

        if not self.has_missing_values[node.split_info.feature_idx]:
            # If no missing values are encountered at fit time, then samples
            # with missing values during predict() will go to whichever child
            # has the most samples.
            node.split_info.missing_go_to_left = (
                left_child_node.n_samples > right_child_node.n_samples
            )

        self.n_nodes += 2
        self.n_categorical_splits += node.split_info.is_categorical

        if self.max_leaf_nodes is not None and n_leaf_nodes == self.max_leaf_nodes:
            self._finalize_leaf(left_child_node)
            self._finalize_leaf(right_child_node)
            self._finalize_splittable_nodes()
            return left_child_node, right_child_node

        if self.max_depth is not None and depth == self.max_depth:
            self._finalize_leaf(left_child_node)
            self._finalize_leaf(right_child_node)
            return left_child_node, right_child_node

        if left_child_node.n_samples < self.min_samples_leaf * 2:
            self._finalize_leaf(left_child_node)
        if right_child_node.n_samples < self.min_samples_leaf * 2:
            self._finalize_leaf(right_child_node)

        if self.with_monotonic_cst:
            # Set value bounds for respecting monotonic constraints
            # See test_nodes_values() for details
            if (
                self.monotonic_cst[node.split_info.feature_idx]
                == MonotonicConstraint.NO_CST
            ):
                lower_left = lower_right = node.children_lower_bound
                upper_left = upper_right = node.children_upper_bound
            else:
                mid = (left_child_node.value + right_child_node.value) / 2
                if (
                    self.monotonic_cst[node.split_info.feature_idx]
                    == MonotonicConstraint.POS
                ):
                    lower_left, upper_left = node.children_lower_bound, mid
                    lower_right, upper_right = mid, node.children_upper_bound
                else:  # NEG
                    lower_left, upper_left = mid, node.children_upper_bound
                    lower_right, upper_right = node.children_lower_bound, mid
            left_child_node.set_children_bounds(lower_left, upper_left)
            right_child_node.set_children_bounds(lower_right, upper_right)

        # Compute histograms of children, and compute their best possible split
        # (if needed)
        should_split_left = not left_child_node.is_leaf
        should_split_right = not right_child_node.is_leaf
        if should_split_left or should_split_right:
            # We will compute the histograms of both nodes even if one of them
            # is a leaf, since computing the second histogram is very cheap
            # (using histogram subtraction).
            n_samples_left = left_child_node.sample_indices.shape[0]
            n_samples_right = right_child_node.sample_indices.shape[0]
            if n_samples_left < n_samples_right:
                smallest_child = left_child_node
                largest_child = right_child_node
            else:
                smallest_child = right_child_node
                largest_child = left_child_node

            # We use the brute O(n_samples) method on the child that has the
            # smallest number of samples, and the subtraction trick O(n_bins)
            # on the other one.
            # Note that both left and right child have the same allowed_features.
            tic = time()
            smallest_child.histograms = self.histogram_builder.compute_histograms_brute(
                smallest_child.sample_indices, smallest_child.allowed_features
            )
            largest_child.histograms = (
                self.histogram_builder.compute_histograms_subtraction(
                    node.histograms,
                    smallest_child.histograms,
                    smallest_child.allowed_features,
                )
            )
            self.total_compute_hist_time += time() - tic

            tic = time()
            if should_split_left:
                self._compute_best_split_and_push(left_child_node)
            if should_split_right:
                self._compute_best_split_and_push(right_child_node)
            self.total_find_split_time += time() - tic

            # Release memory used by histograms as they are no longer needed
            # for leaf nodes since they won't be split.
            for child in (left_child_node, right_child_node):
                if child.is_leaf:
                    del child.histograms

        # Release memory used by histograms as they are no longer needed for
        # internal nodes once children histograms have been computed.
        del node.histograms

        return left_child_node, right_child_node

    def _compute_interactions(self, node):
        r"""Compute features allowed by interactions to be inherited by child nodes.

        Example: Assume constraints [{0, 1}, {1, 2}].
           1      <- Both constraint groups could be applied from now on
          / \
         1   2    <- Left split still fulfills both constraint groups.
        / \ / \      Right split at feature 2 has only group {1, 2} from now on.

        LightGBM uses the same logic for overlapping groups. See
        https://github.com/microsoft/LightGBM/issues/4481 for details.

        Parameters:
        ----------
        node : TreeNode
            A node that might have children. Based on its feature_idx, the interaction
            constraints for possible child nodes are computed.

        Returns
        -------
        allowed_features : ndarray, dtype=uint32
            Indices of features allowed to split for children.
        interaction_cst_indices : list of ints
            Indices of the interaction sets that have to be applied on splits of
            child nodes. The fewer sets the stronger the constraint as fewer sets
            contain fewer features.
        """
        # Note:
        #  - Case of no interactions is already captured before function call.
        #  - This is for nodes that are already split and have a
        #    node.split_info.feature_idx.
        allowed_features = set()
        interaction_cst_indices = []
        for i in node.interaction_cst_indices:
            if node.split_info.feature_idx in self.interaction_cst[i]:
                interaction_cst_indices.append(i)
                allowed_features.update(self.interaction_cst[i])
        return (
            np.fromiter(allowed_features, dtype=np.uint32, count=len(allowed_features)),
            interaction_cst_indices,
        )

    def _finalize_leaf(self, node):
        """Make node a leaf of the tree being grown."""

        node.is_leaf = True
        self.finalized_leaves.append(node)

    def _finalize_splittable_nodes(self):
        """Transform all splittable nodes into leaves.

        Used when some constraint is met e.g. maximum number of leaves or
        maximum depth."""
        while len(self.splittable_nodes) > 0:
            node = self.splittable_nodes.pop()
            self._finalize_leaf(node)

    def make_predictor(self, binning_thresholds):
        """Make a TreePredictor object out of the current tree.

        Parameters
        ----------
        binning_thresholds : array-like of floats
            Corresponds to the bin_thresholds_ attribute of the BinMapper.
            For each feature, this stores:

            - the bin frontiers for continuous features
            - the unique raw category values for categorical features

        Returns
        -------
        A TreePredictor object.
        """
        predictor_nodes = np.zeros(self.n_nodes, dtype=PREDICTOR_RECORD_DTYPE)
        binned_left_cat_bitsets = np.zeros(
            (self.n_categorical_splits, 8), dtype=X_BITSET_INNER_DTYPE
        )
        raw_left_cat_bitsets = np.zeros(
            (self.n_categorical_splits, 8), dtype=X_BITSET_INNER_DTYPE
        )
        _fill_predictor_arrays(
            predictor_nodes,
            binned_left_cat_bitsets,
            raw_left_cat_bitsets,
            self.root,
            binning_thresholds,
            self.n_bins_non_missing,
        )
        return TreePredictor(
            predictor_nodes, binned_left_cat_bitsets, raw_left_cat_bitsets
        )

    def _sample_batch(self, batch_size):
        print("need to implement sample_batch")
        return None

    def split_mab(self):
        """Split the node with highest potential gain.

                Returns
                -------
                left : TreeNode
                    The resulting left child.
                right : TreeNode
                    The resulting right child.
                """
        # Consider the node with the highest loss reduction (a.k.a. gain)
        node = heappop(self.splittable_nodes)

        tic = time()
        (
            sample_indices_left,
            sample_indices_right,
            right_child_pos,
        ) = self.splitter.split_indices(node.split_info, node.sample_indices)
        self.total_apply_split_time += time() - tic

        depth = node.depth + 1
        n_leaf_nodes = len(self.finalized_leaves) + len(self.splittable_nodes)
        n_leaf_nodes += 2

        left_child_node = TreeNode(
            depth,
            sample_indices_left,
            node.split_info.sum_gradient_left,
            node.split_info.sum_hessian_left,
            value=node.split_info.value_left,
        )
        right_child_node = TreeNode(
            depth,
            sample_indices_right,
            node.split_info.sum_gradient_right,
            node.split_info.sum_hessian_right,
            value=node.split_info.value_right,
        )

        node.right_child = right_child_node
        node.left_child = left_child_node

        # set start and stop indices
        left_child_node.partition_start = node.partition_start
        left_child_node.partition_stop = node.partition_start + right_child_pos
        right_child_node.partition_start = left_child_node.partition_stop
        right_child_node.partition_stop = node.partition_stop

        # set interaction constraints (the indices of the constraints sets)
        if self.interaction_cst is not None:
            # Calculate allowed_features and interaction_cst_indices only once. Child
            # nodes inherit them before they get split.
            (
                left_child_node.allowed_features,
                left_child_node.interaction_cst_indices,
            ) = self._compute_interactions(node)
            right_child_node.interaction_cst_indices = (
                left_child_node.interaction_cst_indices
            )
            right_child_node.allowed_features = left_child_node.allowed_features

        if not self.has_missing_values[node.split_info.feature_idx]:
            # If no missing values are encountered at fit time, then samples
            # with missing values during predict() will go to whichever child
            # has the most samples.
            node.split_info.missing_go_to_left = (
                    left_child_node.n_samples > right_child_node.n_samples
            )

        self.n_nodes += 2
        self.n_categorical_splits += node.split_info.is_categorical

        if self.max_leaf_nodes is not None and n_leaf_nodes == self.max_leaf_nodes:
            self._finalize_leaf(left_child_node)
            self._finalize_leaf(right_child_node)
            self._finalize_splittable_nodes()
            return left_child_node, right_child_node

        if self.max_depth is not None and depth == self.max_depth:
            self._finalize_leaf(left_child_node)
            self._finalize_leaf(right_child_node)
            return left_child_node, right_child_node

        if left_child_node.n_samples < self.min_samples_leaf * 2:
            self._finalize_leaf(left_child_node)
        if right_child_node.n_samples < self.min_samples_leaf * 2:
            self._finalize_leaf(right_child_node)

        if self.with_monotonic_cst:
            # Set value bounds for respecting monotonic constraints
            # See test_nodes_values() for details
            if (
                    self.monotonic_cst[node.split_info.feature_idx]
                    == MonotonicConstraint.NO_CST
            ):
                lower_left = lower_right = node.children_lower_bound
                upper_left = upper_right = node.children_upper_bound
            else:
                mid = (left_child_node.value + right_child_node.value) / 2
                if (
                        self.monotonic_cst[node.split_info.feature_idx]
                        == MonotonicConstraint.POS
                ):
                    lower_left, upper_left = node.children_lower_bound, mid
                    lower_right, upper_right = mid, node.children_upper_bound
                else:  # NEG
                    lower_left, upper_left = mid, node.children_upper_bound
                    lower_right, upper_right = node.children_lower_bound, mid
            left_child_node.set_children_bounds(lower_left, upper_left)
            right_child_node.set_children_bounds(lower_right, upper_right)

        # Compute histograms of children, and compute their best possible split
        # (if needed)
        should_split_left = not left_child_node.is_leaf
        should_split_right = not right_child_node.is_leaf



        #TODO: implement statistics carryover from parent to child
        if should_split_left:
            """
            candidates: structure of size (n_features * max_bins * 2)
            Each cell indexed by (feature_idx, bin_idx) is going to contain 'mean value' and 'confidence bound'
            """

            used_samples = 0

            while candidates.candidate_n > 1 and used_samples < left_child_node.n_samples:


                """
                left_child_node.histograms = self.histogram_builder.compute_histograms_brute(
                    smallest_child.sample_indices, smallest_child.allowed_features
                )
                """

                #batch = self._sample_batch(batch_size)

                #TODO: 1. need to integrate this with BinMapper to build bins with inital batch
                #      2. figure out how to handle allowed_features(maybe don't need to modify?)
                if used_samples == 0:


                    left_child_node.histograms = self.histogram_builder.compute_histograms_brute(
                        batch, left_child_node.allowed_features
                    )
                else:
                    left_child_node.histograms = self.histogram_builder.update_histogram(
                        left_child_node.histograms, batch, left_child_node.allowed_features
                    )

                # TODO: reference of candidate must be passed here. Needs to be modified when migrating to cython.
                #self._update_candidates(candidates, left_child_node.histograms)
                """
                node.split_info = self.splitter.find_node_split(
                    n_samples=node.n_samples,
                    histograms=node.histograms,
                    sum_gradients=node.sum_gradients,
                    sum_hessians=node.sum_hessians,
                    value=node.value,
                    lower_bound=node.children_lower_bound,
                    upper_bound=node.children_upper_bound,
                    allowed_features=node.allowed_features,
                )

                TODO: replace this with iterative MAB scheme(probably a place for updating confidence bound?)
                """

                """
                if candidates.candidate_n == 1:
                    return candidate
                elif used_samples >= left_child_node.n_samples:
                    self._calcuate_exact(...)
                """



        if should_split_right:
            right_child_node.histograms = self.histogram_builder.compute_histograms_brute(
                smallest_child.sample_indices, smallest_child.allowed_features
            )

        def update_candidates(candidates, histograms):
            for candidate in candidates.candidates:
                fidx = candidate.fidx
                bidx = candidate.bidx

                candidates.stats[fidx][bidx].gain = split_gain(...)
                candidates.stats[fidx][bidx].confidence_bound = calc_confidence_bound(...)

                if gain > candidate.best_gain:
                    pass
                    #TODO: replace best_candidate with current candidate

            new_candidates = List()
            best_pessimistic_estimate = best_gain - best_confidence_bound

            for candidate in candidates.candidates:
                fidx = candidate.fidx
                bidx = candidate.bidx
                gain = candidates.stats[fidx][bidx].gain
                confidence_bound = candidates.stats[fidx][bidx]

                if gain + confidence_bound >= best_pessimistic_estimate:
                    new_candidates.append(candidate)

            candidates.set_candidates(new_candidates)


def _fill_predictor_arrays(
    predictor_nodes,
    binned_left_cat_bitsets,
    raw_left_cat_bitsets,
    grower_node,
    binning_thresholds,
    n_bins_non_missing,
    next_free_node_idx=0,
    next_free_bitset_idx=0,
):
    """Helper used in make_predictor to set the TreePredictor fields."""
    node = predictor_nodes[next_free_node_idx]
    node["count"] = grower_node.n_samples
    node["depth"] = grower_node.depth
    if grower_node.split_info is not None:
        node["gain"] = grower_node.split_info.gain
    else:
        node["gain"] = -1

    node["value"] = grower_node.value

    if grower_node.is_leaf:
        # Leaf node
        node["is_leaf"] = True
        return next_free_node_idx + 1, next_free_bitset_idx

    split_info = grower_node.split_info
    feature_idx, bin_idx = split_info.feature_idx, split_info.bin_idx
    node["feature_idx"] = feature_idx
    node["bin_threshold"] = bin_idx
    node["missing_go_to_left"] = split_info.missing_go_to_left
    node["is_categorical"] = split_info.is_categorical

    if split_info.bin_idx == n_bins_non_missing[feature_idx] - 1:
        # Split is on the last non-missing bin: it's a "split on nans".
        # All nans go to the right, the rest go to the left.
        # Note: for categorical splits, bin_idx is 0 and we rely on the bitset
        node["num_threshold"] = np.inf
    elif split_info.is_categorical:
        categories = binning_thresholds[feature_idx]
        node["bitset_idx"] = next_free_bitset_idx
        binned_left_cat_bitsets[next_free_bitset_idx] = split_info.left_cat_bitset
        set_raw_bitset_from_binned_bitset(
            raw_left_cat_bitsets[next_free_bitset_idx],
            split_info.left_cat_bitset,
            categories,
        )
        next_free_bitset_idx += 1
    else:
        node["num_threshold"] = binning_thresholds[feature_idx][bin_idx]

    next_free_node_idx += 1

    node["left"] = next_free_node_idx
    next_free_node_idx, next_free_bitset_idx = _fill_predictor_arrays(
        predictor_nodes,
        binned_left_cat_bitsets,
        raw_left_cat_bitsets,
        grower_node.left_child,
        binning_thresholds=binning_thresholds,
        n_bins_non_missing=n_bins_non_missing,
        next_free_node_idx=next_free_node_idx,
        next_free_bitset_idx=next_free_bitset_idx,
    )

    node["right"] = next_free_node_idx
    return _fill_predictor_arrays(
        predictor_nodes,
        binned_left_cat_bitsets,
        raw_left_cat_bitsets,
        grower_node.right_child,
        binning_thresholds=binning_thresholds,
        n_bins_non_missing=n_bins_non_missing,
        next_free_node_idx=next_free_node_idx,
        next_free_bitset_idx=next_free_bitset_idx,
    )


cdef inline Y_DTYPE_C _split_gain(
        Y_DTYPE_C sum_gradient_left,
        Y_DTYPE_C sum_hessian_left,
        Y_DTYPE_C sum_gradient_right,
        Y_DTYPE_C sum_hessian_right,
        Y_DTYPE_C loss_current_node,
        signed char monotonic_cst,
        Y_DTYPE_C lower_bound,
        Y_DTYPE_C upper_bound,
        Y_DTYPE_C l2_regularization) noexcept nogil:
    """Loss reduction

    Compute the reduction in loss after taking a split, compared to keeping
    the node a leaf of the tree.

    See Equation 7 of:
    :arxiv:`T. Chen, C. Guestrin, (2016) XGBoost: A Scalable Tree Boosting System,
    <1603.02754>.`
    """
    cdef:
        Y_DTYPE_C gain
        Y_DTYPE_C value_left
        Y_DTYPE_C value_right

    # Compute values of potential left and right children
    value_left = compute_node_value(sum_gradient_left, sum_hessian_left,
                                    lower_bound, upper_bound,
                                    l2_regularization)
    value_right = compute_node_value(sum_gradient_right, sum_hessian_right,
                                     lower_bound, upper_bound,
                                     l2_regularization)

    if ((monotonic_cst == MonotonicConstraint.POS and value_left > value_right) or
            (monotonic_cst == MonotonicConstraint.NEG and value_left < value_right)):
        # don't consider this split since it does not respect the monotonic
        # constraints. Note that these comparisons need to be done on values
        # that have already been clipped to take the monotonic constraints into
        # account (if any).
        return -1

    gain = loss_current_node
    gain -= _loss_from_value(value_left, sum_gradient_left)
    gain -= _loss_from_value(value_right, sum_gradient_right)
    # Note that for the gain to be correct (and for min_gain_to_split to work
    # as expected), we need all values to be bounded (current node, left child
    # and right child).

    return gain

cdef inline Y_DTYPE_C _loss_from_value(
        Y_DTYPE_C value,
        Y_DTYPE_C sum_gradient) noexcept nogil:
    """Return loss of a node from its (bounded) value

    See Equation 6 of:
    :arxiv:`T. Chen, C. Guestrin, (2016) XGBoost: A Scalable Tree Boosting System,
    <1603.02754>.`
    """
    return sum_gradient * value

class CandidateContainer:
    def __init__(self, node):
        n_bins = node.histograms.shape[1] 
        self.n_features = node.histograms.shape[0]
        self.n_valid_bins = np.zeros((n_features)) + n_bins
        self.valid_features = np.arange((self.n_features))

        self.candidate_stats = np.zeros((n_features, n_bins, 2)) # (mean, confidence_bound) pair array of size (n_features, n_bins)
        self.n_candidates = n_features * n_bins # works where missing doesn't exist
        self.best_candidate = self.find_min_candidate(is_upper=True) # best candidate's (fidx, bidx)

    def find_min_candidate(self, is_upper=False):
        ci = 0
        min_fidx = 0
        min_bidx = 0
        best = self.candidate_stats[0, 0, 0] + self.candidate_stats[0, 0, 1]

        for i in range(self.n_features):
            fidx = self.valid_features[i]
            for bidx in range(self.n_valid_bins[fidx]):
                if is_upper: 
                    ci = self.candidate_stats[fidx, bidx, 1]
                curr = self.candidate_stats[fidx, bidx, 0] + ci

                if curr < best:
                    min_fidx = fidx
                    min_bidx = bidx

        return {min_fidx, min_bidx}


    def filter_candidates(self):
        min_upper_bound = self.candidate_stats[best_candidate, 0] + self.candidate_stats[best_candidate, 1]

        for i in range(self.n_features - 1, -1, -1):
            fidx = self.valid_features[i]
            for bidx in range(self.n_valid_bins[fidx] - 1, -1, -1):
                lower_bound = self.candidate_stats[fidx, bidx, 0] - self.candidate_stats[fidx, bidx, 1]

                # arm gets eliminated
                if lower_bound >= min_upper_bound: 
                    self.n_valid_bins[fidx] -= 1
                    self.n_candidates -= 1

                    if self.n_valid_bins[fidx] == 0:
                        self.n_features -= 1
                        self.valid_features[i] = self.valid_features[self.n_features]
                    else:
                        self.candidate_stats[fidx, bidx] = self.candidate_stats[fidx, self.n_valid_bins[fidx]]

    def update_filter_candidates(self):
        self.best_candidate = self.find_min_upper_bound()
        self.filter_candidates()
