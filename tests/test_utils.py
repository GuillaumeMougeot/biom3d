import pytest
import numpy as np
from biom3d.utils import one_hot_fast

# Helper to create expected outputs for clarity
def create_expected(channels):
    return np.array(channels, dtype=np.uint8)

class TestOneHotFast:

    # --- Tests for `mapping_mode='strict'` (the default) ---

    def test_strict_valid_input(self):
        """Tests 'strict' mode with perfect, contiguous input."""
        values = np.array([[0, 1], [2, 0]])
        actual = one_hot_fast(values, num_classes=3, mapping_mode='strict')
        expected = create_expected([
            [[1, 0], [0, 1]],  # Class 0
            [[0, 1], [0, 0]],  # Class 1
            [[0, 0], [1, 0]],  # Class 2
        ])
        np.testing.assert_array_equal(actual, expected)

    def test_strict_raises_error_on_high_value(self):
        """Tests 'strict' mode fails if a value is >= num_classes."""
        values = np.array([0, 1, 3]) # 3 is out of bounds for num_classes=3
        with pytest.raises(ValueError, match=r"must be in \[0, 2\]"):
            one_hot_fast(values, num_classes=3, mapping_mode='strict')

    def test_strict_raises_error_on_negative_value(self):
        """Tests 'strict' mode fails if a value is negative."""
        values = np.array([-1, 0, 1])
        with pytest.raises(ValueError, match=r"must be in \[0, 2\]"):
            one_hot_fast(values, num_classes=3, mapping_mode='strict')

    def test_strict_allows_missing_labels(self):
        """Tests 'strict' mode works correctly even if some valid labels are missing."""
        values = np.array([0, 2, 0]) # Label 1 is missing, but this is allowed
        actual = one_hot_fast(values, num_classes=3, mapping_mode='strict')
        expected = create_expected([
            [1, 0, 1], # Class 0
            [0, 0, 0], # Class 1 (empty)
            [0, 1, 0], # Class 2
        ])
        np.testing.assert_array_equal(actual, expected)
        
    # --- Tests for `mapping_mode='pad'` ---

    def test_pad_valid_input(self):
        """Tests 'pad' mode works like 'strict' for valid, sparse input."""
        values = np.array([0, 3]) # Labels 1, 2 are missing
        actual = one_hot_fast(values, num_classes=4, mapping_mode='pad')
        expected = create_expected([
            [1, 0], # Class 0
            [0, 0], # Class 1 (padded)
            [0, 0], # Class 2 (padded)
            [0, 1], # Class 3
        ])
        np.testing.assert_array_equal(actual, expected)

    def test_pad_raises_error_on_high_value(self):
        """Tests 'pad' mode fails if a value is out of the [0, num_classes-1] range."""
        values = np.array([0, 4]) # 4 is out of bounds for num_classes=4
        with pytest.raises(ValueError, match=r"must be in \[0, 3\]"):
            one_hot_fast(values, num_classes=4, mapping_mode='pad')

    def test_pad_raises_error_on_negative_value(self):
        """Tests 'pad' mode fails if a value is negative."""
        values = np.array([-1, 0, 3])
        with pytest.raises(ValueError, match=r"must be in \[0, 3\]"):
            one_hot_fast(values, num_classes=4, mapping_mode='pad')

    # --- Tests for `mapping_mode='remap'` ---

    def test_remap_valid_input(self):
        """Tests 'remap' mode with arbitrary, non-contiguous labels."""
        values = np.array([10, 50, 100, 10]) # Remaps 10->0, 50->1, 100->2
        actual = one_hot_fast(values, num_classes=3, mapping_mode='remap')
        expected = create_expected([
            [1, 0, 0, 1], # Class 0 (original 10)
            [0, 1, 0, 0], # Class 1 (original 50)
            [0, 0, 1, 0], # Class 2 (original 100)
        ])
        np.testing.assert_array_equal(actual, expected)

    def test_remap_raises_error_on_mismatched_counts(self):
        """Tests 'remap' mode fails if len(unique) != num_classes."""
        values = np.array([10, 50, 100]) # 3 unique values
        # We expect 4 classes, this is an ambiguity 'remap' should not solve.
        with pytest.raises(ValueError, match=r"number of unique values \(3\) must equal num_classes \(4\)"):
            one_hot_fast(values, num_classes=4, mapping_mode='remap')

    def test_remap_with_negative_numbers(self):
        """Tests 'remap' mode works correctly with negative and zero labels."""
        values = np.array([-5, 0, 10, 0]) # Remaps -5->0, 0->1, 10->2
        actual = one_hot_fast(values, num_classes=3, mapping_mode='remap')
        expected = create_expected([
            [1, 0, 0, 0], # Class 0 (original -5)
            [0, 1, 0, 1], # Class 1 (original 0)
            [0, 0, 1, 0], # Class 2 (original 10)
        ])
        np.testing.assert_array_equal(actual, expected)
        
    # --- Tests for `num_classes=None` (Inference Mode) ---
    
    def test_infer_num_classes(self):
        """Tests that num_classes=None correctly infers classes and uses 'remap'."""
        values = np.array([10, 50, 100, 10])
        # Should behave exactly like the test_remap_valid_input test
        actual = one_hot_fast(values, num_classes=None)
        expected = create_expected([
            [1, 0, 0, 1], # Class 0 (original 10)
            [0, 1, 0, 0], # Class 1 (original 50)
            [0, 0, 1, 0], # Class 2 (original 100)
        ])
        assert actual.shape[0] == 3
        np.testing.assert_array_equal(actual, expected)

    def test_infer_num_classes_with_contiguous_labels(self):
        """Tests inference mode with simple, contiguous labels."""
        values = np.array([0, 1, 2, 0])
        actual = one_hot_fast(values, num_classes=None)
        expected = create_expected([
            [1, 0, 0, 1], # Class 0
            [0, 1, 0, 0], # Class 1
            [0, 0, 1, 0], # Class 2
        ])
        assert actual.shape[0] == 3
        np.testing.assert_array_equal(actual, expected)

    # --- Edge Case Tests ---

    def test_empty_input_array(self):
        """Tests behavior with an empty input array."""
        values = np.array([], dtype=int)
        actual = one_hot_fast(values, num_classes=3)
        # Expected shape is (3, 0)
        assert actual.shape == (3, 0)
        np.testing.assert_array_equal(actual, np.empty((3, 0), dtype=np.uint8))

    def test_empty_input_with_remap(self):
        """Tests remap mode with an empty input array."""
        values = np.array([], dtype=int)
        # num_classes must be 0 for this to be valid
        actual = one_hot_fast(values, num_classes=0, mapping_mode='remap')
        assert actual.shape == (0, 0)
        
        # Test that it fails if num_classes is not 0
        with pytest.raises(ValueError, match="unique values (0) must equal num_classes (1)"):
            one_hot_fast(values, num_classes=1, mapping_mode='remap')