import pytest
import numpy as np
from biom3d.preprocess import correct_mask

class TestCorrectMask:
    """Groups all tests for the correct_mask function."""

    # --- 1. Tests for Valid Masks (No Correction Needed) ---

    def test_valid_3d_label_mask(self):
        """Tests a perfect 3D label mask, which should not be changed."""
        mask = np.array([[[0, 1], [2, 0]]], dtype=np.uint8)
        processed_mask = correct_mask(mask, num_classes=3)
        np.testing.assert_array_equal(processed_mask, mask)

    def test_valid_2d_label_mask(self):
        """Tests a perfect 2D label mask with is_2d=True."""
        mask = np.array([[0, 1], [2, 0]], dtype=np.uint8)
        processed_mask = correct_mask(mask, num_classes=3, is_2d=True)
        assert processed_mask.ndim == 2
        np.testing.assert_array_equal(processed_mask, mask)

    def test_valid_4d_binary_mask(self):
        """Tests a perfect 4D binary mask, which should not be changed."""
        mask = np.zeros((2, 1, 4, 4), dtype=np.uint8)
        mask[0, 0, :2, :2] = 1
        processed_mask = correct_mask(mask, num_classes=2, encoding_type='binary')
        np.testing.assert_array_equal(processed_mask, mask)

    def test_valid_3d_onehot_mask_2d_data(self):
        """Tests a perfect one-hot mask for 2D data (shape C,H,W)."""
        mask = np.array([
            [[1, 0], [0, 0]], # Class 0
            [[0, 1], [0, 1]], # Class 1
            [[0, 0], [1, 0]], # Class 2
        ], dtype=np.uint8)
        processed_mask = correct_mask(mask, num_classes=3, is_2d=True, keep_original_dims=True, encoding_type='onehot')
        assert processed_mask.ndim == 3
        np.testing.assert_array_equal(processed_mask, mask)

    # --- 2. Tests for Correction Logic ---

    def test_correction_remap_3d_label(self):
        """Tests remapping of non-contiguous labels in a 3D mask."""
        mask = np.array([[[10, 20], [30, 10]]], dtype=np.uint8)
        expected = np.array([[[0, 1], [2, 0]]], dtype=np.uint8)
        processed_mask = correct_mask(mask, num_classes=3)
        np.testing.assert_array_equal(processed_mask, expected)
        
    def test_correction_remap_2d_label(self):
        """Tests remapping of non-contiguous labels in a 2D mask."""
        mask = np.array([[10, 20], [30, 10]], dtype=np.uint8)
        expected = np.array([[0, 1], [2, 0]], dtype=np.uint8)
        processed_mask = correct_mask(mask, num_classes=3, is_2d=True)
        np.testing.assert_array_equal(processed_mask, expected)

    def test_correction_binary_heuristic(self):
        """Tests binary correction where the majority value becomes background (0)."""
        # 50 is the most frequent value, so it should become 0. 100 becomes 1.
        mask = np.array([50, 50, 50, 100], dtype=np.uint8).reshape(1, 2, 2)
        expected = np.array([0, 0, 0, 1], dtype=np.uint8).reshape(1, 2, 2)
        processed_mask = correct_mask(mask, num_classes=2)
        np.testing.assert_array_equal(processed_mask, expected)

    def test_correction_4d_binary_per_channel(self):
        """Tests that correction is applied independently to each channel of a binary mask."""
        mask = np.zeros((2, 1, 2, 2), dtype=np.uint8)
        mask[0, 0, 0, 0] = 1          # Channel 0 is already valid: [0, 1]
        mask[1, 0, :, :] = 50         # Channel 1 is invalid: [0, 50]
        mask[1, 0, 0, 0] = 0          # Make 50 the majority in channel 1

        expected = np.zeros_like(mask)
        expected[0, 0, 0, 0] = 1      # Channel 0 is unchanged
        expected[1, 0, :, :] = 0      # Channel 1: 50 becomes 0
        expected[1, 0, 0, 0] = 1      # Channel 1: 0 becomes 1
        
        processed_mask = correct_mask(mask, num_classes=2, encoding_type='binary')
        np.testing.assert_array_equal(processed_mask, expected)

    # --- 3. Tests for Error Handling and Overrides ---

    def test_auto_correct_false_raises_error(self):
        """Ensures an invalid mask raises an error when auto_correct is False."""
        invalid_mask = np.array([10, 20, 30], dtype=np.uint8).reshape(1, 1, 3)
        with pytest.raises(RuntimeError, match="Mask is invalid and auto_correct is False."):
            correct_mask(invalid_mask, num_classes=3, auto_correct=False)
            
    def test_ambiguous_label_correction_raises_error(self):
        """Tests that an error is raised for uncorrectable label ambiguities."""
        # 4 unique labels, but user expects 3 classes. This is ambiguous.
        ambiguous_mask = np.array([10, 20, 30, 40], dtype=np.uint8).reshape(1, 2, 2)
        with pytest.raises(RuntimeError):
            correct_mask(ambiguous_mask, num_classes=3)
            
    def test_invalid_onehot_raises_error(self):
        """Tests that a broken one-hot mask raises an error, as correction is unsupported."""
        # This mask is not one-hot because the sum at [0,0] is 2.
        broken_onehot = np.array([
            [[1, 0], [0, 0]],
            [[1, 1], [0, 1]], # Invalid value here
            [[0, 0], [1, 0]],
        ], dtype=np.uint8)
        with pytest.raises(RuntimeError):
            correct_mask(broken_onehot, num_classes=3, is_2d=True, keep_original_dims=True, encoding_type='onehot')

    def test_invalid_num_classes_raises_error(self):
        """Tests that num_classes < 2 raises an error."""
        mask = np.zeros((2, 2), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="num_classes must be an integer >= 2"):
            correct_mask(mask, num_classes=1, is_2d=True)

    def test_is_2d_with_wrong_ndim_raises_error(self):
        """Tests that using is_2d=True with an unsupported ndim (e.g., 4D) fails."""
        mask = np.zeros((1, 1, 1, 1), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="expected mask.ndim to be 2 or 3"):
            correct_mask(mask, num_classes=2, is_2d=True)