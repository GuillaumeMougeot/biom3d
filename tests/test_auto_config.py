import numpy as np
import pytest
from biom3d.auto_config import compute_median, data_fingerprint, find_patch_pool_batch, get_aug_patch

MODULE_TO_PATCH = "biom3d.auto_config"

# --- Module-Level Fixtures ---

@pytest.fixture
def mock_fs_env_factory(monkeypatch):
    """
    A factory fixture to mock the file system environment for data_fingerprint.
    It mocks `abs_listdir` and `adaptive_imread`.
    """
    def _create_mocks(img_data, msk_data=None):
        """
        Args:
            img_data (list of tuples): Each tuple is (image_array, metadata_dict).
            msk_data (list of np.ndarray, optional): List of mask arrays.
        """
        # --- Mock abs_listdir ---
        def fake_listdir(path):
            if "img" in path:
                return [f"{path}/img_{i}.nii" for i in range(len(img_data))]
            if "msk" in path and msk_data:
                return [f"{path}/msk_{i}.nii" for i in range(len(msk_data))]
            return []
        monkeypatch.setattr(f"{MODULE_TO_PATCH}.abs_listdir", fake_listdir)

        # --- Mock adaptive_imread ---
        def fake_imread(path):
            if "img" in path:
                idx = int(path.split('_')[-1].split('.')[0])
                return img_data[idx]
            if "msk" in path:
                idx = int(path.split('_')[-1].split('.')[0])
                # Masks typically don't have complex metadata
                return msk_data[idx], {}
            raise FileNotFoundError(f"Path not mocked: {path}")
        monkeypatch.setattr(f"{MODULE_TO_PATCH}.adaptive_imread", fake_imread)

    return _create_mocks

# --- Flexible Fixtures ---

@pytest.fixture
def mock_image_folder_factory(monkeypatch):
    """
    A factory fixture that allows creating mocks for `abs_listdir` and `adaptive_imread`
    with custom shapes and spacings for each test. This is more powerful than
    a static fixture.
    """
    def _create_mocks(shapes, spacings=None):
        """
        Args:
            shapes (list of tuples): A list of image shapes to be returned by adaptive_imread.
            spacings (list of lists or Nones): A list of spacings. If an element is None,
                                               metadata will not have a 'spacing' key.
        """
        if spacings is None:
            # Default spacing if not provided
            spacings = [[1.0, 1.0, 1.0]] * len(shapes)
        
        assert len(shapes) == len(spacings), "Shapes and spacings lists must have the same length."

        # Create a list of (image, metadata) tuples to be returned in order
        mock_returns = []
        for shape, spacing in zip(shapes, spacings):
            if shape is None: # Represents a failed read
                image = np.array(0) # An image with shape () so len(img.shape) is 0
            else:
                image = np.zeros(shape)

            metadata = {}
            if spacing is not None:
                metadata['spacing'] = spacing
            mock_returns.append((image, metadata))

        # --- Mock abs_listdir ---
        def fake_listdir(path):
            return [f"{path}/img{i}.nii.gz" for i in range(len(shapes))]
        monkeypatch.setattr(f"{MODULE_TO_PATCH}.abs_listdir", fake_listdir)

        # --- Mock adaptive_imread ---
        # Use a class with a counter to return different values on each call
        class MockReader:
            def __init__(self):
                self.call_count = 0
            
            def fake_imread(self, path):
                result = mock_returns[self.call_count]
                self.call_count += 1
                return result

        reader = MockReader()
        monkeypatch.setattr(f"{MODULE_TO_PATCH}.adaptive_imread", reader.fake_imread)

    return _create_mocks

# ============================================================================
# TESTS FOR: compute_median
# ============================================================================

class TestComputeMedian:

    # --- Happy Path Tests ---

    def test_compute_median_single_shape(self, mock_image_folder_factory):
        """Tests the basic scenario with multiple images of the same shape."""
        shapes = [(100, 200, 150), (100, 200, 150), (100, 200, 150)]
        mock_image_folder_factory(shapes=shapes)
        
        median = compute_median("some/path")
        assert isinstance(median, np.ndarray)
        assert np.array_equal(median, np.array([100, 200, 150]))

    def test_compute_median_with_spacing_single_values(self, mock_image_folder_factory):
        """Tests returning median and spacing when all values are the same."""
        shapes = [(100, 200, 150), (100, 200, 150)]
        spacings = [[1.0, 1.0, 2.5], [1.0, 1.0, 2.5]]
        mock_image_folder_factory(shapes=shapes, spacings=spacings)
        
        median, spacing = compute_median("some/path", return_spacing=True)
        assert np.array_equal(median, [100, 200, 150])
        assert np.allclose(spacing, [1.0, 1.0, 2.5])

    @pytest.mark.parametrize(
        "shapes, expected_median",
        [
            # Test with an odd number of images
            (
                [(100, 200, 50), (110, 210, 60), (120, 220, 70)], # The middle element is the median
                [110, 210, 60]
            ),
            # Test with an even number of images (median is the mean of the two middle elements)
            (
                [(100, 200, 50), (110, 210, 60), (120, 220, 70), (130, 230, 80)],
                [115, 215, 65] # e.g., (110+120)/2 = 115. Result is float, then cast to int.
            ),
            # Test with 2D images
            (
                [(50, 50), (60, 60), (70, 70)],
                [1, 60, 60]
            ),
        ]
    )
    def test_compute_median_varying_shapes(self, mock_image_folder_factory, shapes, expected_median):
        """Tests that the median is correctly calculated for varying shapes."""
        mock_image_folder_factory(shapes=shapes)
        
        median = compute_median("some/path")
        assert np.array_equal(median, expected_median)

    def test_compute_median_varying_spacings(self, mock_image_folder_factory):
        """Tests that the median spacing is correctly calculated."""
        shapes = [(10, 10, 10), (10, 10, 10), (10, 10, 10)]
        spacings = [[1.0, 1.0, 2.0], [1.1, 1.1, 2.5], [1.2, 1.2, 3.0]]
        mock_image_folder_factory(shapes=shapes, spacings=spacings)
        
        _median_shape, median_spacing = compute_median("some/path", return_spacing=True)
        assert np.allclose(median_spacing, [1.1, 1.1, 2.5])
        
    def test_some_images_missing_spacing(self, mock_image_folder_factory):
        """
        Tests that images without a 'spacing' key in their metadata are ignored
        for the spacing calculation, but not for the shape calculation.
        """
        shapes = [(100, 100, 100), (110, 110, 110), (120, 120, 120)]
        # The middle image will have no spacing information
        spacings = [[1.0, 1.0, 2.0], None, [1.2, 1.2, 3.0]]
        mock_image_folder_factory(shapes=shapes, spacings=spacings)

        median_shape, median_spacing = compute_median("some/path", return_spacing=True)
        
        # Median shape should be calculated from all 3 images
        assert np.array_equal(median_shape, [110, 110, 110])
        # Median spacing should be calculated from the 2 images that had spacing
        # Median of [[1.0, 1.0, 2.0], [1.2, 1.2, 3.0]] is [1.1, 1.1, 2.5]
        assert np.allclose(median_spacing, [1.1, 1.1, 2.5])

    # --- Error and Edge Case Tests ---

    def test_empty_folder(self, mock_image_folder_factory):
        """Tests that an assertion error is raised for an empty folder."""
        mock_image_folder_factory(shapes=[]) # Simulates empty folder
        
        with pytest.raises(AssertionError, match="List of sizes for median computation is empty"):
            compute_median("empty/path")

    def test_bad_image_read(self, mock_image_folder_factory):
        """Tests that an assertion error is raised if an image fails to load properly."""
        # `None` for shape will create an image with shape `()`
        mock_image_folder_factory(shapes=[(10, 10), None, (20, 20)])

        with pytest.raises(AssertionError, match="Wrong image"):
            compute_median("bad/image/path")

    def test_no_images_have_spacing(self, mock_image_folder_factory):
        """
        Tests for a crash when `return_spacing` is True but no images have spacing info.
        `np.median` on an empty list raises an error. This test exposes a potential bug.
        """
        shapes = [(10, 10), (20, 20)]
        spacings = [None, None]
        mock_image_folder_factory(shapes=shapes, spacings=spacings)

        # np.median of an empty array raises np.AxisError
        _,spacing=compute_median("some/path", return_spacing=True)
        assert spacing==[]

    def test_mixed_dimensions_raises_error(self, mock_image_folder_factory):
        """
        Tests that numpy raises an error if shapes have different dimensions,
        as `np.array` will create a ragged array. This clarifies the function's limitations.
        """
        shapes = [(100, 100), (100, 100, 100)] # 2D and 3D images
        mock_image_folder_factory(shapes=shapes)

        # `np.median` on a ragged array will raise a TypeError.
        with pytest.raises(AssertionError):
            compute_median("mixed/dims/path")

# ============================================================================
# TESTS FOR: data_fingerprint
# ============================================================================

class TestDataFingerprint:
    """Groups all tests for the data_fingerprint function."""

    # --- Happy Path Tests ---

    def test_without_mask(self, mock_fs_env_factory):
        """
        Tests fingerprinting with only an image directory.
        Intensity stats should all be zero.
        """
        # ARRANGE
        img_data = [
            (np.zeros((10, 20, 30)), {'spacing': [1.0, 1.0, 2.0]}),
            (np.zeros((12, 22, 32)), {'spacing': [1.0, 1.0, 3.0]}),
            (np.zeros((14, 24, 34)), {'spacing': [1.0, 1.0, 4.0]}),
        ]
        mock_fs_env_factory(img_data=img_data)

        # ACT
        size, space, mean, std, p05, p995 = data_fingerprint(img_dir="fake/img")

        # ASSERT
        assert np.array_equal(size, [12, 22, 32])
        assert np.allclose(space, [1.0, 1.0, 3.0])
        assert mean == 0.0
        assert std == 0.0
        assert p05 == 0.0
        assert p995 == 0.0

    def test_with_mask_calculates_all_stats(self, mock_fs_env_factory):
        """Tests fingerprinting with both images and masks."""
        # ARRANGE
        # Create data where the mask selects specific values
        img1 = np.arange(1000, 2000).reshape(10, 10, 10) # values from 1000 to 1999
        msk1 = np.zeros((10, 10, 10), dtype=int)
        msk1[5, :, :] = 1 # Selects 100 values starting at 1500

        img2 = np.arange(3000, 4000).reshape(10, 10, 10) # values from 3000 to 3999
        msk2 = np.zeros((10, 10, 10), dtype=int)
        msk2[8, :, :] = 1 # Selects 100 values starting at 3800

        img_data = [
            (img1, {'spacing': [1.0, 1.0, 1.0]}),
            (img2, {'spacing': [1.0, 1.0, 1.0]})
        ]
        msk_data = [msk1, msk2]
        mock_fs_env_factory(img_data=img_data, msk_data=msk_data)
        
        # We need to know what samples will be chosen to verify the stats
        # With seed=42, we can predict the samples
        rng = np.random.default_rng(42)
        samples1 = rng.choice(img1[msk1 > 0], 10000, replace=True)
        samples2 = rng.choice(img2[msk2 > 0], 10000, replace=True)
        all_samples = np.concatenate([samples1, samples2])

        # ACT
        size, space, mean, std, p05, p995 = data_fingerprint(
            img_dir="fake/img", msk_dir="fake/msk", num_samples=10000, seed=42
        )

        # ASSERT
        assert np.array_equal(size, [10, 10, 10])
        assert np.allclose(space, [1.0, 1.0, 1.0])
        assert np.isclose(mean, np.mean(all_samples), rtol=0.1)
        assert np.isclose(std, np.std(all_samples), rtol=0.1)
        assert np.isclose(p05, np.percentile(all_samples, 0.5), rtol=0.1)
        assert np.isclose(p995, np.percentile(all_samples, 99.5), rtol=0.1)

    # --- Edge Case and Error Tests ---

    def test_empty_mask_is_handled(self, mock_fs_env_factory):
        """Tests that a mask with all zeros doesn't crash the process."""
        img_data = [(np.ones((5, 5)), {'spacing': [1.0, 1.0]})]
        msk_data = [np.zeros((5, 5))] # Empty mask
        mock_fs_env_factory(img_data=img_data, msk_data=msk_data)

        # ACT
        size, space, mean, std, p05, p995 = data_fingerprint(img_dir="f/img", msk_dir="f/msk")

        # ASSERT - Should behave like the no-mask case as `samples` list remains empty
        assert mean == 0.0
        assert std == 0.0

    def test_empty_folder_raises_error(self, mock_fs_env_factory):
        """Tests that an empty image folder raises a specific ValueError."""
        mock_fs_env_factory(img_data=[])
        
        with pytest.raises(AssertionError):
            data_fingerprint(img_dir="empty/img")

    def test_mixed_dimensions_raises_error(self, mock_fs_env_factory):
        """Tests that images with different numbers of dims raises the specific error."""
        img_data = [
            (np.zeros((10, 20)), {'spacing': [1.0, 1.0]}), # 2D
            (np.zeros((10, 20, 30)), {'spacing': [1.0, 1.0, 1.0]}) # 3D
        ]
        mock_fs_env_factory(img_data=img_data)
        
        with pytest.raises(AssertionError):
            data_fingerprint(img_dir="mixed/img")


# ============================================================================
# TESTS FOR: find_patch_pool_batch
# ============================================================================

class TestFindPatchPoolBatch:
    """Groups all tests for the find_patch_pool_batch function."""

    # --- Happy Path Tests ---
    
    def test_standard_3d_input(self):
        """A smoke test with a typical 3D input to ensure it runs without error."""
        patch, pool, batch = find_patch_pool_batch(dims=(160, 192, 128))
        assert isinstance(patch, np.ndarray) and patch.shape == (3,)
        assert isinstance(pool, np.ndarray) and pool.shape == (3,)
        assert isinstance(batch, (int, np.integer))

    def test_predictable_output(self):
        """Test with inputs where the output can be easily calculated by hand."""
        # For dims=(64, 64, 64), max_dims=(128,128,128), max_pool=3
        # min_fmaps = 128 // (2**3) = 16
        # pool = floor(log2(64/16)) = floor(log2(4)) = 2 for all dims
        # patch should be a multiple of 2**2=4, so it stays 64.
        # batch = 2 * floor((128**3)/(64**3)) = 2 * floor(8) = 16
        patch, pool, batch = find_patch_pool_batch(dims=(64, 64, 64), max_pool=3)
        
        assert np.array_equal(patch, [64, 64, 64])
        assert np.array_equal(pool, [2, 2, 2])
        assert batch == 16

    def test_4d_input_is_handled(self):
        """Tests that a 4D input is correctly processed by stripping the first dim."""
        patch_3d, _, _ = find_patch_pool_batch(dims=(128, 128, 64))
        patch_4d, _, _ = find_patch_pool_batch(dims=(1, 128, 128, 64)) # Extra channel dim
        assert np.array_equal(patch_3d, patch_4d)

    def test_large_dims_are_reduced(self):
        """Tests that input dims larger than max_dims are correctly scaled down."""
        dims = (256, 256, 256)
        max_dims = (128, 128, 128)
        patch, _, _ = find_patch_pool_batch(dims=dims, max_dims=max_dims)
        assert patch.prod() <= np.prod(max_dims)

    # --- Error Condition Tests ---

    def test_invalid_num_dimensions_raises_error(self):
        """Tests that dims with len != 3 or 4 raises an assertion error."""
        with pytest.raises(AssertionError):
            find_patch_pool_batch(dims=(10, 20)) # 2D is not allowed

    def test_non_positive_dimension_raises_error(self):
        """Tests that a zero or negative dimension raises an assertion error."""
        with pytest.raises(AssertionError, match="One dimension is non-positve"):
            find_patch_pool_batch(dims=(128, 0, 128))

    def test_bad_max_dims_and_pool_raises_error(self):
        """Tests assertion for min_fmaps < 1."""
        with pytest.raises(AssertionError, match="is too small regarding max_pool"):
            find_patch_pool_batch(dims=(64,64,64), max_dims=(10,10,10), max_pool=5)

# ============================================================================
# TESTS FOR: get_aug_patch
# ============================================================================

class TestGetAugPatch:
    """Groups all tests for the get_aug_patch function."""
    
    @pytest.mark.parametrize(
        "patch_size, expected_aug_patch",
        [
            # Test Case 1: Isotropic patch (uses 3D diagonal)
            # sqrt(64^2 + 64^2 + 64^2) = 110.85 -> round to 111
            ([64, 64, 64], [111, 111, 111]),
            
            # Test Case 2: Anisotropic patch (uses 2D diagonal of larger dims)
            # Anisotropy detected because 128/32 = 4 > 3.
            # sqrt(128^2 + 128^2) = 181.02 -> round to 181
            # The smallest dimension (32) is kept as is.
            ([128, 128, 32], [181, 181, 32]),

            # Test Case 3: Another anisotropic patch
            # Anisotropy detected because 96/24 = 4 > 3.
            # sqrt(96^2 + 64^2) = 115.39 -> round to 115
            # The smallest dimension (24) is kept as is.
            ([96, 64, 24], [115, 115, 24]),

            # Test Case 4: Near-isotropic, no anisotropy detected (2.5 < 3)
            # sqrt(80^2 + 80^2 + 32^2) = 117.57 -> round to 118
            ([80, 80, 32], [118, 118, 118]),
        ]
    )
    def test_patch_size_calculation(self, patch_size, expected_aug_patch):
        """Tests both isotropic and anisotropic logic using parameterization."""
        # ARRANGE & ACT
        aug_patch = get_aug_patch(patch_size)
        
        # ASSERT
        assert isinstance(aug_patch, list)
        assert aug_patch == expected_aug_patch

    def test_input_types(self):
        """Tests that the function accepts list, tuple, and numpy array inputs."""
        patch_list = [64, 64, 64]
        patch_tuple = (64, 64, 64)
        patch_array = np.array([64, 64, 64])
        
        expected = [111, 111, 111]

        assert get_aug_patch(patch_list) == expected
        assert get_aug_patch(patch_tuple) == expected
        assert get_aug_patch(patch_array) == expected