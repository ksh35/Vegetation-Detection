import sys
import unittest

import numpy as np
import torch

sys.path.append(".")
from src.esd_data.augmentations import (
    AddNoise,
    Blur,
    RandomHFlip,
    RandomVFlip,
    ToTensor,
)


class TestAugmentationsHW02(unittest.TestCase):
    """Class to test the augmentations module."""

    def setUp(self):
        self.sample = {
            "X": np.random.rand(4, 3, 200, 200),
            "y": np.random.randint(0, 3, (4, 4)),
        }
        self.blur = Blur(20)
        self.noise = AddNoise(mean=0, std_lim=0.5)
        self.hflip = RandomHFlip(p=1.0)
        self.vflip = RandomVFlip(p=1.0)
        self.to_tensor = ToTensor()

    def tearDown(self):
        pass

    def test_add_noise(self):
        """Test the add noise augmentation."""
        # Apply the noise augmentation to the sample
        sample_noised = self.noise(self.sample)

        # Check that the augmented image is still a numpy array
        self.assertIsInstance(
            sample_noised["X"], np.ndarray, "Image is not a numpy array."
        )

        # Check that the shape of the augmented image is the
        # same as the original
        self.assertEqual(
            sample_noised["X"].shape,
            self.sample["X"].shape,
            "Shape of the image has changed.",
        )

        # Calculate the standard deviation of the pixel values
        # in the original and augmented images
        std_original = np.std(self.sample["X"])
        std_noised = np.std(sample_noised["X"])

        # Assert that the standard deviation of the augmented image
        # is greater than the original
        self.assertGreater(
            std_noised,
            std_original,
            "The add noise augmentation did not add noise to the image.",
        )

    def test_blur(self):
        """Test the blur augmentation."""
        # Apply the blur augmentation to the sample
        sample_blurred = self.blur(self.sample)

        # Check that thhe augmented image is still a numpy array
        self.assertIsInstance(
            sample_blurred["X"], np.ndarray, "Image is not a numpy array."
        )

        # Check that the shape of the augmented image is the
        # same as the original
        self.assertEqual(
            sample_blurred["X"].shape,
            self.sample["X"].shape,
            "Shape of the image has changed.",
        )

        # Calculate the sum of the absolute difference between
        # the original and augmented images
        sum_ab_diff = np.sum(np.abs(sample_blurred["X"] - self.sample["X"]))

        # Assert that the sum of absolute differences is greater
        # than zero, indicating blur has been applied
        self.assertGreater(
            sum_ab_diff, 0, "The image is unchanged after the Blur operation."
        )

    def test_random_hflip(self):
        """Test the random horizontal flip augmentation."""
        # Apply the horizontal flip augmentation to the sample
        sample_hflipped = self.hflip(self.sample)

        # Check that the augmented image is still a numpy array
        self.assertIsInstance(
            sample_hflipped["X"], np.ndarray, "Image is not a numpy array."
        )

        # Check that the shape of the augmented image is the same
        # as the original
        self.assertEqual(
            sample_hflipped["X"].shape,
            self.sample["X"].shape,
            "Shape of the image has changed.",
        )

        # Compare the first column of the original image with the last
        # column of the flipped image

        original_first_column_x = self.sample["X"][:, :, 0]
        flipped_last_column_x = sample_hflipped["X"][:, :, -1]

        original_first_column_y = self.sample["y"][:, 0]
        flipped_last_column_y = sample_hflipped["y"][:, -1]

        # Assert that the first column of the original matches the last
        # column of the flipped image
        np.testing.assert_allclose(
            original_first_column_x,
            flipped_last_column_x,
            err_msg="The horizontal flip operation did not\
                work on input image.",
        )
        np.testing.assert_allclose(
            original_first_column_y,
            flipped_last_column_y,
            err_msg="The horizontal flip operation did not\
                work on mask image.",
        )

    def test_random_vflip(self):
        """Test the random vertical flip augmentation."""
        # Apply the vertical flip augmentation to the sample
        sample_vflipped = self.vflip(self.sample)

        # Check that the augmented image is still a numpy array
        self.assertIsInstance(
            sample_vflipped["X"], np.ndarray, "Image is not a numpy array."
        )

        # Check that the shape of the augmented image is the same
        # as the original
        self.assertEqual(
            sample_vflipped["X"].shape,
            self.sample["X"].shape,
            "Shape of the image has changed.",
        )

        # Compare the first row of the original image with the last
        # row of the flipped image
        original_first_row_x = self.sample["X"][:, 0, :]
        flipped_last_row_x = sample_vflipped["X"][:, -1, :]
        original_first_row_y = self.sample["y"][0, :]
        flipped_last_row_y = sample_vflipped["y"][-1, :]

        # Assert that the first row of the original matches the last
        # row of the flipped image
        np.testing.assert_allclose(
            original_first_row_x,
            flipped_last_row_x,
            err_msg="The vertical flip operation did\
            not work on input image.",
        )
        np.testing.assert_allclose(
            original_first_row_y,
            flipped_last_row_y,
            err_msg="The vertical flip operation did\
            not work on mask image.",
        )

    def test_to_tensor(self):
        """Test the to tensor augmentation."""
        # Apply the to tensor augmentation to the sample
        sample_tensor = self.to_tensor(self.sample)

        # Check that the augmented image is a torch tensor
        self.assertIsInstance(
            sample_tensor["X"], torch.Tensor, "Image is not a torch Tensor."
        )


if __name__ == "__main__":
    unittest.main()
