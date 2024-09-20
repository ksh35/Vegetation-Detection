"""Legacy augmentations.py from hw03 winter 2024"""

import cv2
import numpy as np
import torch
import random
from typing import Dict


def apply_per_band(img, transform):
    result = np.zeros_like(img)
    for band in range(img.shape[0]):
        transformed_band = transform(img[band].copy())
        result[band] = transformed_band

    return result


class Blur(object):
    """
    Blurs each band separately using cv2.blur

    Parameters:
        kernel: Size of the blurring kernel
        in both x and y dimensions, used
        as the input of cv.blur

    This operation is only done to the X.
    """

    def __init__(self, kernel=3):
        self.kernel = kernel

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Performs the blur transformation.

        Input:
            sample: Dict[str, np.ndarray]
                Has two keys, 'X' and 'y'.
                Each of them has shape (time*band, width, height)

        Output:
            transformed: Dict[str, np.ndarray]
                Has two keys, 'X' and 'y'.
                Each of them has shape (time*band, width, height)
        """
        # sample must have X and y in a dictionary format
        img, mask = sample["X"], sample["y"]
        kernel = random.randint(1, self.kernel)
        # --- start here ---

        # apply per band the cv2.blur function (you can pass it as a lambda)
        X = apply_per_band(img, lambda x: cv2.blur(x, (kernel, kernel)))
        # return the {X : img, y : mask}
        return {"X": X, "y": mask}


class AddNoise(object):
    """
    Adds random gaussian noise using np.random.normal.

    Parameters:
        mean: float
            Mean of the gaussian noise
        std_lim: float
            Maximum value of the standard deviation
    """

    def __init__(self, mean=0, std_lim=0.0):
        self.mean = mean
        self.std_lim = std_lim

    def __call__(self, sample):
        """
        Performs the add noise transformation.
        A random standard deviation is first calculated using
        random.uniform to be between 0 and self.std_lim

        Random noise is then added to each pixel with
        mean self.mean and the standard deviation
        that was just calculated

        The resulting value is then clipped to be between
        0 and 1.

        This operation is only done to the X.

        Input:
            sample: Dict[str, np.ndarray]
                Has two keys, 'X' and 'y'.
                Each of them has shape (time*band, width, height)

        Output:
            transformed: Dict[str, np.ndarray]
                Has two keys, 'X' and 'y'.
                Each of them has shape (time*band, width, height)
        """
        # sample must have X and y in a dictionary format
        img, mask = sample["X"], sample["y"]

        # --- start here ---

        # calculate the random standard deviation between 0 and the standard limit
        random_std = random.uniform(0, self.std_lim)
        # generate the noise using a random normal distribution from the mean, random standard deviation, and in the shape of the img
        noise = np.random.normal(self.mean, random_std, img.shape)
        # add the noise to the img and clip between 0 and 1
        img = img + noise
        img = np.clip(img, 0, 1)

        # return the {X : img, y : mask}
        return {"X": img, "y": mask}


class RandomVFlip(object):
    """
    Randomly flips all bands vertically in an image with probability p.

    Parameters:
        p: probability of flipping image.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Performs the random flip transformation using cv.flip.

        Input:
            sample: Dict[str, np.ndarray]
                Has two keys, 'X' and 'y'.
                Each of them has shape (time*band, width, height)

        Output:
            transformed: Dict[str, np.ndarray]
                Has two keys, 'X' and 'y'.
                Each of them has shape (time*band, width, height)
        """
        # sample must have X and y in a dictionary format
        img, mask = sample["X"], sample["y"]

        # --- start here ---
        # some sort of if statement that uses self.p to determine whether or not to act,
        # probably using random.random() (there are so many ways to do this, we just want it to
        # act only self.p% of the time)
        if(random.random() < self.p):
            # apply per band the cv2.flip function (you can pass it as a lambda)
            img = apply_per_band(img, lambda x: cv2.flip(x, 0))
            # also flip the mask (not per band, just directly on the mask)
            mask = cv2.flip(mask, 0)

        # return the {X : img, y : mask}
        return {"X": img, "y": mask}


class RandomHFlip(object):
    """
    Randomly flips all bands horizontally in an image with probability p.

    Parameters:
        p: probability of flipping image.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Performs the random flip transformation using cv.flip.

        Input:
            sample: Dict[str, np.ndarray]
                Has two keys, 'X' and 'y'.
                Each of them has shape (time*band, width, height)

        Output:
            transformed: Dict[str, np.ndarray]
                Has two keys, 'X' and 'y'.
                Each of them has shape (time*band, width, height)
        """
        # sample must have X and y in a dictionary format
        img, mask = sample["X"], sample["y"]

        # --- start here ---
        # some sort of if statement that uses self.p to determine whether or not to act,
        # probably using random.random() (there are so many ways to do this, we just want it to
        # act only self.p% of the time)
        if(random.random() < self.p):
            # apply per band the cv2.flip function (you can pass it as a lambda)
            img = apply_per_band(img, lambda x: cv2.flip(x, 1))
            # also flip the mask (not per band, just directly on the mask)
            mask = cv2.flip(mask, 1)

        # return the {X : img, y : mask}
        return {"X": img, "y": mask}


class ToTensor(object):
    """
    Converts numpy.array to torch.tensor
    """

    def __call__(self, sample):
        """
        Transforms all numpy arrays to tensors

        Input:
            sample: Dict[str, np.ndarray]
                Has two keys, 'X' and 'y'.
                Each of them has shape (time*band, width, height)

        Output:
            transformed: Dict[str, torch.Tensor]
                Has two keys, 'X' and 'y'.
                Each of them has shape (time*band, width, height)
        """
        img, mask = sample["X"], sample["y"]

        return {"X": torch.from_numpy(img), "y": torch.from_numpy(mask)}
