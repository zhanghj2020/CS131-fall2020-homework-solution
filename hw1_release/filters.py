"""
CS131 - Computer Vision: Foundations and Applications
Assignment 1
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/16/2017
Python Version: 3.5+
"""

import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    for i in range(Hi):
        for j in range(Wi):
            conv_sum = 0
            for x in range(-(Hk // 2), Hk // 2 + 1):
                for y in range(-(Wk // 2), Wk // 2 + 1):
                    if i + x < 0 or i + x >= Hi:
                        continue
                    if j + y < 0 or j + y >= Wi:
                        continue
                    #print(-Hk // 2, y, Hk - x - Hk // 2 -1, Wk - y - Wk // 2 -1)
                    conv_sum += image[i + x][j + y] * kernel[Hk - x - Hk // 2 - 1][Wk - y - Wk // 2 - 1]
            out[i][j] = conv_sum
                    
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    out = np.zeros((H + 2 * pad_height, W + 2 * pad_width))
    out[pad_height : pad_height + H, pad_width : pad_width + W] = image[:, :]
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    kernel = np.flip(np.flip(kernel, axis=0), axis=1)
    ### YOUR CODE HERE
    pad_image = zero_pad(image, Hk // 2, Wk // 2)
    for i in range(Hi):
        for j in range(Wi):
            out[i][j] = np.sum(pad_image[i:i+Hk, j:j+Wk] * kernel)

    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of image f and template g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    g = np.flip(np.flip(g, axis=1), axis=0)
    out = conv_fast(f, g)
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of image f and template g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    g = g - np.mean(g)
    out = cross_correlation(f, g)
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of image f and template g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    Hi, Wi = f.shape
    Hk, Wk = g.shape
    out = np.zeros((Hi, Wi))
    pad_f = zero_pad(f, Hk // 2, Wk // 2)
    g_mean = np.mean(g)
    g_std = np.std(g)

    for i in range(Hi):
        for j in range(Wi):
            f_patch_mean = np.mean(pad_f[i:i+Hk, j:j+Wk])
            f_patch_std = np.std(pad_f[i:i+Hk, j:j+Wk])
            out[i][j] = np.sum(((pad_f[i:i+Hk, j:j+Wk] - f_patch_mean) / f_patch_std) * ((g - g_mean) / g_std))

    ### END YOUR CODE

    return out
