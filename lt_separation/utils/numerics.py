import numpy as np
from typing import Union, Sequence


def get_chi2(model, data, error):
    return np.sum(
        np.divide(
            (model - data) ** 2, error**2, where=error > 0, out=np.zeros_like(model)
        )
    )


def sum_over(arr: np.ndarray, axis: Union[int, Sequence[int]] = 0):
    """
    Sum the values and errors in an array of shape (..., 2) where the last dimension contains [value, error].
    Parameters
    ----------
    arr: numpy array of shape (..., 2) where the last dimension contains [value, error]
    axis: axis along which to sum
    Returns
    -------
    numpy array of shape (..., 2) where the last dimension contains [sum, error]
    """
    value = arr[..., 0]
    error = arr[..., 1]
    ans = np.sum(value, axis=axis)
    err = np.sqrt(np.sum(error**2, axis=axis))
    return np.stack([ans, err], axis=-1)


def weighted_mean_with_std(arr: np.ndarray, axis: int = -1):
    """
    Compute the weighted average and its standard deviation for an array of shape (..., 2) where the last dimension contains [value, weight]. If the sum of weights is zero, the average is returned as 0.0.
    Parameters
    ----------
    arr: numpy array of shape (..., 2) where the last dimension contains [value, weight]
    axis: axis along which to compute the weighted average

    Returns
    -------
    numpy array of shape (..., 2) where the last dimension contains [weighted average, its standard deviation]
    """
    values = arr[..., 0]
    weights = arr[..., 1]
    wsum = np.sum(weights, axis=axis, keepdims=True)

    mean = np.divide(
        np.sum(values * weights, axis=axis, keepdims=True),
        wsum,
        out=np.zeros_like(wsum, dtype=values.dtype),
        where=wsum != 0,
    )

    var = np.sum(weights * (values - mean) ** 2, axis=axis, keepdims=True)
    var = np.divide(
        var, wsum, out=np.zeros_like(wsum, dtype=values.dtype), where=wsum != 0
    )
    mean = np.squeeze(mean, axis=axis)
    std = np.sqrt(np.squeeze(var, axis=axis))

    return np.stack([mean, std], axis=-1)


def weighted_mean_with_error(arr: np.ndarray, axis: int = 0):
    """
    Compute the weighted average and its error for an array of shape (..., 2) where the last dimension contains [value, error]. Only accept ararys with sum 1/err^2 > 0.
    Parameters
    ----------
    arr: numpy array of shape (..., 2) where the last dimension contains [value, error]
    axis: axis along which to compute the weighted average
    Returns
    -------
    numpy array of shape (..., 2) where the last dimension contains [weighted average, its error]
    """
    arr = np.asarray(arr, dtype=float)
    values = arr[..., 0]
    errors = arr[..., 1]

    weights = np.divide(1.0, errors**2, out=np.zeros_like(errors), where=errors != 0)

    wsum = np.sum(weights, axis=axis)
    if np.any(wsum < 0):
        raise ValueError("Sum of weights is negative, cannot compute weighted average.")

    avg = np.divide(
        np.sum(values * weights, axis=axis),
        wsum,
        out=np.zeros_like(wsum, dtype=float),
        where=wsum != 0,
    )

    avg_err = np.sqrt(
        np.divide(1.0, wsum, out=np.zeros_like(wsum, dtype=float), where=wsum != 0)
    )

    mask_all_zero = np.all(errors == 0, axis=axis)
    if np.any(mask_all_zero):
        # broadcast to match avg shape
        avg = np.where(mask_all_zero, 0.0, avg)
        avg_err = np.where(mask_all_zero, 0.0, avg_err)

    return np.stack([avg, avg_err], axis=-1)


def sum_with_error(arr1: np.ndarray, arr2: np.ndarray):
    """
    Sum two values with errors.
    arr[...,0] = value, arr[...,1] = absolute error.
    """
    summ = arr1[..., 0] + arr2[..., 0]
    summ_err = np.sqrt(arr1[..., 1] ** 2 + arr2[..., 1] ** 2)
    return np.stack([summ, summ_err], axis=-1)


def subtract_with_error(arr1: np.ndarray, arr2: np.ndarray):
    """
    Subtract arr2 from arr1, propagating errors.
    arr[...,0] = value, arr[...,1] = absolute error.
    """
    return sum_with_error(arr1, np.stack([-arr2[..., 0], arr2[..., 1]], axis=-1))


def multiply_with_error(arr1: np.ndarray, arr2: np.ndarray):
    """
    Multiply two values with errors.
    arr[...,0] = value, arr[...,1] = absolute error.
    """
    prod = arr1[..., 0] * arr2[..., 0]
    prod_err = np.sqrt(
        (arr2[..., 0] * arr1[..., 1]) ** 2 + (arr1[..., 0] * arr2[..., 1]) ** 2
    )
    return np.stack([prod, prod_err], axis=-1)


def take_ratio_with_error(arr1: np.ndarray, arr2: np.ndarray):
    val1, err1 = arr1[..., 0], arr1[..., 1]
    val2, err2 = arr2[..., 0], arr2[..., 1]

    rat = np.divide(
        val1, val2, out=np.full_like(val1, np.nan, dtype=float), where=val2 != 0
    )

    # Propagate error:
    # σ_R = sqrt( (σ_A/B)^2 + (A σ_B / B^2)^2 )
    rat_err = np.sqrt(
        np.divide(
            (err1 * val2) ** 2 + (val1 * err2) ** 2,
            val2**4,
            out=np.full_like(val1, np.nan, dtype=float),
            where=val2 != 0,
        )
    )

    return np.stack([rat, rat_err], axis=-1)


def is_valid(arr: np.ndarray) -> bool:
    """
    Check if an array is valid:
    - No NaN or Inf
    - If shaped (..., 2), requires second component (error) >= 0
    - If shaped (2,), also checks arr[1] >= 0
    """
    if not np.all(np.isfinite(arr)):
        return False

    # Handle 1D case like [value, error]
    if arr.ndim == 1 and arr.shape[0] == 2:
        return arr[1] >= 0

    # Handle structured arrays with [..., 2]
    if arr.shape[-1] == 2:
        return np.all(arr[..., 1] >= 0)

    # If no error column, just data validity check
    return True
