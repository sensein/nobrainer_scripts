from collections import namedtuple
import csv
import os
import tempfile

import numpy as np
import tensorflow as tf

_cache_dir = os.path.join(tempfile.gettempdir(), "nobrainer-data")


class StreamingStats:
    """Object to calculate statistics on streaming data.
    Compatible with scalars and n-dimensional arrays.
    Examples
    --------
    ```python
    >>> s = StreamingStats()
    >>> s.update(10).update(20)
    >>> s.mean()
    15.0
    ```
    ```python
    >>> import numpy as np
    >>> a = np.array([[0, 2], [4, 8]])
    >>> b = np.array([[2, 4], [8, 16]])
    >>> s = StreamingStats()
    >>> s.update(a).update(b)
    >>> s.mean()
    array([[ 1.,  3.],
       [ 6., 12.]])
    ```
    """

    def __init__(self):
        self._n_samples = 0
        self._current_mean = 0.0
        self._M = 0.0

    def update(self, value):
        """Update the statistics with the next value.
        Parameters
        ----------
        value: scalar, array-like
        Returns
        -------
        Modified instance.
        """
        if self._n_samples == 0:
            self._current_mean = value
        else:
            prev_mean = self._current_mean
            curr_mean = prev_mean + (value - prev_mean) / (self._n_samples + 1)
            _M = self._M + (prev_mean - value) * (curr_mean - value)
            # Set the instance attributes after computation in case there are
            # errors during computation.
            self._current_mean = curr_mean
            self._M = _M
        self._n_samples += 1
        return self

    def mean(self):
        """Return current mean of streaming data."""
        return self._current_mean

    def var(self):
        """Return current variance of streaming data."""
        return self._M / self._n_samples

    def std(self):
        """Return current standard deviation of streaming data."""
        return self.var() ** 0.5

    def entropy(self):
        """Return current entropy of streaming data."""
        eps = 1e-07
        mult = np.multiply(np.log(self.mean() + eps), self.mean())
        return -mult
        # return -np.sum(mult, axis=axis)
