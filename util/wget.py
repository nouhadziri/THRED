"""
Taken from https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py
"""

from tqdm import tqdm
from urllib.request import urlretrieve


class TqdmUpTo(tqdm):
    """Alternative Class-based version of the above.
    Provides `update_to(n)` which uses `tqdm.update(delta_n)`.
    Inspired by [twine#242](https://github.com/pypa/twine/pull/242),
    [here](https://github.com/pypa/twine/commit/42e55e06).
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def download(download_link, local_file=None):
    download_file = download_link.replace('/', ' ').split()[-1]

    if local_file is None:
        local_file = download_file

    with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                  desc=download_file) as t:  # all optional kwargs
        urlretrieve(download_link, filename=local_file, reporthook=t.update_to,
                    data=None)
