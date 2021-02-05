import sys
sys.path.append('.')
import nobrainer
import csv

def _read_csv(filepath, skip_header=True, delimiter=","):
    """Return list of tuples from a CSV, where each tuple contains the items
    in a row.
    """
    with open(filepath, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        if skip_header:
            next(reader)
        return [tuple(row) for row in reader]

volume_filepaths = _read_csv('single_volume.csv')

nobrainer.tfrecord.write(
    features_labels=volume_filepaths,
    filename_template='/data/single_volume-{shard:03d}.tfrec',
    examples_per_shard=1
)

