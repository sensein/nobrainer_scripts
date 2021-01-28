import math
from pathlib import Path
import pandas as pd
import nibabel as nib
import numpy as np
import tensorflow as tf

from nibabel.processing import conform , resample_from_to
from nobrainer.volume import from_blocks_numpy
from nobrainer.volume import standardize_numpy
from nobrainer.volume import to_blocks_numpy
from utils import StreamingStats


def predict_from_array(
    inputs,
    model,
    block_shape,
    batch_size=1,
    normalizer=None,
    n_samples=1,
    return_variance=False,
    return_entropy=False,
):
    """Return a prediction given a filepath and an ndarray of features.
    Parameters
    ----------
    inputs: ndarray, array of features.
    model: `tf.keras.Model`, trained model.
    block_shape: tuple of length 3, shape of sub-volumes on which to
        predict.
    batch_size: int, number of sub-volumes per batch for predictions.
    normalizer: callable, function that accepts an ndarray and returns an
        ndarray. Called before separating volume into blocks.
    n_samples: The number of sampling. If set as 1, it will just return the
        single prediction value. The default value is 1
    return_variance: Boolean. If set True, it returns the running population
        variance along with mean. Note, if the n_samples is smaller or equal to 1,
        the variance will not be returned; instead it will return None
    return_entropy: Boolean. If set True, it returns the running entropy.
        along with mean.
    Returns
    -------
    ndarray of predictions.
    """
    if normalizer:
        features = normalizer(inputs)
    else:
        features = inputs
    if block_shape is not None:
        features = to_blocks_numpy(features, block_shape=block_shape)
    else:
        features = features[None]  # Add batch dimension.

    # Add a dimension for single channel.
    features = features[..., None]

    # Predict per block to reduce memory consumption.
    n_blocks = features.shape[0]
    n_batches = math.ceil(n_blocks / batch_size)

    if not return_variance and not return_entropy and n_samples == 1:
        outputs = model.predict(features, batch_size=1, verbose=0)
        if outputs.shape[-1] == 1:
            # Binarize according to threshold.
            outputs = outputs > 0.3
            outputs = outputs.squeeze(-1)
            # Nibabel doesn't like saving boolean arrays as Nifti.
            outputs = outputs.astype(np.uint8)
        else:
            # Hard classes for multi-class segmentation.
            outputs = np.argmax(outputs, -1)
        outputs = from_blocks_numpy(outputs, output_shape=inputs.shape)
        return outputs    
    else:       
    #raise NotImplementedError("Predicting from Bayesian nets is not implemented yet.")
        # Variational inference
        means = np.zeros_like(features.squeeze(-1))
        variances = np.zeros_like(features.squeeze(-1))
        entropies = np.zeros_like(features.squeeze(-1))
        progbar = tf.keras.utils.Progbar(n_batches)
        progbar.update(0)
        for j in range(0, n_blocks, batch_size):

            this_x = features[j : j + batch_size]
            s = StreamingStats()
            for n in range(n_samples):
                new_prediction = model.predict(this_x, batch_size=1, verbose=1)
                s.update(new_prediction)

            means[j : j + batch_size] = np.argmax(s.mean(),axis=-1)  # max mean
            variances[j : j + batch_size] = np.sum(s.var(), axis = -1)
            entropies[j : j + batch_size] = np.sum(s.entropy(), axis = -1) # entropy
            progbar.add(1)

        total_means = from_blocks_numpy(means, output_shape=inputs.shape)
        total_variance = from_blocks_numpy(variances, output_shape=inputs.shape)
        total_entropy = from_blocks_numpy(entropies, output_shape=inputs.shape)

    include_variance = (n_samples > 1) and (return_variance)
    if include_variance:
        if return_entropy:
            return total_means, total_variance, total_entropy
        else:
            return total_means, total_variance
    else:
        if return_entropy:
            return total_means, total_entropy
        else:
            return (total_means,)


def _get_model(path):
    """Return `tf.keras.Model` object from a filepath.
    Parameters
    ----------
    path: str, path to HDF5 or SavedModel file.
    Returns
    -------
    Instance of `tf.keras.Model`.
    Raises
    ------
    `ValueError` if cannot load model.
    """
    if isinstance(path, tf.keras.Model):
        return path
    try:
        return tf.keras.models.load_model(path, compile=False)
    except OSError:
        # Not an HDF5 file.
        pass

    try:
        path = Path(path)
        if path.suffix == ".json":
            path = path.parent.parent
        return tf.keras.experimental.load_from_saved_model(str(path))
    except Exception:
        pass

    raise ValueError(
        "Failed to load model. Is the model in HDF5 format or SavedModel" " format?"
    )

def _reslice(input, reference):
    """Reslice volume using nibabel."""
    return nib.processing.resample_from_to(input, reference)

def get_reverse_dict(n_classes):
    print('Mapping back from segmentation classes 0f 0-{} into freesurfer labels'.format(n_classes-1, n_classes))
    if n_classes == 50: 
        tmp = pd.read_csv('50-class-mapping.csv', header=0,usecols=[0,1],dtype=np.int32)
        mydict = dict(tuple(zip(tmp['new'],tmp['original'])))
        return mydict
    elif n_classes == 115:
        tmp = pd.read_csv('115-class-mapping.csv', header=0,usecols=[0,1],dtype=np.int32)
        mydict = dict(tuple(zip(tmp['new'],tmp['original'])))
        del tmp
        return mydict
    else: raise(NotImplementedError)

def replace_in_numpy(x, mapping, zero=True):
    """Replace values in numpy ndarray `x` using dictionary `mapping`.

    """
    # Extract out keys and values
    k = np.array(list(mapping.keys()))
    v = np.array(list(mapping.values()))

    # Get argsort indices
    sidx = k.argsort()

    ks = k[sidx]
    vs = v[sidx]
    idx = np.searchsorted(ks,x)

    if not zero:
        idx[idx==len(vs)] = 0
        mask = ks[idx] == x
        return np.where(mask, vs[idx], x)
    else:
        return vs[idx]


if __name__ == "__main__":

    
    required_shape = (256, 256, 256)
    block_shape = (128, 128, 128)
    n_classes = 115
    n_samples = 1


    model_path = "training_files/kwyk_4gpu_21-01-03_01-18/saved_model"
    data = "data/pac_0_orig.nii.gz"

    outfile_ext = '.nii.gz'
    outfile_stem = "kwyk_output"

    # Load the model
    model = _get_model(model_path)

    # Load the input file
    _orig_infile = nib.load(data)
    img = _orig_infile
    ndim = len(img.shape)
    if ndim != 3:
        raise ValueError("Input volume must have three dimensions but got {}.".format(ndim))

    # check data dimension and conform    
    if img.shape != required_shape:
        print("++ Conforming volume to 1mm^3 voxels and size 256x256x256.")
        img = conform(_orig_infile, out_shape= required_shape)

    inputs = np.asarray(img.dataobj)
    img.uncache()
    inputs = inputs.astype(np.float32)

    # forward pass of the model
    outputs = predict_from_array(inputs, model, block_shape, batch_size=1, normalizer= standardize_numpy, n_samples= n_samples, 
        return_variance=True, return_entropy=True)

    # replace the outputs with freesurfer labels 
    outputs = replace_in_numpy(outputs, get_reverse_dict(n_classes))
    

    # variational or simple inference            
    if n_samples > 1:
        # change the numpy array to a nibabel image object with affine and header 
        outputs = nib.spatialimages.SpatialImage(
                dataobj=outputs[0], affine=img.affine, header=img.header, extra=img.extra),\
                nib.spatialimages.SpatialImage(
                dataobj=outputs[1], affine=img.affine, header=img.header, extra=img.extra),\
                nib.spatialimages.SpatialImage(
                dataobj=outputs[2], affine=img.affine, header=img.header, extra=img.extra)

        means, variance, entropy = outputs
    else:
        # change the numpy array to a nibabel image object with affine and header 
        outputs = nib.spatialimages.SpatialImage(
                dataobj=outputs[0], affine=img.affine, header=img.header, extra=img.extra),\
                nib.spatialimages.SpatialImage(
                dataobj=outputs[1], affine=img.affine, header=img.header, extra=img.extra)
                
        means, entropy = outputs
        variance = None


    outfile_means_orig = "{}_means_orig{}".format(outfile_stem, outfile_ext)
    outfile_variance_orig = "{}_variance_orig{}".format(outfile_stem, outfile_ext)
    outfile_entropy_orig = "{}_entropy_orig{}".format(outfile_stem, outfile_ext)

    outfile_means = "{}_means{}".format(outfile_stem, outfile_ext)
    outfile_variance = "{}_variance{}".format(outfile_stem, outfile_ext)
    outfile_entropy = "{}_entropy{}".format(outfile_stem, outfile_ext)

    print("++ Saving results.")
    data = np.round(means.get_fdata()).astype(np.uint8)
    means = nib.Nifti1Image(data, header=means.header, affine=means.affine)
    means.header.set_data_dtype(np.uint8)

    # Save output
    if n_samples > 1:

        nib.save(means, outfile_means)
        _means_orig = _reslice(means,_orig_infile)
        nib.save(_means_orig, outfile_means_orig)
        # Save variance
        nib.save(variance, outfile_variance)
        _var_orig = _reslice(variance,_orig_infile)
        nib.save(_var_orig, outfile_variance_orig)
        # Save entropy
        nib.save(entropy, outfile_entropy)
        _entropy_orig = _reslice(entropy,_orig_infile)
        nib.save(_entropy_orig, outfile_entropy_orig)
    else:
        nib.save(means, outfile_means)
        _means_orig = _reslice(means,_orig_infile)
        nib.save(_means_orig, outfile_means_orig)
        # Save entropy
        nib.save(entropy, outfile_entropy)
        _entropy_orig = _reslice(entropy,_orig_infile)
        nib.save(_entropy_orig, outfile_entropy_orig)



    



