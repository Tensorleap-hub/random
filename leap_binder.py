from typing import Dict
from code_loader.inner_leap_binder.leapbinder_decorators import *
from numpy.typing import NDArray
import tensorflow as tf

@tensorleap_preprocess()
def preprocess_func_leap() -> List[PreprocessResponse]:

    # Generate a PreprocessResponse for each data slice, to later be read by the encoders.
    # The length of each data slice is provided, along with the data dictionary.
    # In this example we pass `images` and `labels` that later are encoded into the inputs and outputs
    train = PreprocessResponse(length=100, data={})
    val = PreprocessResponse(length=100, data={})
    response = [train, val]
    return response


# Input encoder fetches the image with the index `idx` from the `images` array set in
# the PreprocessResponse data. Returns a numpy array containing the sample's image.
@tensorleap_input_encoder('image', channel_dim=1)
def input_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    return np.random.random((3, 512, 512)).astype(np.float32)
    #return preprocess.data['images'][idx].astype('float32')

@tensorleap_input_encoder('input_2', channel_dim=1)
def input_encoder_2(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    return np.random.random((13)).astype(np.float32)
    #return preprocess.data['images'][idx].astype('float32')

@tensorleap_custom_visualizer('rotated_vis', LeapDataType.Image)
def image_visualizer(image: npt.NDArray[np.float32]) -> LeapImage:
    image = image.transpose([1, 2, 0])
    return LeapImage((image).astype(np.uint8))

@tensorleap_gt_encoder('classes')
def gt_encoder(idx: int, preprocessing: PreprocessResponse) -> np.ndarray:
    return np.random.random((2)).astype(np.float32)

# Ground truth encoder fetches the label with the index `idx` from the `labels` array set in
# the PreprocessResponse's data. Returns a numpy array containing a hot vector label correlated with the sample.
@tensorleap_custom_loss('dummy')
def dummy_loss(x, y, z, gt):
    return tf.zeros_like(x.shape[0])

# Metadata functions allow to add extra data for a later use in analysis.
# This metadata adds the int digit of each sample (not a hot vector).
@tensorleap_metadata('metadata_sample_index')
def metadata_sample_index(idx: int, preprocess: PreprocessResponse) -> int:
    return idx

# Adding a name to the prediction, and supplying it with label names.
leap_binder.add_prediction(name='a', labels=['0'])
leap_binder.add_prediction(name='b', labels=[str(i) for i in range(4)])
leap_binder.add_prediction(name='c', labels=[str(i) for i in range(7)])



if __name__ == '__main__':
    leap_binder.check()
