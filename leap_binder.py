from typing import Dict
from code_loader.inner_leap_binder.leapbinder_decorators import *
from numpy.typing import NDArray
import numpy as np

def add_col_padding(image: np.array, np_dtype: str = 'uint8') -> np.array:
    padded_columns = np.zeros(shape=(image.shape[0], 5, image.shape[2])).astype(np_dtype)
    image_with_padding = np.concatenate((image, padded_columns), axis=1)
    return image_with_padding

def add_row_padding(image: np.array, np_dtype: str = 'uint8') -> np.array:
    padded_rows = np.zeros(shape=(5, image.shape[1], image.shape[2])).astype(np_dtype)
    image_with_padding = np.concatenate((image, padded_rows), axis=0)
    return image_with_padding

def rescale_min_max(image: np.array, np_dtype: str = 'uint8') -> np.array:
    image_min = np.min(image)
    image_max = np.max(image)
    image_rescaled = (image - image_min) / (image_max - image_min)
    image_rescaled = (image_rescaled * 255.0).astype(np_dtype)
    return image_rescaled

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
    return np.random.random((3, 8, 256, 256)).astype(np.float32)
    #return preprocess.data['images'][idx].astype('float32')

@tensorleap_gt_encoder('gt')
def gt_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    return np.random.random((1, 400)).astype(np.float32)


@tensorleap_custom_visualizer('rotated_vis', LeapDataType.Image)
def image_visualizer(image: npt.NDArray[np.float32]) -> LeapImage:
    image = image.transpose([1, 2, 0])
    return LeapImage((image).astype(np.uint8))


# Ground truth encoder fetches the label with the index `idx` from the `labels` array set in
# the PreprocessResponse's data. Returns a numpy array containing a hot vector label correlated with the sample.
@tensorleap_custom_loss('dummy')
def dummy_loss(gt, pred):
    return np.zeros(1)

# Metadata functions allow to add extra data for a later use in analysis.
# This metadata adds the int digit of each sample (not a hot vector).
@tensorleap_metadata('metadata_sample_index')
def metadata_sample_index(idx: int, preprocess: PreprocessResponse) -> int:
    return idx


def frames_grid_heatmap(frames: NDArray[np.float32]) -> LeapImage:
    """
    Arranges a batch of heatmap frames into a 2-row, 4-column grid with separators.

    Args:
        images (np.ndarray): Input array of shape (1, 8, 256, 256, 3).

    Returns:
        np.ndarray: Concatenated array of heatmaps with separators.
    """
    print(f"$$$ heatmap shape: {frames.shape}")
    frames = np.squeeze(frames, 0)
    assert frames.shape == (8, 256, 256, 1), "Input shape must be (1, 8, 256, 256, 1)"

    # frames = frames[0]  # Remove batch dimension
    stacked_rows = []

    for i in range(2):  # Two rows
        stacked_columns = []
        for j in range(i * 4, (i * 4) + 4):  # Four columns per row
            heatmap = frames[j, :, :, :]  # Extract frame (H, W, 3)

            if j < (i * 4) + 4 - 1:
                heatmap = add_col_padding(heatmap)  # Add vertical separator

            stacked_columns.append(heatmap)

        hstack = np.hstack(stacked_columns)  # Stack columns horizontally
        if i < 1:
            hstack = add_row_padding(hstack)  # Add horizontal separator

        stacked_rows.append(hstack)

    res = np.vstack(stacked_rows)  # Stack rows vertically
    return res


@tensorleap_custom_visualizer('frames_grid', visualizer_type=LeapDataType.Image, heatmap_function=frames_grid_heatmap)
def frames_grid_visualzier(frames: NDArray[np.float32]) -> LeapImage:
    """
    Arranges a batch of RGB video frames into a 2-row, 4-column grid with separators.

    Args:
        images (np.ndarray): Input array of shape (1, 3, 8, 224, 224).

    Returns:
        np.ndarray: Concatenated array of images with separators.
    """
    frames = np.squeeze(frames, 0)
    print(f"$$$ frames_grid shape: {frames.shape}")
    assert frames.shape == (3, 8, 256, 256), "Input shape must be (1, 3, 8, 224, 224)"

    # frames = frames[0]  # Remove batch dimension
    stacked_rows = []

    for i in range(2):  # Two rows
        stacked_columns = []
        for j in range(i * 4, (i * 4) + 4):  # Four columns per row
            image = frames[:, j, :, :]  # Extract frame
            image = np.moveaxis(image, 0, -1)  # Convert (C, H, W) to (H, W, C)
            image = rescale_min_max(image)  # Rescale values if necessary

            if j < (i * 4) + 4 - 1:
                image = add_col_padding(image)  # Add vertical separator

            stacked_columns.append(image)

        hstack = np.hstack(stacked_columns)  # Stack columns horizontally
        if i < 1:
            hstack = add_row_padding(hstack)  # Add horizontal separator

        stacked_rows.append(hstack)

    res = np.vstack(stacked_rows)  # Stack rows vertically
    return LeapImage(res)


def frame_heatmap(frames: NDArray[np.float32]) -> LeapImage:
    print(f"$$$ heatmap shape: {frames.shape}")
    return frames[0, 0, ...]


@tensorleap_custom_visualizer('frame', visualizer_type=LeapDataType.Image, heatmap_function=frame_heatmap)
def frame_visualzier(frames: np.ndarray) -> LeapImage:
    frames = np.squeeze(frames, 0)
    # frames = frames.transpose(1,0,2,3)
    frame = frames[:, 0, ...].transpose(1, 2, 0)
    frame = ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8)
    return LeapImage(frame)


# Adding a name to the prediction, and supplying it with label names.
leap_binder.add_prediction(name='classes', labels=[])


if __name__ == '__main__':
    leap_binder.check()
