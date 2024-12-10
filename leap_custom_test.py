from leap_binder import input_encoder, preprocess_func_leap, leap_binder, input_encoder_2,  input_encoder, gt_encoder, dummy_loss
import os
import numpy as np
from code_loader.helpers import visualize
import onnxruntime as ort
import tensorflow as tf

def check_custom_test():
    check_generic = True
    plot_vis = True
    if check_generic:
        leap_binder.check()
    print("started custom tests")
    responses = preprocess_func_leap()
    ort_session = ort.InferenceSession("model/new_2.onnx")
    for subset in responses:  # train, val
        for idx in range(3): # analyze first 3 images
            image = input_encoder(idx, subset)
            input_2 = input_encoder_2(idx, subset)
            onnx_outputs = ort_session.run(None, {'input_rgb': image[None, ...],
                                                                           'input_pick_features': input_2[None, ...]})
            onnx_outputs = [tf.convert_to_tensor(otpt) for otpt in onnx_outputs]
            # get input and gt
            gt = gt_encoder(idx, subset)


            # infer model
            ls = dummy_loss(onnx_outputs[0][None, ...],
                                   onnx_outputs[1][None, ...],
                                   onnx_outputs[2][None, ...],
                                   tf.convert_to_tensor(gt[None, ...]))
            # print metadata
            for metadata_handler in leap_binder.setup_container.metadata:
                curr_metadata = metadata_handler.function(idx, subset)
                print(f"Metadata {metadata_handler.name}: {curr_metadata}")

    print("finish tests")


if __name__ == '__main__':
    check_custom_test()
