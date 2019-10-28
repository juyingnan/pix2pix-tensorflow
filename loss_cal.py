import tensorflow as tf
from skimage import io

def cal_loss(targets, outputs, ch, masked=True):
    _targets = targets
    _outputs = outputs

    if masked:
        if ch == 3:
            mask_layer = tf.math.ceil(
                (tf.abs(targets[..., 0]) + tf.abs(targets[..., 1]) + tf.abs(targets[..., 2])) / 3)
            mask = tf.stack([mask_layer, mask_layer, mask_layer], axis=3)
        else:
            mask_layer = targets[..., 3]
            mask = tf.stack([mask_layer, mask_layer, mask_layer, mask_layer], axis=2)
        _targets = (_targets * mask)[..., :3]
        _outputs = (_outputs * mask)[..., :3]

    # if a.norm:
    #     _outputs = tf.keras.backend.l2_normalize(_outputs, axis=a.norm_axis)
    #     _targets = tf.keras.backend.l2_normalize(_targets, axis=a.norm_axis)
    #
    # if a.cosine:
    #     gen_loss_L1 = tf.reduce_mean(tf.abs(tf.losses.cosine_distance(_targets, _outputs, axis=a.norm_axis)))
    # else:
    _targets = tf.cast(_targets, tf.float32)
    _outputs = tf.cast(_outputs, tf.float32)
    gen_loss_L1 = tf.reduce_mean(tf.abs(_targets - _outputs))
    return gen_loss_L1


target_path = r'D:\Projects\SIIE\helsinki\20191024\normal_depthcombine\test\normal_4ch.png'
output_path = r'C:\Users\bunny\Desktop\paro_3.png'

target = io.imread(target_path)
output = io.imread(output_path)

loss = cal_loss(target, output, 4, masked=True)
print(loss)