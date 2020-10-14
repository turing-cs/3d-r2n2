import tensorflow as tf
from lib import utils


def shuffle_sequence(value):
    with tf.compat.v1.name_scope("shuffle_sequence"):
        ret = tf.transpose(a=value, perm=[1, 0, 2, 3, 4])
        ret = tf.random.shuffle(ret)
        ret = tf.transpose(a=ret, perm=[1, 0, 2, 3, 4])
    return value


class Preprocessor():
    def __init__(self, X):
        with tf.compat.v1.name_scope("Preprocessor"):
            params = utils.read_params()
            if params["TRAIN"]["TIME_STEP_COUNT"] == "RANDOM":
                n_timesteps = tf.random.uniform(
                    [], minval=1, maxval=25, dtype=tf.int32)
                tf.compat.v1.summary.scalar("n_timesteps", n_timesteps)
            elif isinstance(params["TRAIN", "TIME_STEP_COUNT"], int) and params["TRAIN"]["TIME_STEP_COUNT"] > 0:
                n_timesteps = params["TRAIN"]["TIME_STEP_COUNT"]
            else:
                n_timesteps = tf.shape(input=X)[1]

            n_batchsize = tf.shape(input=X)[0]
            X_dropped_alpha = X[:, :, :, :, 0:3]  # drop alpha channel
            X_cropped = tf.image.random_crop(
                X_dropped_alpha, [n_batchsize, n_timesteps, 127, 127, 3])   # randomly crop

            if params["TRAIN"]["SHUFFLE_IMAGE_SEQUENCE"]:
                X_shuffled = shuffle_sequence(X_cropped)
                self.out_tensor = X_shuffled
            else:
                self.out_tensor = X_cropped
