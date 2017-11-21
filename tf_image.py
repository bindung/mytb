# Copyright 2015 The TensorFlow Authors and Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Custom image operations.
Most of the following methods extend TensorFlow image library, and part of
the code is shameless copy-paste of the former!
"""
import tensorflow as tf
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
import tf_extended as tfe

_R_MEAN = 123.
_G_MEAN = 117.
_B_MEAN = 104.

# Some training pre-processing parameters.
BBOX_CROP_OVERLAP = 0.1  # Minimum overlap to keep a bbox after cropping.
CROP_RATIO_RANGE = (0.7, 1.3)  # Distortion ratio during cropping.
EVAL_SIZE = (300, 300)


# =========================================================================== #
# Modification of TensorFlow image routines.
# =========================================================================== #
def Random_Brightness(image, max_delta=32.):
    """ Random_brightness function 
    With probably of 0.5, return the original image, otherwide, random brightness according 
    the max_delta value.
    Args:
        image: an image must be in range(0,1)
        max_delta: arg in tf.image_random_Brightness
    Returns:
        image_util: 
    """
    choice = tf.random_uniform([], minval=0, maxval=2, dtype=tf.int32)
    image = tf.cond(tf.equal(choice, tf.constant(1)),
                    lambda: image,
                    lambda: tf.image.random_brightness(image, max_delta=max_delta / 255.))
    return image


def Random_Contrast(image, lower=0.5, upper=1.5):
    """ Random_Contrast function 
    With probably of 0.5, return the original image, otherwide, random contrast according 
    the upper and lower value.
    Args:
        image: an image must be in range(0,1)
        lower & upper : paras in tf.image.random_contrast
    Returns:
        image_util: 
    """
    choice = tf.random_uniform([], minval=0, maxval=2, dtype=tf.int32)
    image = tf.cond(tf.equal(choice, tf.constant(1)),
                    lambda: image,
                    lambda: tf.image.random_contrast(image, lower=lower, upper=upper))
    return image


def Random_Hue(image, max_delta=0.2):
    """ Random_Hue function
    With probably of 0.5, return the original image, otherwide, random Hue according
    the max_delta value.
    Args:
        image: an image must be in range(0,1)
        lower & upper : paras in tf.image.random_hue
    Returns:
        image_util:
    """
    choice = tf.random_uniform([], minval=0, maxval=2, dtype=tf.int32)
    image = tf.cond(tf.equal(choice, tf.constant(1)),
                    lambda: image,
                    lambda: tf.image.random_hue(image, max_delta=max_delta))
    return image


def Random_Saturation(image, lower=0.5, upper=1.5):
    """ Random_Hue function
    With probably of 0.5, return the original image, otherwide, random Hue according
    the lower upper value.
    Args:
        image: an image must be in range(0,1)
        lower & upper : paras in tf.image.random_saturation
    Returns:
        image_util:
    """
    choice = tf.random_uniform([], minval=0, maxval=2, dtype=tf.int32)
    image = tf.cond(tf.equal(choice, tf.constant(1)),
                    lambda: image,
                    lambda: tf.image.random_saturation(image, lower=lower, upper=upper))
    return image


def Convert2HSV(image):
    """ Convert2HSV
    Convert image format from RGB to HSV
    :param image: HSV format
    :return: image : HSV format
    """
    return tf.image.rgb_to_hsv(image)


def Convert2RGB(image):
    """ Convert2RGB
    Convert image format from HSV to RGB
    :param image:
    :return: image: RGB format
    """
    return tf.image.hsv_to_rgb(image)


def Random_light_noise(image):
    """ Random_light_noise
    Shuffle the channels of images
    :param image: with RGB or HSV
    :return: image: shuffled channels
    """

    def f1():
        image_tensor = tf.transpose(image, perm=[2, 0, 1])
        image_tensor = tf.random_shuffle(image_tensor)
        return tf.transpose(image_tensor, perm=[1, 2, 0])

    choice = tf.random_uniform([], minval=0, maxval=2, dtype=tf.int32)
    image = tf.cond(tf.greater(choice, tf.constant(0)),
                    lambda: image,
                    f1)
    return image


def resize_image_bboxes_with_crop_or_pad2(image, bboxes, height, width):
    """resize_image_bboxes_with_crop_or_pad2
    Random resize image into a large image, ratio is choosen from [1,2,]
    :param image:
    :param bboxes:
    :return: images:
             bboxes:
    """
    height = tf.cast(height, tf.int32)
    width = tf.cast(width, tf.int32)
    ratio = tf.random_uniform([], minval=1, maxval=3, dtype=tf.int32)

    def f1():
        return image, bboxes

    def f2():
        offset_height = tf.random_uniform([], minval=0, maxval=ratio * height - height, dtype=tf.int32)
        offset_width = tf.random_uniform([], minval=0, maxval=ratio * width - width, dtype=tf.int32)
        target_height = height * ratio
        target_width = width * ratio
        image_tensor = tf.image.pad_to_bounding_box(image,
                                                    offset_height,
                                                    offset_width,
                                                    target_height,
                                                    target_width)
        bboxes_pad = bboxes_crop_or_pad(bboxes, height, width,
                                        offset_height, offset_width,
                                        target_height, target_width)
        return image_tensor, bboxes_pad

    image, bboxes = tf.cond(tf.greater(ratio, tf.constant(1)), f2, f1)
    return image, bboxes


def distorted_bounding_box_crop(image,
                                labels,
                                bboxes,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.3, 2.0),
                                area_range=(0.1, 1.0),
                                max_attempts=200,
                                scope=None):
    """Generates cropped_image using a one of the bboxes randomly distorted.
    Args:
        image: A `Tensor` representing an image of arbitrary size.
        labels : A Tensor inlcudes all labels
        bboxes : A Tensor inlcudes cordinates of bbox in shape [N, 4]
        min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
            area of the image must contain at least this fraction of any bounding box
            supplied.
        aspect_ratio_range: An optional list of `floats`. The cropped area of the
            image must have an aspect ratio = width / height within this range.
        area_range: An optional list of `floats`. The cropped area of the image
            must contain a fraction of the supplied image within in this range.
        max_attempts: An optional `int`. Number of attempts at generating a cropped
            region of the image of the specified constraints. After `max_attempts`
            failures, return the entire image.
        scope: Optional scope for name_scope.
    Returns:
        A tuple, a 3-D Tensor cropped_image and the distorted bbox
    """
    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, labels, bboxes]):
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].
        bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=tf.expand_dims(bboxes, 0),
            min_object_covered=min_object_covered,
            aspect_ratio_range=CROP_RATIO_RANGE,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=False)
        distort_bbox = distort_bbox[0, 0]

        # Crop the image to the specified bounding box.
        cropped_image = tf.slice(image, bbox_begin, bbox_size)

        # Update bounding boxes: resize and filter out.
        bboxes = tfe.bboxes_resize(distort_bbox, bboxes)
        labels, bboxes, num = tfe.bboxes_filter_overlap(labels, bboxes,
                                                        BBOX_CROP_OVERLAP)
        return cropped_image, labels, bboxes, num


def distorter(images):
    """ distorter
    Integrate all distort functions in a pipeline
    :param image:
    :return: image with distortion
    """

    def f1():
        image = Random_Brightness(images)
        image = Random_Contrast(image)
        # image = Convert2HSV(image)
        image = Random_Saturation(image)
        image = Random_Hue(image)
        # image = Convert2RGB(image)
        return image

    def f2():
        image = Random_Brightness(images)
        # image = Convert2HSV(image)
        image = Random_Saturation(image)
        image = Random_Hue(image)
        # image = Convert2RGB(image)
        image = Random_Contrast(image)
        return image

    choice = tf.random_uniform([], minval=0, maxval=2, dtype=tf.int32)
    image = tf.cond(tf.equal(choice, tf.constant(1)),
                    f1,
                    f2)
    image = Random_light_noise(image)
    return image


def Random_crop(image, labels, bboxes):
    """ Random crop
    With the probablity of 1/6 , it will return the original pic.
    Or it will random crop. The crop ratio is randomly choosen from np.linspace(0,1.,21_
    :param image:
    :param out_shape:
    :param labels:
    :param bboxes:
    :return: image:
             labels:
             bboxes:
    """

    def f1():
        return image, labels, bboxes

    def f2(image=image, labels=labels, bboxes=bboxes):
        num = tf.constant(1)

        def random_distorted_bounding_box_crop(vals, index):
            object_covered = np.linspace(0.1, 1., 19)
            image, labels, bboxes, num = vals
            return distorted_bounding_box_crop(image, labels, bboxes,
                                               min_object_covered=object_covered[index])

        vals = \
            _apply_with_random_selector_tuples((image, labels, bboxes, num),
                                               random_distorted_bounding_box_crop,
                                               num_cases=19)
        image, labels, bboxes, num = vals

        return image, labels, bboxes

    object_covered = tf.random_uniform([], minval=0, maxval=6, dtype=tf.int32, seed=None, name=None)
    #image_tensor, labels, bboxes = tf.cond(tf.greater(object_covered, tf.constant(4)), f1, f2)
    image_tensor, labels, bboxes = f2()
    return image_tensor, labels, bboxes


def _assert(cond, ex_type, msg):
    """A polymorphic assert, works with tensors and boolean expressions.
    If `cond` is not a tensor, behave like an ordinary assert statement, except
    that a empty list is returned. If `cond` is a tensor, return a list
    containing a single TensorFlow assert op.
    Args:
      cond: Something evaluates to a boolean value. May be a tensor.
      ex_type: The exception class to use.
      msg: The error message.
    Returns:
      A list, containing at most one assert op.
    """
    if _is_tensor(cond):
        return [control_flow_ops.Assert(cond, [msg])]
    else:
        if not cond:
            raise ex_type(msg)
        else:
            return []


def _is_tensor(x):
    """Returns `True` if `x` is a symbolic tensor-like object.
    Args:
      x: A python object to check.
    Returns:
      `True` if `x` is a `tf.Tensor` or `tf.Variable`, otherwise `False`.
    """
    return isinstance(x, (ops.Tensor, variables.Variable))


def _ImageDimensions(image):
    """Returns the dimensions of an image tensor.
    Args:
      image: A 3-D Tensor of shape `[height, width, channels]`.
    Returns:
      A list of `[height, width, channels]` corresponding to the dimensions of the
        input image.  Dimensions that are statically known are python integers,
        otherwise they are integer scalar tensors.
    """
    if image.get_shape().is_fully_defined():
        return image.get_shape().as_list()
    else:
        static_shape = image.get_shape().with_rank(3).as_list()
        dynamic_shape = array_ops.unstack(array_ops.shape(image), 3)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]


def _Check3DImage(image, require_static=True):
    """Assert that we are working with properly shaped image.
    Args:
      image: 3-D Tensor of shape [height, width, channels]
        require_static: If `True`, requires that all dimensions of `image` are
        known and non-zero.
    Raises:
      ValueError: if `image.shape` is not a 3-vector.
    Returns:
      An empty list, if `image` has fully defined dimensions. Otherwise, a list
        containing an assert op is returned.
    """
    try:
        image_shape = image.get_shape().with_rank(3)
    except ValueError:
        raise ValueError("'image' must be three-dimensional.")
    if require_static and not image_shape.is_fully_defined():
        raise ValueError("'image' must be fully defined.")
    if any(x == 0 for x in image_shape):
        raise ValueError("all dims of 'image.shape' must be > 0: %s" %
                         image_shape)
    if not image_shape.is_fully_defined():
        return [check_ops.assert_positive(array_ops.shape(image),
                                          ["all dims of 'image.shape' "
                                           "must be > 0."])]
    else:
        return []


def fix_image_flip_shape(image, result):
    """Set the shape to 3 dimensional if we don't know anything else.
    Args:
      image: original image size
      result: flipped or transformed image
    Returns:
      An image whose shape is at least None,None,None.
    """
    image_shape = image.get_shape()
    if image_shape == tensor_shape.unknown_shape():
        result.set_shape([None, None, None])
    else:
        result.set_shape(image_shape)
    return result


# =========================================================================== #
# Image + BBoxes methods: cropping, resizing, flipping, ...
# =========================================================================== #
def bboxes_crop_or_pad(bboxes,
                       height, width,
                       offset_y, offset_x,
                       target_height, target_width):
    """Adapt bounding boxes to crop or pad operations.
    Coordinates are always supposed to be relative to the image.

    Arguments:
      bboxes: Tensor Nx4 with bboxes coordinates [y_min, x_min, y_max, x_max];
      height, width: Original image dimension;
      offset_y, offset_x: Offset to apply,
        negative if cropping, positive if padding;
      target_height, target_width: Target dimension after cropping / padding.
    """
    with tf.name_scope('bboxes_crop_or_pad'):
        # Rescale bounding boxes in pixels.
        scale = tf.cast(tf.stack([height, width, height, width]), bboxes.dtype)
        bboxes = bboxes * scale
        # Add offset.
        offset = tf.cast(tf.stack([offset_y, offset_x, offset_y, offset_x]), bboxes.dtype)
        bboxes = bboxes + offset
        # Rescale to target dimension.
        scale = tf.cast(tf.stack([target_height, target_width,
                                  target_height, target_width]), bboxes.dtype)
        bboxes = bboxes / scale
        return bboxes


def resize_image_bboxes_with_crop_or_pad(image, bboxes,
                                         target_height, target_width):
    """Crops and/or pads an image to a target width and height.
    Resizes an image to a target width and height by either centrally
    cropping the image or padding it evenly with zeros.

    If `width` or `height` is greater than the specified `target_width` or
    `target_height` respectively, this op centrally crops along that dimension.
    If `width` or `height` is smaller than the specified `target_width` or
    `target_height` respectively, this op centrally pads with 0 along that
    dimension.
    Args:
      image: 3-D tensor of shape `[height, width, channels]`
      target_height: Target height.
      target_width: Target width.
    Raises:
      ValueError: if `target_height` or `target_width` are zero or negative.
    Returns:
      Cropped and/or padded image of shape
        `[target_height, target_width, channels]`
    """
    with tf.name_scope('resize_with_crop_or_pad'):
        image = ops.convert_to_tensor(image, name='image')

        assert_ops = []
        assert_ops += _Check3DImage(image, require_static=False)
        assert_ops += _assert(target_width > 0, ValueError,
                              'target_width must be > 0.')
        assert_ops += _assert(target_height > 0, ValueError,
                              'target_height must be > 0.')

        image = control_flow_ops.with_dependencies(assert_ops, image)
        # `crop_to_bounding_box` and `pad_to_bounding_box` have their own checks.
        # Make sure our checks come first, so that error messages are clearer.
        if _is_tensor(target_height):
            target_height = control_flow_ops.with_dependencies(
                assert_ops, target_height)
        if _is_tensor(target_width):
            target_width = control_flow_ops.with_dependencies(assert_ops, target_width)

        def max_(x, y):
            if _is_tensor(x) or _is_tensor(y):
                return math_ops.maximum(x, y)
            else:
                return max(x, y)

        def min_(x, y):
            if _is_tensor(x) or _is_tensor(y):
                return math_ops.minimum(x, y)
            else:
                return min(x, y)

        def equal_(x, y):
            if _is_tensor(x) or _is_tensor(y):
                return math_ops.equal(x, y)
            else:
                return x == y

        height, width, _ = _ImageDimensions(image)
        width_diff = target_width - width
        offset_crop_width = max_(-width_diff // 2, 0)
        offset_pad_width = max_(width_diff // 2, 0)

        height_diff = target_height - height
        offset_crop_height = max_(-height_diff // 2, 0)
        offset_pad_height = max_(height_diff // 2, 0)

        # Maybe crop if needed.
        height_crop = min_(target_height, height)
        width_crop = min_(target_width, width)
        cropped = tf.image.crop_to_bounding_box(image, offset_crop_height, offset_crop_width,
                                                height_crop, width_crop)
        bboxes = bboxes_crop_or_pad(bboxes,
                                    height, width,
                                    -offset_crop_height, -offset_crop_width,
                                    height_crop, width_crop)
        # Maybe pad if needed.
        resized = tf.image.pad_to_bounding_box(cropped, offset_pad_height, offset_pad_width,
                                               target_height, target_width)
        bboxes = bboxes_crop_or_pad(bboxes,
                                    height_crop, width_crop,
                                    offset_pad_height, offset_pad_width,
                                    target_height, target_width)

        # In theory all the checks below are redundant.
        if resized.get_shape().ndims is None:
            raise ValueError('resized contains no shape.')

        resized_height, resized_width, _ = _ImageDimensions(resized)

        assert_ops = []
        assert_ops += _assert(equal_(resized_height, target_height), ValueError,
                              'resized height is not correct.')
        assert_ops += _assert(equal_(resized_width, target_width), ValueError,
                              'resized width is not correct.')

        resized = control_flow_ops.with_dependencies(assert_ops, resized)
        return resized, bboxes


def resize_image(image, size,
                 method=tf.image.ResizeMethod.BILINEAR,
                 align_corners=False):
    """Resize an image and bounding boxes.
    """
    # Resize image.
    with tf.name_scope('resize_image'):
        color_ordering = np.random.randint(4)
        #height, width, channels = _ImageDimensions(image)
        #image = tf.expand_dims(image, 0)
        image = tf.image.resize_images(image, size, color_ordering, align_corners)
        #image = tf.reshape(image, tf.stack([size[0], size[1], 3]))
        return image


def random_flip_left_right(image, bboxes, seed=None):
    """Random flip left-right of an image and its bounding boxes.
    """

    def flip_bboxes(bboxes):
        """Flip bounding boxes coordinates.
        """
        bboxes = tf.stack([bboxes[:, 0], 1 - bboxes[:, 3],
                           bboxes[:, 2], 1 - bboxes[:, 1]], axis=-1)
        return bboxes

    # Random flip. Tensorflow implementation.
    with tf.name_scope('random_flip_left_right'):
        image = ops.convert_to_tensor(image, name='image')
        # _Check3DImage(image, require_static=False)
        uniform_random = random_ops.random_uniform([], 0, 1.0, seed=seed)
        mirror_cond = math_ops.less(uniform_random, .5)
        # Flip image.
        result = control_flow_ops.cond(mirror_cond,
                                       lambda: array_ops.reverse_v2(image, [1]),
                                       lambda: image)
        # Flip bboxes.
        bboxes = control_flow_ops.cond(mirror_cond,
                                       lambda: flip_bboxes(bboxes),
                                       lambda: bboxes)
        return fix_image_flip_shape(image, result), bboxes


def distort_color(image, scope=None):
    """Distort the color of the image.

    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.

    Args:
    image: Tensor containing single image.
    thread_id: preprocessing thread ID.
    scope: Optional scope for op_scope.
    Returns:
    color-distorted image
    """
    color_ordering = np.random.randint(2)
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)

    # The random_* ops do not necessarily clamp.
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def distort_color_2(image, color_ordering=0, fast_mode=True, scope=None):
    """Distort the color of a Tensor image.

    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.

    Args:
        image: 3-D Tensor containing single image in [0, 1].
        color_ordering: Python int, a type of distortion (valid values: 0-3).
        fast_mode: Avoids slower ops (random_hue and random_contrast)
        scope: Optional scope for name_scope.
    Returns:
        3-D Tensor color-distorted image on range [0, 1]
    Raises:
        ValueError: if color_ordering not in [0, 3]
    """
    # color_ordering = np.random.randint(4)
    with tf.name_scope(scope, 'distort_color', [image]):
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
            else:
                raise ValueError('color_ordering must be in [0, 3]')
        # The random_* ops do not necessarily clamp


        return tf.clip_by_value(image, 0.0, 1.0)


def tf_summary_image(image, bboxes, name='image', unwhitened=False):
    """Add image with bounding boxes to summary.
    """
    if unwhitened:
        image = tf_image_unwhitened(image)
    image = tf.expand_dims(image, 0)
    bboxes = tf.expand_dims(bboxes, 0)
    image_with_box = tf.image.draw_bounding_boxes(image, bboxes)
    tf.summary.image(name, image_with_box)


def tf_image_whitened(image, means=[_R_MEAN, _G_MEAN, _B_MEAN]):
    """Subtracts the given means from each image channel.

    Returns:
        the centered image.
    """
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    mean = tf.constant(means, dtype=image.dtype) / 255.
    image = image - mean
    return image


def tf_image_unwhitened(image, means=[_R_MEAN, _G_MEAN, _B_MEAN]):
    """Re-convert to original image distribution, and convert to int if
    necessary.

    Returns:
      Centered image.
    """
    mean = tf.constant(means, dtype=image.dtype) / 255.
    image = image + mean
    image = tf.clip_by_value(image, 0., 1.)
    return image


def np_image_unwhitened(image, means=[_R_MEAN, _G_MEAN, _B_MEAN], to_int=True):
    """Re-convert to original image distribution, and convert to int if
    necessary. Numpy version.

    Returns:
      Centered image.
    """
    img = np.copy(image)
    img += np.array(means, dtype=img.dtype)
    if to_int:
        img = img.astype(np.uint8)
    return img


def apply_with_random_selector(x, func, num_cases):
    """Computes func(x, sel), with sel sampled from [0...num_cases-1].

    Args:
        x: input Tensor.
        func: Python function to apply.
        num_cases: Python int32, number of cases to sample sel from.

    Returns:
        The result of func(x, sel), where func receives the value of the
        selector as a python integer, but sel is sampled dynamically.
    """
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    # Pass the real x only to one of the func calls.
    return control_flow_ops.merge([
        func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
        for case in range(num_cases)])[0]


def _apply_with_random_selector_tuples(x, func, num_cases):
    """Computes func(x, sel), with sel sampled from [0...num_cases-1].

    Args:
        x: A tuple of input tensors.
        func: Python function to apply.
        num_cases: Python int32, number of cases to sample sel from.

    Returns:
        The result of func(x, sel), where func receives the value of the
        selector as a python integer, but sel is sampled dynamically.
    """
    num_inputs = len(x)
    rand_sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    # Pass the real x only to one of the func calls.

    tuples = [list() for t in x]
    for case in range(num_cases):
        new_x = [control_flow_ops.switch(t, tf.equal(rand_sel, case))[1] for t in x]
        output = func(tuple(new_x), case)
        for j in range(num_inputs):
            tuples[j].append(output[j])

    for i in range(num_inputs):
        tuples[i] = control_flow_ops.merge(tuples[i])[0]
    return tuple(tuples)
