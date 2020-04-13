import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
from model.style_content_processing import StyleContentCreator

EPOCHS = 10
STEPS = 100


def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def imshow(image, ax, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    ax.imshow(image)
    if title:
        ax.set_title(title)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--content_path', required=True, help='relative path to the content image')
    ap.add_argument('-s', '--style_path', required=True, help='relative path to the style image')
    args = vars(ap.parse_args())

    # Test if GPU is available
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        print('GPU not available')
    else:
        print('GPU available')

    # Load images from specified paths
    content_path = args['content_path']
    style_path = args['style_path']
    content_image = load_img(content_path)
    style_image = load_img(style_path)

    # Show images
    fig, (ax_c, ax_s) = plt.subplots(ncols=2)
    imshow(content_image, ax_c, title='Content image')
    imshow(style_image, ax_s, title='Style image')
    plt.show()

    # Set content and style layers of interest
    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    # Build StyleContentCreator
    creator = StyleContentCreator(style_image, content_image, style_layers, content_layers)
    # Compute output image
    out = creator.compute_image(EPOCHS, STEPS)
    out.show()


if __name__ == '__main__':
    main()
