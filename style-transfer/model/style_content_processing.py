import tensorflow as tf
import numpy as np
import PIL.Image
import time
import IPython.display as display


def vgg_layers(layer_names):
    """ Creates a VGG model that returns a list of intermediate output values."""
    # Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations


def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


class StyleContentExtractor(tf.keras.models.Model):

    def __init__(self, style_layers, content_layers):
        super(StyleContentExtractor, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs, **kwargs):
        # Expects float input in [0, 1]
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}


class StyleContentCreator:

    def __init__(self,
                 style_image,
                 content_image,
                 style_layers,
                 content_layers,
                 style_weight=1e-2,
                 content_weight=1e4,
                 variation_weight=30):
        # Save content image and variation weight as instance attributes
        self.content_image = content_image
        self.variation_weight = variation_weight

        # Get number of layers
        num_content_layers = len(content_layers)
        num_style_layers = len(style_layers)

        # Build StyleContentExtractor
        self.extractor = StyleContentExtractor(style_layers, content_layers)
        # Get style and content
        style_targets = self.extractor(style_image)['style']
        content_targets = self.extractor(content_image)['content']

        # Create optimizer
        self.opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

        # Declare loss function
        def style_content_loss(outputs):
            style_outputs = outputs['style']
            content_outputs = outputs['content']
            style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                                   for name in style_outputs.keys()])
            style_loss *= style_weight / num_style_layers

            content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                                     for name in content_outputs.keys()])
            content_loss *= content_weight / num_content_layers
            loss = style_loss + content_loss
            return loss

        self.loss = style_content_loss

    def compute_image(self, epochs, steps_per_epoch):
        # Prepare output image
        image = tf.Variable(self.content_image)

        # Define function to compute one step of the gradient descent
        @tf.function()
        def train_step(img):
            with tf.GradientTape() as tape:
                outputs = self.extractor(img)
                loss = self.loss(outputs)
                loss += self.variation_weight * tf.image.total_variation(img)

            grad = tape.gradient(loss, img)
            self.opt.apply_gradients([(grad, img)])
            img.assign(tf.clip_by_value(img, clip_value_min=0.0, clip_value_max=1.0))

        # Perform the training step according to the number of epochs specified
        start = time.time()
        step = 0
        for n in range(epochs):
            for m in range(steps_per_epoch):
                step += 1
                train_step(image)
                print(".", end='')
            display.clear_output(wait=True)
            display.display(tensor_to_image(image))
            print("Train step: {}".format(step))
        end = time.time()
        print("Total time: {:.1f}".format(end-start))

        return tensor_to_image(image)
