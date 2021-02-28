#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 23:56:17 2021

This is the code to build a WaveGAN using the Keras API. My motivation for 
this was to build a high-level implementation of the model so that it can be 
easily tampered with. No original concepts introduced, just the classic 
WaveGAN in Keras.

Segments of code have been used from:
    - the original WaveGAN code by Chris Donahue
    https://github.com/chrisdonahue/wavegan/
    
    - the conditional WaveGAN by Adrian Barahona
    https://github.com/adrianbarahona/conditional_wavegan_knocking_sounds/
    
    - the official Keras documentation on WGAN-GP (for defining the loss)
    https://keras.io/examples/generative/wgan_gp/

@author: nk
"""
#%% Dependencies:

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#%%
'''
Utility functions to construct the WaveGAN:
'''
def apply_phaseshuffle(x, rad, pad_type='reflect'):
    '''
    Phase Reshuffle
    
    As implemented by Chris Donahue for the original WaveGAN
    https://github.com/chrisdonahue/wavegan/
    '''
    b, x_len, nch = x.get_shape().as_list()
    
    phase = tf.random.uniform([], minval=-rad, maxval=rad + 1, dtype=tf.int32)
    pad_l = tf.maximum(phase, 0)
    pad_r = tf.maximum(-phase, 0)
    phase_start = pad_r
    
    x = tf.pad(x, [[0, 0], [pad_l, pad_r], [0, 0]], mode=pad_type)

    x = x[:, phase_start:phase_start+x_len]
    x.set_shape([b, x_len, nch])

    return x

def Conv1DTranspose(input_tensor, filters, kernel_size, strides=4, padding='same'
                    , name = None, activation = 'relu'):
    '''
    1D Convolutional Transpose layer
    
    Keras does not offer a callable transpose convolutional layer in 1D, 
    so one has to be defined using the 2D counterpart.
    
    Credit for this code to Adrian Barahona, who implemented it in his 
    conditional version of the WaveGAN:
    https://github.com/adrianbarahona/conditional_wavegan_knocking_sounds/
    '''
    x = layers.Conv2DTranspose(filters=filters, kernel_size=(1, kernel_size), strides=(1, strides), padding=padding, 
                        name = name, activation = activation)(keras.backend.expand_dims(input_tensor, axis=1))
    x = keras.backend.squeeze(x, axis=1)
    
    return x

#%%
'''
DISCRIMINATOR
'''
def make_discriminator(audio_input_dim = 16384,
                       dim = 64,
                       kernel_len = 25,
                       phaseshuffle_rad = 2,
                       return_sbf = False):
    if phaseshuffle_rad > 0:
        phaseshuffle = lambda x: apply_phaseshuffle(x, phaseshuffle_rad)
    else:
        phaseshuffle = lambda x: x
        
    discriminator_input = layers.Input(shape = (audio_input_dim, 1))

    x = layers.Conv1D(dim, kernel_len, strides = 4, padding = 'same')(discriminator_input)
    x = layers.LeakyReLU(alpha = 0.2)(x)
    x = phaseshuffle(x, phaseshuffle_rad)
    
    x = layers.Conv1D(dim * 2, kernel_len, strides = 4, padding = 'same')(x)
    x = layers.LeakyReLU(alpha = 0.2)(x)
    x = phaseshuffle(x, phaseshuffle_rad)
    
    x = layers.Conv1D(dim * 4, kernel_len, strides = 4, padding = 'same')(x)
    x = layers.LeakyReLU(alpha = 0.2)(x)
    x = phaseshuffle(x, phaseshuffle_rad)
    
    x = layers.Conv1D(dim * 8, kernel_len, strides = 4, padding = 'same')(x)
    x = layers.LeakyReLU(alpha = 0.2)(x)
    x = phaseshuffle(x, phaseshuffle_rad)
    
    x = layers.Conv1D(dim * 16, kernel_len, strides = 4, padding = 'same')(x)
    x = layers.LeakyReLU(alpha = 0.2)(x)
    
    shape_before_flatten = keras.backend.int_shape(x)
    
    x = layers.Flatten()(x)
    discriminator_output = layers.Dense(1, activation  = 'sigmoid')(x)
    
    discriminator = keras.Model(discriminator_input, discriminator_output, name = 'discriminator')
    
    if return_sbf:
        return discriminator, shape_before_flatten
  
    else:
        return discriminator
    
#%%
'''
GENERATOR
'''
def make_generator(latent_dim = 100,
                   audio_input_dim = 16384,
                   dim = 64,
                   kernel_len = 25,
                   shape_before_flatten = (None, 16, 1024)):
    generator_input = layers.Input(shape = (latent_dim,))
    
    x = layers.Dense(audio_input_dim)(generator_input)
    x = layers.Reshape((shape_before_flatten[1], shape_before_flatten[2]))(x)
    x = layers.ReLU()(x)
    
    x = Conv1DTranspose(x, dim * 8, kernel_size = kernel_len)
    x = layers.ReLU()(x)
    
    x = Conv1DTranspose(x, dim * 4, kernel_size = kernel_len)
    x = layers.ReLU()(x)
    
    x = Conv1DTranspose(x, dim * 2, kernel_size = kernel_len)
    x = layers.ReLU()(x)
    
    x = Conv1DTranspose(x, dim, kernel_size = kernel_len)
    x = layers.ReLU()(x)
    
    x = Conv1DTranspose(x, 1, kernel_size = kernel_len, activation = 'tanh')
    
    generator_output = x
    
    generator = keras.Model(generator_input, generator_output, name = 'generator')
    
    return generator

#%%
'''
WaveGAN CLASS
'''
class WaveGAN(keras.Model):
    def __init__(
        self,
        discriminator,
        generator,
        latent_dim=100,
        discriminator_extra_steps=5,
        gp_weight=10.0,
        ):
        
        super(WaveGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WaveGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_images):
        
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        
        # Get the batch size
        batch_size = tf.shape(real_images)[0]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator(random_latent_vectors, training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_images, training=True)
                # Get the logits for the real images
                real_logits = self.discriminator(real_images, training=True)

                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_images, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}
    
#%%
'''
OPTIMIZERS AND LOSS FUNCTIONS
'''
# Instantiate the optimizer for both networks
generator_optimizer = keras.optimizers.Adam(
    learning_rate=1e-4, beta_1=0.5, beta_2=0.9
)
discriminator_optimizer = keras.optimizers.Adam(
    learning_rate=1e-4, beta_1=0.5, beta_2=0.9
)

# Define the loss functions for the discriminator,
# which should be (fake_loss - real_loss).
# We will add the gradient penalty later to this loss function.
def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss

# Define the loss functions for the generator.
def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)

#%%
'''
COMPILE
'''
# Instantiate Discriminator
d_model = make_discriminator()
# Instantiate Generator
g_model = make_generator()
# Instantiate WaveGan
wavegan = WaveGAN(
    discriminator=d_model,
    generator=g_model
    )
# Compile the WaveGAN model.
wavegan.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
    g_loss_fn=generator_loss,
    d_loss_fn=discriminator_loss
    )