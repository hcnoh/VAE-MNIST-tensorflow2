import tensorflow as tf


def encoder(k, name, units_list=[256, 256, 128]):
    encoder = tf.keras.Sequential(name=name)
    for i, units in enumerate(units_list):
        encoder.add(tf.keras.layers.Dense(units=units, activation=tf.nn.tanh, name="dense_%i" % i))
    encoder.add(tf.keras.layers.Dense(units=2*k, name="dense_final"))

    return encoder


def decoder(k, x_depth, name, units_list=[128, 256, 256]):
    decoder = tf.keras.Sequential(name=name)
    for i, units in enumerate(units_list):
        decoder.add(tf.keras.layers.Dense(units=units, activation=tf.nn.tanh, name="dense_%i" % i))
    decoder.add(tf.keras.layers.Dense(units=x_depth, activation=tf.nn.sigmoid, name="dense_final"))

    return decoder


class VAE(object):
    def __init__(self, k, x_depth):
        self.k = k
        self.x_depth = x_depth

        self.encoder = encoder(k=self.k, name="encoder")
        self.decoder = decoder(k=self.k, x_depth=self.x_depth, name="decoder")
    
    def encode(self, x, training):
        encoded = self.encoder(x, training=training)
        mu = encoded[:, :self.k]
        log_sigma = encoded[:, self.k:]

        return mu, log_sigma
    
    def decode(self, z, training):
        x = self.decoder(z, training=training)

        return x
    
    def reparam(self, eps, mu, log_sigma):
        sigma = tf.exp(log_sigma)
        z = tf.sqrt(sigma) * eps + mu

        return z
    
    def encoder_loss(self, mu, log_sigma):
        sigma = tf.exp(log_sigma)
        loss = (1/2) * (
            tf.reduce_sum(sigma, axis=-1, keepdims=True) + \
            tf.reduce_sum(mu**2, axis=-1, keepdims=True) - \
            self.k - \
            tf.reduce_sum(log_sigma, axis=-1, keepdims=True)
        )

        return loss
    
    def decoder_loss(self, x, f_z):
        loss = tf.reduce_sum((x - f_z)**2, axis=-1, keepdims=True)

        return loss