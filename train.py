import os
import pickle

import numpy as np
import tensorflow as tf

import config as conf

from data.mnist import Mnist
from models.vae import VAE


def main():
    model_spec_name = "%s-model-spec.json" % conf.MODEL_NAME
    model_rslt_name = "%s-results.pickle" % conf.MODEL_NAME

    model_save_path = os.path.join(conf.MODEL_SAVE_DIR, conf.MODEL_NAME)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    model_ckpt_path = os.path.join(model_save_path, "model-ckpt")
    model_spec_path = os.path.join(model_save_path, model_spec_name)
    model_rslt_path = os.path.join(model_save_path, model_rslt_name)

    loader = Mnist()

    features = np.vstack([loader.train_features, loader.test_features]).astype(np.float32)

    num_sets = loader.num_train_sets + loader.num_test_sets
    
    feature_depth = loader.feature_depth
    feature_shape = loader.feature_shape

    latent_depth = conf.LATENT_DEPTH

    batch_size = conf.BATCH_SIZE
    num_epochs = conf.NUM_EPOCHS

    model = VAE(latent_depth, feature_depth)

    opt = tf.keras.optimizers.Adam()

    @tf.function
    def train_step(x, eps):
        with tf.GradientTape() as tape:
            mu, log_sigma = model.encode(x, training=True)
            z = model.reparam(eps, mu, log_sigma)
            f_z = model.decode(z, training=True)

            encoder_loss = tf.reduce_mean(model.encoder_loss(mu, log_sigma))
            decoder_loss = tf.reduce_mean(model.decoder_loss(x, f_z))
            loss = encoder_loss + decoder_loss

            grads_loss = tape.gradient(
                target=loss, sources=model.encoder.trainable_variables+model.decoder.trainable_variables)
            opt.apply_gradients(
                zip(grads_loss, model.encoder.trainable_variables+model.decoder.trainable_variables)
            )

        return encoder_loss, decoder_loss, loss

    ckpt = tf.train.Checkpoint(encoder=model.encoder, decoder=model.decoder)

    steps_per_epoch = num_sets // batch_size
    train_steps = steps_per_epoch * num_epochs

    encoder_losses = []
    decoder_losses = []
    losses = []
    encoder_losses_epoch = []
    decoder_losses_epoch = []
    losses_epoch = []
    fs = []
    for i in range(1, train_steps+1):
        epoch = i // steps_per_epoch

        idxes = np.random.choice(num_sets, batch_size, replace=False)
        x_i = features[idxes]
        eps_i = np.random.normal(size=[batch_size, latent_depth]).astype(np.float32)

        encoder_loss_i, decoder_loss_i, loss_i = train_step(x_i, eps_i)
        
        encoder_losses.append(encoder_loss_i)
        decoder_losses.append(decoder_loss_i)
        losses.append(loss_i)

        if i % steps_per_epoch == 0:
            f_eps = model.decode(eps_i, training=False)

            encoder_loss_epoch = np.mean(encoder_losses[-steps_per_epoch:])
            decoder_loss_epoch = np.mean(decoder_losses[-steps_per_epoch:])
            loss_epoch = np.mean(losses[-steps_per_epoch:])

            print("Epoch: %i,  Encoder Loss: %f,  Decoder Loss: %f" % \
                (epoch, encoder_loss_epoch, decoder_loss_epoch)
            )

            encoder_losses_epoch.append(encoder_loss_epoch)
            decoder_losses_epoch.append(decoder_loss_epoch)
            losses_epoch.append(loss_epoch)

            fs.append(f_eps)
            
            ckpt.save(file_prefix=model_ckpt_path)

            with open(model_rslt_path, "wb") as f:
                pickle.dump((encoder_losses_epoch, decoder_losses_epoch, losses_epoch, fs), f)


if __name__ == "__main__":
    main()