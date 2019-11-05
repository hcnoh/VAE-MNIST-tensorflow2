# Variational Autoencoder (VAE) Implementation for Generating MNIST Images in TensorFlow2

This repository is for the TensorFlow2 implementation for VAE. This repository provides the training module and Jupyter notebook for testing a generation of the trained models. MNIST dataset was used for this repository.

![](/assets/img/README/README_2019-10-17-19-44-46.png)

## Install Dependencies
1. Install Python 3.5.2.
2. Install TensorFlow ver 2.0.0. If you can use a GPU machine, install the GPU version of TensorFlow, or just install the CPU version of it.
3. Install Python packages(Requirements). You can install them simply by using following bash command.

    ```bash
    $ pip install -r requirements
    ```

    You can use `Virtualenv`, so that the packages for requirements can be managed easily. If your machine have `Virtualenv` package, the following bash command would be useful.

    ```bash
    $ virtualenv vae-mnist-tf2-venv
    $ source ./vae-mnist-tf2-venv/bin/activate
    $ pip install -r requirements.txt
    ```

## Training
*Note: MNIST-in-CSV dataset was used for this repository. But you can use MNIST dataset module in TensorFlow. But the following process is for just using MNIST-in-CSV dataset.*

1. **Download the dataset.**

    The link for MNIST-in-CSV: [https://www.kaggle.com/oddrationale/mnist-in-csv](https://www.kaggle.com/oddrationale/mnist-in-csv)

2. **Unpack the dataset.**

    You can check that there are two csv files named `mnist_train.csv` and `mnist_test.csv`.

3. **Modify the path for dataset in `config.py`.**

4. **Modify the path for directory for saving model checkpoint.**

5. **Execute training process by `train.py`.**

## Checking Results and Testing Generation
The Jupyter notebook for checking results and testing the image generation is provided. Please check `result_plot.ipynb`.

## Results

1. **Ploting the Encoder and Decoder Losses**

    ![](/assets/img/README/README_2019-10-17-19-36-52.png)

2. **Image Generation Results**

    ![](/assets/img/README/README_2019-10-17-19-40-04.png)

3. **Plotting the Latent Distribution**

    ![](/assets/img/README/README_2019-10-17-19-41-35.png)

## References
- VAE Tutorial: [Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908)
- VAE: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)