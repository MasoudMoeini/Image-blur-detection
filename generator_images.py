{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "image_blur_detection.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "authorship_tag": "ABX9TyMHMeSdJd7YQDHVHrAupTpR",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/MasoudMoeini/Image-blur-detection/blob/main/generator_images.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 62,
      "metadata": {
        "id": "p-5xmsZ5RQes",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "d63cbb18-3821-49c2-81d1-77223cf775e7"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "The tensorboard extension is already loaded. To reload it, use:\n",
            "  %reload_ext tensorboard\n"
          ]
        }
      ],
      "source": [
        "%load_ext tensorboard\n",
        "import tensorflow.compat.v1 as tf\n",
        "tf.disable_v2_behavior()\n",
        "import numpy as np\n",
        "import matplotlib . pyplot as plt\n",
        "from tensorflow.keras import layers, losses\n",
        "# Base CNN\n",
        "x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3]) # input image\n",
        "y = tf.placeholder(tf.float32, shape=[None, 224, 224, 3]) # Blur area detected"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#!unzip -qq BlurDatasetResultShi.zip\n",
        "!unzip -qq ccv_data.zip"
      ],
      "metadata": {
        "id": "wJfwkZX9TC3O"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "rm -rf ./logs/"
      ],
      "metadata": {
        "id": "8pmv42pPJ4SW"
      },
      "execution_count": 63,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Conv1\n",
        "# Input Tensor Shape: [batch_size, 224, 224, 3]\n",
        "# Output Tensor Shape: [batch_size, 224, 228, 32]\n",
        "conv1 = tf.layers.conv2d(x, filters=32, kernel_size=[3,3], padding=\"same\", activation=tf.nn.relu)\n",
        "\n",
        "# conv2\n",
        "# Input Tensor Shape: [batch_size, 224, 224, 32]\n",
        "# Output Tensor Shape: [batch_size, 224, 228, 32]\n",
        "conv2 = tf.layers.conv2d(conv1, filters=32, kernel_size=[3,3], padding=\"same\", activation=tf.nn.relu)\n",
        "\n",
        "# pool1\n",
        "# Input Tensor Shape: [batch_size, 224, 224, 32]\n",
        "# Output Tensor Shape: [batch_size, 112, 112, 32]\n",
        "pool1 = tf.layers.max_pooling2d(conv2, pool_size=[2,2], strides=2, padding=\"same\")\n",
        "\n",
        "#conv3\n",
        "# Input Tensor Shape: [batch_size, 112, 112, 32]\n",
        "# Output Tensor Shape: [batch_size, 112, 112, 64]\n",
        "conv3 = tf.layers.conv2d(conv2, filters=64, kernel_size=[3,3], padding=\"same\", activation=tf.nn.relu)\n",
        "\n",
        "#conv4\n",
        "# Input Tensor Shape: [batch_size, 112, 112, 64]\n",
        "# Output Tensor Shape: [batch_size, 112, 112, 64]\n",
        "conv4 = tf.layers.conv2d(conv3, filters=64, kernel_size=[3,3], padding=\"same\", activation=tf.nn.relu)\n",
        "\n",
        "#pool2\n",
        "# Input Tensor Shape: [batch_size, 112, 112, 64]\n",
        "# Output Tensor Shape: [batch_size, 56, 56, 64]\n",
        "pool2 = tf.layers.max_pooling2d(conv4, pool_size=[2,2], strides=2, padding=\"same\")\n",
        "\n",
        "#conv5\n",
        "# Input Tensor Shape: [batch_size, 56, 56, 64]\n",
        "# Output Tensor Shape: [batch_size, 56, 56, 128]\n",
        "conv5 = tf.layers.conv2d(pool2, filters=128, kernel_size=[3,3], padding=\"same\", activation=tf.nn.relu)\n",
        "\n",
        "#conv6\n",
        "# Input Tensor Shape: [batch_size, 56, 56, 128]\n",
        "# Output Tensor Shape: [batch_size, 56, 56, 128]\n",
        "conv6 = tf.layers.conv2d(conv5, filters=128, kernel_size=[3,3], padding=\"same\", activation=tf.nn.relu)\n",
        "\n",
        "\n",
        "#pool3\n",
        "# Input Tensor Shape: [batch_size, 56, 56, 128]\n",
        "# Output Tensor Shape: [batch_size, 28, 28, 128]\n",
        "pool3 = tf.layers.max_pooling2d(conv6, pool_size=[2,2], strides=2, padding=\"same\")\n",
        "\n",
        "#conv7\n",
        "# Input Tensor Shape: [batch_size, 28, 28, 128]\n",
        "# Output Tensor Shape: [batch_size, 28, 28, 256]\n",
        "conv7 = tf.layers.conv2d(pool3, filters=256, kernel_size=[3,3], padding=\"same\", activation=tf.nn.relu)\n",
        "#conv8\n",
        "# Input Tensor Shape: [batch_size, 28, 28, 256]\n",
        "# Output Tensor Shape: [batch_size, 28, 28, 256]\n",
        "conv8 = tf.layers.conv2d(conv7, filters=128, kernel_size=[3,3], padding=\"same\", activation=tf.nn.relu)\n",
        "\n",
        "#pool4\n",
        "# Input Tensor Shape: [batch_size, 28, 28, 256]\n",
        "# Output Tensor Shape: [batch_size, 14, 14, 256]\n",
        "\n",
        "pool4 = tf.layers.max_pooling2d(conv8, pool_size=[2,2], strides=2, padding=\"same\")\n",
        "#conv9\n",
        "# Input Tensor Shape: [batch_size, 14, 14, 256]\n",
        "# Output Tensor Shape: [batch_size, 14, 14, 512]\n",
        "conv9 = tf.layers.conv2d(pool4, filters=512, kernel_size=[3,3], padding=\"same\", activation=tf.nn.relu)\n",
        "\n",
        "#pool5\n",
        "# Input Tensor Shape: [batch_size, 14, 14, 512]\n",
        "# Output Tensor Shape: [batch_size, 7, 7, 512]\n",
        "pool5 = tf.layers.max_pooling2d(conv9, pool_size=[2,2], strides=2, padding=\"same\")\n",
        "\n",
        "#dim = int(np.prod(pool5.get_shape()[1:])) #7*7*512\n",
        "#fcl = tf.reshape(pool5, shape=[-1, dim], name ='fc1')#[batch_size,7*7*512]\n",
        "# decoder\n",
        "\n",
        "# Input Tensor Shape: [batch_size, 7, 7, 512]\n",
        "# Output Tensor Shape: [batch_size, 14, 14, 512]\n",
        "net=tf.layers.conv2d_transpose(pool5,512,[3, 3],strides = 2,padding='SAME')\n",
        "\n",
        "# Input Tensor Shape: [batch_size, 7, 7, 512]\n",
        "# Output Tensor Shape: [batch_size, 28, 28, 256]\n",
        "net=tf.layers.conv2d_transpose(net,256,[3, 3],strides = 2,padding='SAME')\n",
        "\n",
        "# Input Tensor Shape: [batch_size, 28, 28, 128]\n",
        "# Output Tensor Shape: [batch_size, 224, 224, 128]\n",
        "net=tf.layers.conv2d_transpose(net,128,[3, 3],strides = 4,padding='SAME')\n",
        "\n",
        "# Input Tensor Shape: [batch_size, 224, 224, 128]\n",
        "# Output Tensor Shape: [batch_size, 224, 224, 64]\n",
        "net=tf.layers.conv2d_transpose(net,64,[3, 3],strides = 1,padding='SAME')\n",
        "\n",
        "# Input Tensor Shape: [batch_size, 224, 224, 64]\n",
        "# Output Tensor Shape: [batch_size, 224, 224, 32]\n",
        "net=tf.layers.conv2d_transpose(net,32,[3, 3],strides = 1, padding='SAME', activation = tf.nn.tanh)\n",
        "\n",
        "# Input Tensor Shape: [batch_size, 224, 224, 32]\n",
        "# Output Tensor Shape: [batch_size, 224, 224, 3]\n",
        "net=tf.layers.conv2d_transpose(net,3,[3, 3],strides = 1, padding='SAME', activation = tf.nn.tanh)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7BbR2mvcKA5S",
        "outputId": "cead0e2f-003b-4127-ecf8-0bd42f369b58"
      },
      "execution_count": 64,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:4: UserWarning: `tf.layers.conv2d` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv2D` instead.\n",
            "  after removing the cwd from sys.path.\n",
            "/usr/local/lib/python3.7/dist-packages/keras/legacy_tf_layers/convolutional.py:575: UserWarning: `layer.apply` is deprecated and will be removed in a future version. Please use `layer.__call__` method instead.\n",
            "  return layer.apply(inputs)\n",
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:9: UserWarning: `tf.layers.conv2d` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv2D` instead.\n",
            "  if __name__ == '__main__':\n",
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:14: UserWarning: `tf.layers.max_pooling2d` is deprecated and will be removed in a future version. Please use `tf.keras.layers.MaxPooling2D` instead.\n",
            "  \n",
            "/usr/local/lib/python3.7/dist-packages/keras/legacy_tf_layers/pooling.py:600: UserWarning: `layer.apply` is deprecated and will be removed in a future version. Please use `layer.__call__` method instead.\n",
            "  return layer.apply(inputs)\n",
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:19: UserWarning: `tf.layers.conv2d` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv2D` instead.\n",
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:24: UserWarning: `tf.layers.conv2d` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv2D` instead.\n",
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:29: UserWarning: `tf.layers.max_pooling2d` is deprecated and will be removed in a future version. Please use `tf.keras.layers.MaxPooling2D` instead.\n",
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:34: UserWarning: `tf.layers.conv2d` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv2D` instead.\n",
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:39: UserWarning: `tf.layers.conv2d` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv2D` instead.\n",
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:45: UserWarning: `tf.layers.max_pooling2d` is deprecated and will be removed in a future version. Please use `tf.keras.layers.MaxPooling2D` instead.\n",
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:50: UserWarning: `tf.layers.conv2d` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv2D` instead.\n",
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:54: UserWarning: `tf.layers.conv2d` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv2D` instead.\n",
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:60: UserWarning: `tf.layers.max_pooling2d` is deprecated and will be removed in a future version. Please use `tf.keras.layers.MaxPooling2D` instead.\n",
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:64: UserWarning: `tf.layers.conv2d` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv2D` instead.\n",
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:69: UserWarning: `tf.layers.max_pooling2d` is deprecated and will be removed in a future version. Please use `tf.keras.layers.MaxPooling2D` instead.\n",
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:77: UserWarning: `tf.layers.conv2d_transpose` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv2DTranspose` instead.\n",
            "/usr/local/lib/python3.7/dist-packages/keras/legacy_tf_layers/convolutional.py:1736: UserWarning: `layer.apply` is deprecated and will be removed in a future version. Please use `layer.__call__` method instead.\n",
            "  return layer.apply(inputs)\n",
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:81: UserWarning: `tf.layers.conv2d_transpose` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv2DTranspose` instead.\n",
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:85: UserWarning: `tf.layers.conv2d_transpose` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv2DTranspose` instead.\n",
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:89: UserWarning: `tf.layers.conv2d_transpose` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv2DTranspose` instead.\n",
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:93: UserWarning: `tf.layers.conv2d_transpose` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv2DTranspose` instead.\n",
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:97: UserWarning: `tf.layers.conv2d_transpose` is deprecated and will be removed in a future version. Please Use `tf.keras.layers.Conv2DTranspose` instead.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "## Optimize\n",
        "n_epochs = 100\n",
        "learning_rate = 0.01\n",
        "loss = tf.reduce_mean(tf.square(net - y))\n",
        "optimizer = tf.train.AdamOptimizer(learning_rate)\n",
        "train  = optimizer.minimize(loss)"
      ],
      "metadata": {
        "id": "FBu6qF1NS4I6"
      },
      "execution_count": 65,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from generator_images import load_images_from_folder\n",
        "train_input_path= 'ccv_data/train/images/'\n",
        "train_target_path= 'ccv_data/train/blur/'\n",
        "\n",
        "train_input_images = load_images_from_folder(train_input_path)\n",
        "train_target_images = load_images_from_folder(train_target_path)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "JQ44wJapToTh",
        "outputId": "a7a2677a-04b3-49b9-a70d-4836e5bc6bed"
      },
      "execution_count": 66,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            " 800 from ccv_data/train/images/ successfully uploaded\n",
            " 800 from ccv_data/train/blur/ successfully uploaded\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "BATCH_SIZE = 10\n",
        "import cv2\n",
        "img = np.empty((BATCH_SIZE, 224, 224, 3), np.float32)\n",
        "l = list(enumerate(train_input_images[0:BATCH_SIZE]))\n",
        "len(l)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "J5DNHJrsaCRv",
        "outputId": "8dba3b4b-2734-4e9c-98d1-bef47708d4d3"
      },
      "execution_count": 50,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "10"
            ]
          },
          "metadata": {},
          "execution_count": 50
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "## Define the Xavier initialization\n",
        "#xav_init =  tf.contrib.layers.xavier_initializer()\n",
        "## Define the L2 regularizer\n",
        "#l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)"
      ],
      "metadata": {
        "id": "v4Qd34hsP3iI"
      },
      "execution_count": 16,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "BATCH_SIZE = 10\n",
        "### Number of batches :  length dataset / batch size\n",
        "n_batches = len(train_input_images) // BATCH_SIZE\n",
        "## Set params\n",
        "n_epochs = 100\n",
        "learning_rate = 0.01"
      ],
      "metadata": {
        "id": "W4gKOgf6NBRO"
      },
      "execution_count": 67,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "init=tf.global_variables_initializer()\n",
        "averageloss=[]\n",
        "with tf.Session() as sess:\n",
        "    sess.run(init)\n",
        "    for epoch in range(n_epochs):\n",
        "        print('epoch ',epoch)\n",
        "        DLOSS=[]\n",
        "        for i in range(n_batches):\n",
        "            #m=i*batch_size\n",
        "            #batch_images=train_input_images[i*BATCH_SIZE : (i+1)*BATCH_SIZE]\n",
        "            #batch_target=train_target_images [i*BATCH_SIZE : (i+1)*BATCH_SIZE]\n",
        "            batch_images = train_input_images[i*BATCH_SIZE : (i+1)*BATCH_SIZE].reshape(BATCH_SIZE,244,244,3)\n",
        "            batch_target = train_target_images [i*BATCH_SIZE : (i+1)*BATCH_SIZE].reshape(BATCH_SIZE,244,244,3)\n",
        "            batch_loss = sess.run([loss, train], feed_dict={x: batch_images, y: batch_target})\n",
        "            DLOSS.append(batch_loss)\n",
        "            print('epoch {} batch number {}    batch loss: {}'.format(epoch,i,batch_loss))\n",
        "        MeanDloss=np.mean(DLOSS)\n",
        "        averageloss.append(np.mean(DLOSS))\n",
        "        print(' Average batches loss: {} '.format(MeanDloss))\n",
        "       "
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 374
        },
        "id": "sPqqHoDRP0e_",
        "outputId": "6b15f380-25d5-4616-95cd-06c2b5389779"
      },
      "execution_count": 68,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "epoch  0\n"
          ]
        },
        {
          "output_type": "error",
          "ename": "AttributeError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-68-5e9b37c23ec0>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m     10\u001b[0m             \u001b[0;31m#batch_images=train_input_images[i*BATCH_SIZE : (i+1)*BATCH_SIZE]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     11\u001b[0m             \u001b[0;31m#batch_target=train_target_images [i*BATCH_SIZE : (i+1)*BATCH_SIZE]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 12\u001b[0;31m             \u001b[0mbatch_images\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtrain_input_images\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mi\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0mBATCH_SIZE\u001b[0m \u001b[0;34m:\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0mi\u001b[0m\u001b[0;34m+\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0mBATCH_SIZE\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mreshape\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mBATCH_SIZE\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m244\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m244\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m3\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     13\u001b[0m             \u001b[0mbatch_target\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtrain_target_images\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0mi\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0mBATCH_SIZE\u001b[0m \u001b[0;34m:\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0mi\u001b[0m\u001b[0;34m+\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0mBATCH_SIZE\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mreshape\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mBATCH_SIZE\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m244\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m244\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m3\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     14\u001b[0m             \u001b[0mbatch_loss\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0msess\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mrun\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mloss\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtrain\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mfeed_dict\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m{\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0mbatch_images\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0mbatch_target\u001b[0m\u001b[0;34m}\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mAttributeError\u001b[0m: 'list' object has no attribute 'reshape'"
          ]
        }
      ]
    }
  ]
}