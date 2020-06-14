import os
import argparse as arg
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow_addons as tfa

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.python.keras import backend as K

import create_dataset

class Generator(tf.keras.Model):
    def __init__(self):
        super().__init__()

        input_shape = (256, 256, 3)

        # ダウンサンプリング    
        self.pre = [
            kl.Conv2D(64, kernel_size=7, strides=1, 
                    padding="same", input_shape=input_shape),
            tfa.layers.InstanceNormalization(axis=-1),
            kl.Activation(tf.nn.relu),

            kl.Conv2D(128, kernel_size=3, strides=2, padding="same"),
            tfa.layers.InstanceNormalization(axis=-1),
            kl.Activation(tf.nn.relu),

            kl.Conv2D(256, kernel_size=3, strides=2, padding="same"),
            tfa.layers.InstanceNormalization(axis=-1),
            kl.Activation(tf.nn.relu)   
        ]

        # Residual Block
        self.res = [
            [
                Res_block(256) for _ in range(9)
            ]
        ]

        #
        self.t1 = kl.Conv2DTranspose(128, kernel_size=3, strides=2, padding="same", input_shape=(256, 256, 3))
        self.in1 = tfa.layers.InstanceNormalization(axis=-1)
        self.act1 = kl.Activation(tf.nn.relu)

        self.t2 = kl.Conv2DTranspose(64, kernel_size=3, strides=2, padding="same")
        self.in2 = tfa.layers.InstanceNormalization(axis=-1)
        self.act2 = kl.Activation(tf.nn.relu)

        self.conv1 =kl.Conv2D(3, kernel_size=7, strides=1, padding="same", activation="tanh")
        self.in3 = tfa.layers.InstanceNormalization(axis=-1)
        self.act3 = kl.Activation(tf.nn.tanh)

    # 順伝播
    def call(self, x):

        # Pre stage
        pre = x
        for layer in self.pre:
            pre = layer(pre)

        # Residual Block
        res = pre
        for layer in self.res:
            for l in layer:
                res = l(res)

        # 
        d1 = self.act1(self.in1(self.t1(res)))
        d2 = self.act2(self.in2(self.t2(d1)))
        d3 = self.act3(self.in3(self.conv1(d2)))

        return d3

# Pixel Shuffle
class Pixel_shuffer(tf.keras.Model):
    def __init__(self, out_ch):
        super().__init__()

        input_shape = (256, 256, 64)

        self.conv = kl.Conv2D(out_ch, kernel_size=3, strides=1,
                            padding="same", input_shape=input_shape)
        self.act = kl.Activation(tf.nn.relu)
    
    # forward proc
    def call(self, x):

        d1 = self.conv(x)
        d2 = self.act(tf.nn.depth_to_space(d1, 2))
        return d2


# Discriminator 
class Discriminator(tf.keras.Model):
    def __init__(self):
        super().__init__()

        input_shape = (256, 256, 3)


        self.conv1 = kl.Conv2D(64, kernel_size=4, strides=2,
                            padding="same", input_shape=input_shape)
        self.act1 = kl.Activation(tf.nn.relu)

        self.conv2 = kl.Conv2D(128, kernel_size=4, strides=2,
                            padding="same")
        self.in1 = tfa.layers.InstanceNormalization(axis=-1)
        self.act2 = kl.LeakyReLU()

        self.conv3 = kl.Conv2D(256, kernel_size=4, strides=2,
                            padding="same")
        self.in2 = tfa.layers.InstanceNormalization(axis=-1)
        self.act3 = kl.LeakyReLU()

        self.conv4 = kl.Conv2D(512, kernel_size=4, strides=2,
                            padding="same")
        self.in3 = tfa.layers.InstanceNormalization(axis=-1)
        self.act4 = kl.LeakyReLU()

        self.conv5 = kl.Conv2D(512, kernel_size=4, strides=1,
                            padding="same")
        self.in4 = tfa.layers.InstanceNormalization(axis=-1)
        self.act5 = kl.LeakyReLU()

        self.conv6 = kl.Conv2D(1, kernel_size=4, strides=1, padding="same")

    # forward proc
    def call(self, x):

        d1 = self.act1(self.conv1(x))
        d2 = self.act2(self.in1(self.conv2(d1)))
        d3 = self.act3(self.in2(self.conv3(d2)))
        d4 = self.act4(self.in3(self.conv4(d3)))
        d5 = self.act5(self.in4(self.conv5(d4)))
        
        d6 = self.conv6(d5)

        return d6


# Residual Block
class Res_block(tf.keras.Model):
    def __init__(self, ch):
        super().__init__()

        input_shape = (256, 256, 3)
        
        self.conv1 = kl.Conv2D(ch, kernel_size=3, strides=1,
                            padding="same", input_shape=input_shape)
        self.in1 = tfa.layers.InstanceNormalization(axis=-1)
        self.av1 = kl.Activation(tf.nn.relu)

        self.conv2 = kl.Conv2D(ch, kernel_size=3, strides=1,
                            padding="same")
        self.in2 = tfa.layers.InstanceNormalization(axis=-1)

        self.add = kl.Add()

    # forward proc
    def call(self, x):
       
        d1 = self.av1(self.in1(self.conv1(x)))
        d2 = self.in2(self.conv2(d1))

        return self.add([x, d2])

# Train
class trainer():
    def __init__(self, tr_X, tr_Y):

        output_shape = tf.keras.Input(shape=(256, 256, 3))
        self.L1norm_X = 0
        self.L1norm_Y = 0


        self.discriminatorX = Discriminator()
        self.discriminatorX.compile(optimizer=tf.keras.optimizers.Adam(),
                                    loss=tf.keras.losses.MeanSquaredError(),
                                    metrics=['accuracy'])

        self.discriminatorY = Discriminator()
        self.discriminatorY.compile(optimizer=tf.keras.optimizers.Adam(),
                                    loss=tf.keras.losses.MeanSquaredError(),
                                    metrics=['accuracy'])

        self.generatorX = Generator()
        self.generatorY = Generator()

        
        img_input_X = tf.keras.Input(shape=(256, 256, 3))
        img_input_Y = tf.keras.Input(shape=(256, 256, 3))
        gen_imgs_X = self.generatorX(img_input_X)
        gen_imgs_Y = self.generatorY(img_input_Y)
        
        re_imgs_X = self.generatorY(gen_imgs_X)
        re_imgs_Y = self.generatorX(gen_imgs_Y)

        self.discriminatorX.trainable = False
        label_X = self.discriminatorX(gen_imgs_X)
        self.discriminatorY.trainable = False
        label_Y = self.discriminatorY(gen_imgs_Y)

        self.cyclegan = tf.keras.Model(inputs=[img_input_X, img_input_Y], outputs=[label_X, label_Y, re_imgs_X, re_imgs_Y])
        self.cyclegan.compile(optimizer=tf.keras.optimizers.Adam(),
                        loss={"discriminator": tf.keras.losses.MeanSquaredError(), "discriminator_1" : tf.keras.losses.MeanSquaredError(),
                                "generator_1" : tf.keras.losses.MeanAbsoluteError(), "generator" : tf.keras.losses.MeanAbsoluteError()},
                        loss_weights={"discriminator" : 1, "discriminator_1" : 1, "generator_1" : 10, "generator" : 10})

    def train(self, tr_X, tr_Y, out_path, batch_size, iterations):

        h_batch = int(batch_size / 2)

        gan_valid = np.ones((h_batch, 1))

        for ite in range(iterations):

            idx = np.random.randint(0, tr_X.shape[0], h_batch)
            imgs_X = tr_X[idx]  
            gen_imgs_X = self.generatorX.predict(imgs_X)
            
            self.discriminatorX.trainable = True
            dx_loss_real = self.discriminatorX.train_on_batch(imgs_X, np.ones((h_batch, 16, 16)))
            dx_loss_fake = self.discriminatorX.train_on_batch(gen_imgs_X, np.zeros((h_batch, 16, 16)))
            dx_loss = 0.5 * np.add(dx_loss_real, dx_loss_fake)

            idx = np.random.randint(0, tr_Y.shape[0], h_batch)
            imgs_Y = tr_Y[idx]
            gen_imgs_Y = self.generatorY.predict(imgs_Y)
            
            self.discriminatorY.trainable = True
            dy_loss_real = self.discriminatorY.train_on_batch(imgs_Y, np.ones((h_batch, 16, 16)))
            dy_loss_fake = self.discriminatorY.train_on_batch(gen_imgs_Y, np.zeros((h_batch, 16, 16)))
            dy_loss = 0.5 * np.add(dy_loss_real, dy_loss_fake)

            idx = np.random.randint(0, tr_X.shape[0], h_batch)
            imgs_X = tr_X[idx]
            idx = np.random.randint(0, tr_Y.shape[0], h_batch)
            imgs_Y = tr_Y[idx]
            
            gen_imgs_X = self.generatorX.predict(imgs_X)
            re_imgs_X = self.generatorY.predict(gen_imgs_X)

            gen_imgs_Y = self.generatorY.predict(imgs_Y)
            re_imgs_Y = self.generatorX.predict(gen_imgs_Y)

            self.discriminatorX.trainable = False
            self.discriminatorY.trainable = False
            self.g_loss = self.cyclegan.train_on_batch([imgs_X, imgs_Y], [gan_valid, gan_valid, imgs_X, imgs_Y])

            np.set_printoptions(precision=2)
            print("iteration " + str(ite) + " [DX loss: " + str(dx_loss) + " DY loss: " + str(dy_loss) + "]" + "[G loss: " + str(self.g_loss) + "]")

            if (ite+1) % 50 == 0:
                self.save_imgs(ite, tr_X) 

    def save_imgs(self, iteration, tr_X):

  
        img = tr_X[2]
        img = img[None, :, :, :]

        gen_imgs = self.generatorX.predict(img)

        

        gen_imgs = gen_imgs * 127.5 + 127.5
        gen_imgs = np.reshape(gen_imgs, (256, 256, 3))


        print(gen_imgs)
        img = Image.fromarray(np.uint8(gen_imgs))
        img.show()
        
           

    def cycle_consistencyX2Y(self, X, X2Y):
        
        LAMBDA = 10

        re_X = self.generatorY(X2Y)

        self.L1norm_X = K.sum(K.abs(re_X - X))

        fully_objective = LAMBDA * (self.L1norm_X + self.L1norm_Y)

       
        return fully_objective

    def cycle_consistencyY2X(self, Y, Y2X):
        LAMBDA = 10

        re_Y = self.generatorX(Y2X)
        self.L1norm_Y = K.sum(K.abs(re_Y - Y))


        fully_objective = LAMBDA * (self.L1norm_Y + self.L1norm_X)

        return fully_objective


def main():

    parser = arg.ArgumentParser(description='CycleGAN')
    parser.add_argument('--data_dir', '-d', type=str, default=None,
                        help='画像フォルダパスの指定(未指定ならエラー)')
    parser.add_argument('--out', '-o', type=str,
                        default=os.path.dirname(os.path.abspath(__file__)),
                        help='パラメータの保存先指定(デフォルト値=./test.h5')
    parser.add_argument('--batch_size', '-b', type=int, default=2,
                        help='ミニバッチサイズの指定(デフォルト値=64)')
    parser.add_argument('--iter', '-i', type=int, default=2000,
                        help='学習回数の指定(デフォルト値=10)')
    args = parser.parse_args()

    if args.data_dir == None:
        print("\nException: Folder not specified.\n")
        sys.exit()
    if os.path.exists(args.data_dir) != True:
        print("\nException: Folder \"{}\" is not found.\n".format(args.data_dir))
        sys.exit()

    print("=== Setting information ===")
    print("# Images folder: {}".format(os.path.abspath(args.data_dir)))
    print("# Output folder: {}".format(args.out))
    print("# Minibatch-size: {}".format(args.batch_size))
    print("# Iteration: {}".format(args.iter))
    print("===========================")

    os.makedirs(args.out, exist_ok=True)

    tr_X, tr_Y = create_dataset.create_dataset(args.data_dir)

    Trainer = trainer(tr_X, tr_Y)
    Trainer.train(tr_X, tr_Y, out_path=args.out, batch_size=args.batch_size, iterations=args.iter)

if __name__ == '__main__':
    main()