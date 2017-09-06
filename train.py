import os
import sys
import argparse
import tensorflow as tf
import numpy as np
import dbread as db
from model import Pix2Pix
import scipy.misc

parser = argparse.ArgumentParser(description='Easy Implementation of Pix2Pix')

# parameters
parser.add_argument('--train', type=str, default='filelist.txt')
parser.add_argument('--out_dir', type=str, default='./output')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--direction', type=str, default='AtoB')    # AtoB or BtoA


def normalize(im):
    return im * (2.0 / 255.0) - 1


def denormalize(im):
    return (im + 1.) / 2.


def split_images(img, direction):
    tmp = np.split(img, 2, axis=2)
    img_A = tmp[0]
    img_B = tmp[1]
    if direction == 'AtoB':
        return img_A, img_B
    elif direction == 'BtoA':
        return img_B, img_A
    else:
        sys.exit("'--direction' should be 'AtoB' or 'BtoA'")


# Function for save the generated result
def save_visualization(X, nh_nw, save_path='./vis/sample.jpg'):
    nh, nw = nh_nw
    h, w = X.shape[1], X.shape[2]
    img = np.zeros((h * nh, w * nw, 3))

    for n, x in enumerate(X):
        j = int(n / nw)
        i = int(n % nw)
        img[j * h:j * h + h, i * w:i * w + w, :] = x

    scipy.misc.imsave(save_path, img)


def main():
    global_epoch = tf.Variable(0, trainable=False, name='global_step')
    global_epoch_increase = tf.assign(global_epoch, tf.add(global_epoch, 1))

    args = parser.parse_args()
    direction = args.direction
    filelist_train = args.train
    result_dir = args.out_dir + '/result'
    ckpt_dir = args.out_dir + '/checkpoint'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    total_epoch = args.epochs
    batch_size = args.batch_size

    database = db.DBreader(filelist_train, batch_size=batch_size, labeled=False, resize=[256, 512])

    sess = tf.Session()
    model = Pix2Pix(sess, batch_size)

    saver = tf.train.Saver(tf.global_variables())

    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    total_batch = database.total_batch

    epoch = sess.run(global_epoch)
    while True:
        if epoch == total_epoch:
            break
        for step in range(total_batch):
            img_input, img_target = split_images(database.next_batch(), direction)
            img_target = normalize(img_target)
            img_input = normalize(img_input)

            loss_D = model.train_discrim(img_input, img_target)         # Train Discriminator and get the loss value
            loss_GAN, loss_L1 = model.train_gen(img_input, img_target)  # Train Generator and get the loss value

            if step % 100 == 0:
                print('Epoch: [', epoch, '/', total_epoch, '], ', 'Step: [', step, '/', total_batch, '], D_loss: ', loss_D, ', G_loss_GAN: ', loss_GAN, ', G_loss_L1: ', loss_L1)

            if step % 500 == 0:
                generated_samples = denormalize(model.sample_generator(img_input, batch_size=batch_size))
                img_target = denormalize(img_target)
                img_input = denormalize(img_input)

                img_for_vis = np.concatenate([img_input, generated_samples, img_target], axis=2)
                savepath = result_dir + '/output_' + 'EP' + str(epoch).zfill(3) + "_Batch" + str(step).zfill(6) + '.jpg'
                save_visualization(img_for_vis, (batch_size, 1), save_path=savepath)

        epoch = sess.run(global_epoch_increase)
        saver.save(sess, ckpt_dir + '/model_epoch'+str(epoch).zfill(3))


if __name__ == "__main__":
    main()
