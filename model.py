from __future__ import print_function

import os
import time
import random
import math

from PIL import Image
import tensorflow as tf
import numpy as np

from utils import *

def concat(layers):
    return tf.concat(layers, axis=3)

def DecomNet(input_im, layer_num, channel=64, kernel_size=3):
    input_max = tf.reduce_max(input_im, axis=3, keepdims=True)
    input_im = concat([input_max, input_im])
    with tf.variable_scope('DecomNet', reuse=tf.AUTO_REUSE):
        conv = tf.layers.conv2d(input_im, channel, kernel_size * 3, padding='same', activation=None, name="shallow_feature_extraction")
        for idx in range(layer_num):
            conv = tf.layers.conv2d(conv, channel, kernel_size, padding='same', activation=tf.nn.relu, name='activated_layer_%d' % idx)
        conv = tf.layers.conv2d(conv, 4, kernel_size, padding='same', activation=None, name='recon_layer')

    R = tf.sigmoid(conv[:,:,:,0:3])
    L = tf.sigmoid(conv[:,:,:,3:4])

    return R, L

def RelightNet(input_L, input_R, channel=64, kernel_size=3):
    input_im = concat([input_R, input_L])
    with tf.variable_scope('RelightNet'):
        conv0 = tf.layers.conv2d(input_im, channel, kernel_size, padding='same', activation=None)
        conv1 = tf.layers.conv2d(conv0, channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1, channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(conv2, channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)
        
        up1 = tf.image.resize_nearest_neighbor(conv3, (tf.shape(conv2)[1], tf.shape(conv2)[2]))
        deconv1 = tf.layers.conv2d(up1, channel, kernel_size, padding='same', activation=tf.nn.relu) + conv2
        up2 = tf.image.resize_nearest_neighbor(deconv1, (tf.shape(conv1)[1], tf.shape(conv1)[2]))
        deconv2= tf.layers.conv2d(up2, channel, kernel_size, padding='same', activation=tf.nn.relu) + conv1
        up3 = tf.image.resize_nearest_neighbor(deconv2, (tf.shape(conv0)[1], tf.shape(conv0)[2]))
        deconv3 = tf.layers.conv2d(up3, channel, kernel_size, padding='same', activation=tf.nn.relu) + conv0
        
        deconv1_resize = tf.image.resize_nearest_neighbor(deconv1, (tf.shape(deconv3)[1], tf.shape(deconv3)[2]))
        deconv2_resize = tf.image.resize_nearest_neighbor(deconv2, (tf.shape(deconv3)[1], tf.shape(deconv3)[2]))
        feature_gather = concat([deconv1_resize, deconv2_resize, deconv3])
        feature_fusion = tf.layers.conv2d(feature_gather, channel, 1, padding='same', activation=None)
        output = tf.layers.conv2d(feature_fusion, 1, 3, padding='same', activation=None)
    return output

class lowlight_enhance(object):
    def __init__(self, sess):
        self.sess = sess
        self.DecomNet_layer_num = 5

        # build the model
        self.input_low = tf.placeholder(tf.float32, [None, None, None, 3], name='input_low')
        self.input_high = tf.placeholder(tf.float32, [None, None, None, 3], name='input_high')

        [R_low, I_low] = DecomNet(self.input_low, layer_num=self.DecomNet_layer_num)
        [R_high, I_high] = DecomNet(self.input_high, layer_num=self.DecomNet_layer_num)
        
        I_delta = RelightNet(I_low, R_low)

        I_low_3 = concat([I_low, I_low, I_low])
        I_high_3 = concat([I_high, I_high, I_high])
        I_delta_3 = concat([I_delta, I_delta, I_delta])

        self.output_R_low = R_low
        self.output_I_low = I_low_3
        self.output_I_delta = I_delta_3
        self.output_S = R_low * I_delta_3

        # loss
        self.recon_loss_low = tf.reduce_mean(tf.abs(R_low * I_low_3 -  self.input_low))
        self.recon_loss_high = tf.reduce_mean(tf.abs(R_high * I_high_3 - self.input_high))
        self.recon_loss_mutal_low = tf.reduce_mean(tf.abs(R_high * I_low_3 - self.input_low))
        self.recon_loss_mutal_high = tf.reduce_mean(tf.abs(R_low * I_high_3 - self.input_high))
        self.equal_R_loss = tf.reduce_mean(tf.abs(R_low - R_high))
        self.relight_loss = tf.reduce_mean(tf.abs(R_low * I_delta_3 - self.input_high))

        self.Ismooth_loss_low = self.smooth(I_low, R_low)
        self.Ismooth_loss_high = self.smooth(I_high, R_high)
        self.Ismooth_loss_delta = self.smooth(I_delta, R_low)

        self.loss_Decom = self.recon_loss_low + self.recon_loss_high + 0.001 * self.recon_loss_mutal_low + 0.001 * self.recon_loss_mutal_high + 0.1 * self.Ismooth_loss_low + 0.1 * self.Ismooth_loss_high + 0.01 * self.equal_R_loss
        self.loss_Relight = self.relight_loss + 3 * self.Ismooth_loss_delta

        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')

        self.var_Decom = [var for var in tf.trainable_variables() if 'DecomNet' in var.name]
        self.var_Relight = [var for var in tf.trainable_variables() if 'RelightNet' in var.name]

        self.train_op_Decom = optimizer.minimize(self.loss_Decom, var_list = self.var_Decom)
        self.train_op_Relight = optimizer.minimize(self.loss_Relight, var_list = self.var_Relight)

        self.sess.run(tf.global_variables_initializer())

        self.saver_Decom = tf.train.Saver(var_list = self.var_Decom)
        self.saver_Relight = tf.train.Saver(var_list = self.var_Relight)
        # Best and last checkpoints need independent retention histories.
        self.best_saver_Decom = tf.train.Saver(var_list=self.var_Decom, max_to_keep=1, name='save_best_Decom')
        self.best_saver_Relight = tf.train.Saver(var_list=self.var_Relight, max_to_keep=1, name='save_best_Relight')

        print("[*] Initialize model successfully...")

    def gradient(self, input_tensor, direction):
        self.smooth_kernel_x = tf.reshape(tf.constant([[0, 0], [-1, 1]], tf.float32), [2, 2, 1, 1])
        self.smooth_kernel_y = tf.transpose(self.smooth_kernel_x, [1, 0, 2, 3])

        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        return tf.abs(tf.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'))

    def ave_gradient(self, input_tensor, direction):
        return tf.layers.average_pooling2d(self.gradient(input_tensor, direction), pool_size=3, strides=1, padding='SAME')

    def smooth(self, input_I, input_R):
        input_R = tf.image.rgb_to_grayscale(input_R)
        return tf.reduce_mean(self.gradient(input_I, "x") * tf.exp(-10 * self.ave_gradient(input_R, "x")) + self.gradient(input_I, "y") * tf.exp(-10 * self.ave_gradient(input_R, "y")))

    def evaluate(self, epoch_num, sample_low_data, sample_dir, train_phase):
        print("[*] Saving training samples for phase %s / epoch %d..." % (train_phase, epoch_num))

        for idx in range(len(sample_low_data)):
            input_low_sample = np.expand_dims(sample_low_data[idx], axis=0)

            if train_phase == "Decom":
                result_1, result_2 = self.sess.run([self.output_R_low, self.output_I_low], feed_dict={self.input_low: input_low_sample})
            if train_phase == "Relight":
                result_1, result_2 = self.sess.run([self.output_S, self.output_I_delta], feed_dict={self.input_low: input_low_sample})

            save_images(os.path.join(sample_dir, 'train_%s_%d_%d.png' % (train_phase, idx + 1, epoch_num)), result_1, result_2)

    def train(self, train_low_data, train_high_data, sample_low_data, batch_size, patch_size, epoch, lr, sample_dir, ckpt_dir, eval_every_epoch, train_phase, resume=False, loss_callback=None, best_ckpt_dir=None, epoch_callback=None, best_mode='min', initial_best_score=None):
        if len(train_low_data) != len(train_high_data):
            raise ValueError("Training low/high image counts must match.")
        if batch_size <= 0 or patch_size <= 0 or epoch <= 0 or eval_every_epoch <= 0:
            raise ValueError("Batch size, patch size, epochs and evaluation interval must be positive.")
        numBatch = len(train_low_data) // batch_size
        if numBatch == 0:
            raise ValueError("Training needs at least batch_size paired images.")
        if len(lr) < epoch:
            raise ValueError("The learning-rate schedule must cover every training epoch.")
        if loss_callback is not None and not callable(loss_callback):
            raise ValueError("loss_callback must be callable.")
        best_enabled = best_ckpt_dir is not None or epoch_callback is not None
        if best_enabled and (not best_ckpt_dir or not callable(epoch_callback)):
            raise ValueError("Best-model selection requires both best_ckpt_dir and an epoch_callback.")
        if best_mode not in ('min', 'max'):
            raise ValueError("best_mode must be 'min' or 'max'.")
        best_score = None if initial_best_score is None else float(initial_best_score)
        if best_score is not None and not math.isfinite(best_score):
            raise ValueError("initial_best_score must be finite.")
        for low_image, high_image in zip(train_low_data, train_high_data):
            if low_image.shape != high_image.shape:
                raise ValueError("Each low/high training image pair must have the same shape.")
            if low_image.ndim != 3 or low_image.shape[2] != 3:
                raise ValueError("Training images must have three RGB channels.")
            if min(low_image.shape[:2]) < patch_size:
                raise ValueError("Training images must be at least patch_size in both dimensions.")

        if train_phase == "Decom":
            train_op = self.train_op_Decom
            train_loss = self.loss_Decom
            saver = self.saver_Decom
            best_saver = self.best_saver_Decom if best_enabled else None
        elif train_phase == "Relight":
            train_op = self.train_op_Relight
            train_loss = self.loss_Relight
            saver = self.saver_Relight
            best_saver = self.best_saver_Relight if best_enabled else None
        else:
            raise ValueError("Unknown training phase: %s" % train_phase)

        if resume:
            load_model_status, global_step = self.load(saver, ckpt_dir)
            if not load_model_status:
                raise FileNotFoundError("Cannot resume without a checkpoint in %s" % ckpt_dir)
            if global_step is None:
                raise ValueError("The checkpoint filename must end with a numeric training step to resume.")
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Fresh training: checkpoint restore disabled.")

        if start_epoch > epoch:
            raise ValueError("The resumed checkpoint is beyond the requested total epochs.")

        print("[*] Start training for phase %s, with start epoch %d start iter %d : " % (train_phase, start_epoch, iter_num))

        start_time = time.time()
        image_id = (start_step * batch_size) % len(train_low_data)
        last_checkpoint = None
        last_saved_step = None

        for epoch_index in range(start_epoch, epoch):
            epoch_start_step = start_step if epoch_index == start_epoch else 0
            for batch_id in range(epoch_start_step, numBatch):
                # generate data for a batch
                batch_input_low = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
                batch_input_high = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
                for patch_id in range(batch_size):
                    h, w, _ = train_low_data[image_id].shape
                    x = random.randint(0, h - patch_size)
                    y = random.randint(0, w - patch_size)
            
                    rand_mode = random.randint(0, 7)
                    batch_input_low[patch_id, :, :, :] = data_augmentation(train_low_data[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode)
                    batch_input_high[patch_id, :, :, :] = data_augmentation(train_high_data[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode)
                    
                    image_id = (image_id + 1) % len(train_low_data)
                    if image_id == 0:
                        tmp = list(zip(train_low_data, train_high_data))
                        random.shuffle(tmp)
                        train_low_data, train_high_data  = zip(*tmp)

                # train
                _, loss = self.sess.run([train_op, train_loss], feed_dict={self.input_low: batch_input_low, \
                                                                           self.input_high: batch_input_high, \
                                                                           self.lr: lr[epoch_index]})

                iter_num += 1
                elapsed_seconds = time.time() - start_time
                print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
                      % (train_phase, epoch_index + 1, batch_id + 1, numBatch, elapsed_seconds, loss))
                if loss_callback is not None:
                    loss_callback({
                        "phase": train_phase,
                        "epoch": int(epoch_index + 1),
                        "batch": int(batch_id + 1),
                        "total_batches": int(numBatch),
                        "global_step": int(iter_num),
                        "loss": float(loss),
                        "learning_rate": float(lr[epoch_index]),
                        "elapsed_seconds": float(elapsed_seconds),
                    })

            if best_enabled:
                score = float(epoch_callback(train_phase, epoch_index + 1))
                if not math.isfinite(score):
                    raise ValueError("Epoch callback returned a non-finite score for %s epoch %d." % (train_phase, epoch_index + 1))
                improved = (best_score is None or
                            (best_mode == 'min' and score < best_score) or
                            (best_mode == 'max' and score > best_score))
                if improved:
                    self.save(best_saver, iter_num, best_ckpt_dir, "RetinexNet-%s" % train_phase)
                    best_score = score
                    print("[*] Best %s checkpoint: epoch %d, score %.8f" % (train_phase, epoch_index + 1, score))

            # Preview training images and save a last checkpoint.
            if (epoch_index + 1) % eval_every_epoch == 0:
                self.evaluate(epoch_index + 1, sample_low_data, sample_dir=sample_dir, train_phase=train_phase)
                last_checkpoint = self.save(saver, iter_num, ckpt_dir, "RetinexNet-%s" % train_phase)
                last_saved_step = iter_num

        if last_saved_step != iter_num:
            last_checkpoint = self.save(saver, iter_num, ckpt_dir, "RetinexNet-%s" % train_phase)

        print("[*] Finish training for phase %s." % train_phase)
        return last_checkpoint

    def save(self, saver, iter_num, ckpt_dir, model_name):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        print("[*] Saving model %s" % model_name)
        return saver.save(self.sess, \
                   os.path.join(ckpt_dir, model_name), \
                   global_step=iter_num)

    def load(self, saver, ckpt_dir):
        full_path = tf.train.latest_checkpoint(ckpt_dir)
        if full_path:
            try:
                checkpoint_name = full_path.replace('\\', '/').rsplit('/', 1)[-1]
                global_step = int(checkpoint_name.rsplit('-', 1)[-1])
            except ValueError:
                global_step = None
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            print("[*] Failed to load model from %s" % ckpt_dir)
            return False, 0

    def test(self, test_low_data, test_high_data, test_low_data_names, save_dir, decom_flag, checkpoint_dir=None):
        if not checkpoint_dir:
            raise ValueError("Testing requires an explicit checkpoint_dir containing Decom and Relight checkpoints.")
        if len(test_low_data) != len(test_low_data_names):
            raise ValueError("Each test image must have a corresponding filename.")
        print("[*] Reading checkpoint...")
        load_model_status_Decom, _ = self.load(self.saver_Decom, os.path.join(checkpoint_dir, 'Decom'))
        load_model_status_Relight, _ = self.load(self.saver_Relight, os.path.join(checkpoint_dir, 'Relight'))
        missing = [phase for phase, loaded in (("Decom", load_model_status_Decom), ("Relight", load_model_status_Relight)) if not loaded]
        if missing:
            raise FileNotFoundError("Missing %s checkpoint(s) under %s" % (', '.join(missing), checkpoint_dir))
        print("[*] Load weights successfully...")
        os.makedirs(save_dir, exist_ok=True)
        
        print("[*] Testing...")
        for idx in range(len(test_low_data)):
            print(test_low_data_names[idx])
            name, extension = os.path.splitext(os.path.basename(test_low_data_names[idx]))
            suffix = extension.lstrip('.') or 'png'

            input_low_test = np.expand_dims(test_low_data[idx], axis=0)
            [R_low, I_low, I_delta, S] = self.sess.run([self.output_R_low, self.output_I_low, self.output_I_delta, self.output_S], feed_dict = {self.input_low: input_low_test})

            if decom_flag == 1:
                save_images(os.path.join(save_dir, name + "_R_low." + suffix), R_low)
                save_images(os.path.join(save_dir, name + "_I_low." + suffix), I_low)
                save_images(os.path.join(save_dir, name + "_I_delta." + suffix), I_delta)
            save_images(os.path.join(save_dir, name + "_S."   + suffix), S)

