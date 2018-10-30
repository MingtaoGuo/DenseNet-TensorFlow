from network import DenseNet
from ops import softmax, to_OneHot
import tensorflow as tf
from utils import read_cifar_data, shuffle
import numpy as np

def test_acc(path, class_nums, growth_rate, depth):
    inputs = tf.placeholder(tf.float32, [None, 32, 32, 3])
    labels = tf.placeholder(tf.int64, [None])
    train_phase = tf.placeholder(tf.bool)
    logits = DenseNet(inputs, nums_out=class_nums, growth_rate=growth_rate, train_phase=train_phase, depth=depth)
    pred = softmax(logits)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis=1), labels), tf.float32))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, "./save_para//.\\densenet.ckpt")
    data, labels_ = read_cifar_data(path)
    acc = 0
    for i in range(data.shape[0] // 100):
        acc += sess.run(accuracy, feed_dict={inputs: data[i * 100:i*100 + 100], labels: labels_[i * 100:i*100 + 100], train_phase: False})
    return acc / (data.shape[0] // 100)

def validation_acc(inputs_ph, labels_ph, train_phase_ph, accuracy, sess, path):
    data, labels_ = read_cifar_data(path)
    data, labels_ = data[5000:], labels_[5000:]
    acc = 0
    for i in range(data.shape[0] // 100):
        acc += sess.run(accuracy, feed_dict={inputs_ph: data[i * 100:i*100 + 100], labels_ph: labels_[i * 100:i*100 + 100], train_phase_ph: False})
    return acc / (data.shape[0] // 100)

def train(batch_size, class_nums, growth_rate, weight_decay, depth, cifar10_path, train_epoch):
    inputs = tf.placeholder(tf.float32, [None, 32, 32, 3])
    labels = tf.placeholder(tf.int64, [None])
    train_phase = tf.placeholder(tf.bool)
    learning_rate = tf.placeholder(tf.float32)
    logits = DenseNet(inputs, nums_out=class_nums, growth_rate=growth_rate, train_phase=train_phase, depth=depth)
    pred = softmax(logits)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis=1), labels), tf.float32))
    one_hot_label = to_OneHot(labels, class_nums)
    cross_entropy_loss = tf.reduce_mean(-tf.log(tf.reduce_sum(pred * one_hot_label, axis=1) + 1e-10))
    regular = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    Opt = tf.train.MomentumOptimizer(learning_rate, momentum=0.9, use_nesterov=True).minimize(cross_entropy_loss + weight_decay * regular)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    path = cifar10_path + "data_batch_"
    valid_path = cifar10_path + "data_batch_5"
    lr = 0.1
    loss_list = []
    train_acc_list = []
    test_acc_list = []
    saver = tf.train.Saver()
    # saver.restore(sess, "./save_para//.\\densenet.ckpt")
    for epoch in range(train_epoch):
        if epoch == train_epoch // 2 or epoch == train_epoch * 3 // 4:
            lr /= 10
        for i in range(1, 6):
            if i != 5:
                data, labels_ = read_cifar_data(path + str(i))
                data, labels_ = shuffle(data, labels_)
            else:
                data, labels_ = read_cifar_data(path + str(i))
                data, labels_ = shuffle(data[:5000], labels_[:5000])
            for j in range(data.shape[0] // batch_size - 1):
                batch_data = data[j * batch_size:j * batch_size + batch_size, :, :, :]
                batch_labels = labels_[j * batch_size:j * batch_size + batch_size]
                [_, loss, acc] = sess.run([Opt, cross_entropy_loss, accuracy], feed_dict={inputs: batch_data, labels: batch_labels, train_phase: True, learning_rate: lr})
                loss_list.append(loss)
                train_acc_list.append(acc)
                if j % 100 == 0:
                    print("Epoch: %d, iter: %d, loss: %f, train_acc: %f"%(epoch, j, loss, acc))
                    np.savetxt("loss.txt", loss_list)
                    np.savetxt("train_acc.txt", train_acc_list)
                    np.savetxt("test_acc.txt", test_acc_list)
            vali_acc = validation_acc(inputs, labels, train_phase, accuracy, sess, valid_path)
            test_acc_list.append(vali_acc)
            print("Validation Accuracy: %f"%(vali_acc))
            saver.save(sess, "./save_para//densenet.ckpt")



if __name__ == "__main__":
    train(batch_size=64, class_nums=10, growth_rate=12, weight_decay=1e-4, depth=40)