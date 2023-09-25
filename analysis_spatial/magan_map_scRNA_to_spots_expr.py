import magic
import pandas as pd
import numpy as np
import phate
from matplotlib import pyplot as plt
import scprep
import warnings
warnings.simplefilter('ignore')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import sys
import time
import sys
import os
import fcsparser
import functools
import sklearn.decomposition
import seaborn as sns
import scanpy
import graphtools
import pandas as pd
import tensorflow_probability as tfp
import h5py
import scanpy
import glob
import scipy
import phate, scprep
from keras import backend as K
from scipy.stats import gaussian_kde

fig = plt.figure()
fig.set_size_inches((8, 8))

def pearson_r(y_true, y_pred):
    epsilon = 10e-5
    x = y_true
    y = y_pred
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym, axis=0)
    x_square_sum = K.sum(xm * xm, axis=0)
    y_square_sum = K.sum(ym * ym, axis=0)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / (r_den + epsilon)
    return K.mean(r)

adata = scanpy.read_h5ad('./data/V1_Human_Lymph_Node/processed.h5ad')
adata.var['symbol'] = adata.var.index
adata.var.set_index('gene_ids', inplace=True)
adata_ref = scanpy.read_h5ad('./data/reference.h5ad')

intersection_genes = np.array(list(set(adata.var_names).intersection(adata_ref.var_names)))

spat_magic_op = magic.MAGIC(random_state=42, verbose=False, solver='approximate')
data = spat_magic_op.fit_transform(adata.to_df())
spat_genes = np.array(adata.var_names)

ref_magic_op = magic.MAGIC(random_state=42, verbose=False, solver='approximate')
data_ref = ref_magic_op.fit_transform(adata_ref.to_df())
ref_genes = np.array(adata_ref.var_names)

col_pairs =  np.array([[spat_genes.tolist().index(col), ind] for ind, col in enumerate(ref_genes) if (col in spat_genes)])
print('len(col_pairs): {}'.format(col_pairs.shape[0]))
print (col_pairs.shape[0] == len(intersection_genes))

data_ref[intersection_genes] = (data_ref[intersection_genes] - data_ref[intersection_genes].mean(axis=0) + data[intersection_genes].mean(axis=0))

def uniform_density_sample(data, N_PCA=2, nbins=20, points_per_bin=10):
    tmp = sklearn.decomposition.PCA(N_PCA).fit_transform(data)

    sample = []
    _, binsx = np.histogram(tmp[:, 0], bins=nbins)
    _, binsy = np.histogram(tmp[:, 1], bins=nbins)

    for i, binx in enumerate(binsx[:-1]):
        if i % 10 == 0: print('{} / {}'.format(i, nbins - 1))
        for j, biny in enumerate(binsy[:-1]):
            maskx = np.logical_and(tmp[:, 0] > binsx[i], tmp[:, 0] < binsx[i + 1])
            masky = np.logical_and(tmp[:, 1] > binsy[j], tmp[:, 1] < binsy[j + 1])
            mask = np.logical_and(maskx, masky)
            pts = np.argwhere(mask)
            if pts.shape[0] > 0:
                n_sample = min(pts.shape[0], points_per_bin)
                pt = np.random.choice(pts.reshape([-1]), n_sample).tolist()
                sample.extend(pt)
    sample = data[sample, :]

    return sample

x1_raw_nonuniform = data.values.astype(np.float32) / 10
x2_raw_nonuniform = data_ref.values.astype(np.float32) / 10

x1_raw = uniform_density_sample(x1_raw_nonuniform, points_per_bin=10)
x2_raw = uniform_density_sample(x2_raw_nonuniform, points_per_bin=10)

print(x1_raw.shape, x1_raw.min(), x1_raw.max())
print(x2_raw.shape, x2_raw.min(), x2_raw.max())

N_PCA = 20

PCA1 = sklearn.decomposition.PCA(N_PCA, random_state=42)
PCA1.fit(x1_raw)
x1 = PCA1.transform(x1_raw)

PCA2 = sklearn.decomposition.PCA(N_PCA, random_state=42)
PCA2.fit(x2_raw)
x2 = PCA2.transform(x2_raw)

x1_final = PCA1.transform(x1_raw_nonuniform)
x2_final = PCA2.transform(x2_raw_nonuniform)

print(x1.shape, x1.min(), x1.max())
print(x2.shape, x2.min(), x2.max())
##########################################

class Loader(object):
    """A Loader class for feeding numpy matrices into tensorflow models."""

    def __init__(self, data, labels=None, shuffle=False):
        """Initialize the loader with data and optionally with labels."""
        self.start = 0
        self.epoch = 0
        self.data = [x for x in [data, labels] if x is not None]
        self.labels_given = labels is not None

        if shuffle:
            self.r = list(range(data.shape[0]))
            np.random.shuffle(self.r)
            self.data = [x[self.r] for x in self.data]

    def next_batch(self, batch_size=100):
        """Yield just the next batch."""
        num_rows = self.data[0].shape[0]

        if self.start + batch_size < num_rows:
            batch = [x[self.start:self.start + batch_size] for x in self.data]
            self.start += batch_size
        else:
            self.epoch += 1
            batch_part1 = [x[self.start:] for x in self.data]
            batch_part2 = [x[:batch_size - (x.shape[0] - self.start)] for x in self.data]
            batch = [np.concatenate([x1, x2], axis=0) for x1, x2 in zip(batch_part1, batch_part2)]

            self.start = batch_size - (num_rows - self.start)

        if not self.labels_given:  # don't return length-1 list
            return batch[0]
        else:  # return list of data and labels
            return batch

    def iter_batches(self, batch_size=100):
        """Iterate over the entire dataset in batches."""
        num_rows = self.data[0].shape[0]

        end = 0

        if batch_size > num_rows:
            if not self.labels_given:
                yield [x for x in self.data][0]
            else:
                yield [x for x in self.data]
        else:
            for i in range(num_rows // batch_size):
                start = i * batch_size
                end = (i + 1) * batch_size

                if not self.labels_given:
                    yield [x[start:end] for x in self.data][0]
                else:
                    yield [x[start:end] for x in self.data]
            if end < num_rows:
                if not self.labels_given:
                    yield [x[end:] for x in self.data][0]
                else:
                    yield [x[end:] for x in self.data]



load1 = Loader(x1, shuffle=True)
load2 = Loader(x2, shuffle=True)
loadeval1 = Loader(x1, shuffle=False)
loadeval2 = Loader(x2, shuffle=False)
outdim1 = x1.shape[1]
outdim2 = x2.shape[1]

############################################################
####################
TRAINING_STEPS = 15000
batch_size = 128
learning_rate = .0001
nfilt = 256

lambda_cycle = 1 # orig 5
lambda_correspondence = 5 # orig 0.1
#############################################
#############################################
##### tf graph
def tbn(name):

    return tf.get_default_graph().get_tensor_by_name(name)

def minibatch(input_, num_kernels=15, kernel_dim=10, name='',):
    with tf.variable_scope(name):
        W = tf.get_variable('{}/Wmb'.format(name), [input_.get_shape()[-1], num_kernels * kernel_dim])
        b = tf.get_variable('{}/bmb'.format(name), [num_kernels * kernel_dim])

    x = tf.matmul(input_, W) + b
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    diffs = tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    abs_diffs = tf.reduce_mean(tf.abs(diffs), 2)
    minibatch_features = tf.reduce_mean(tf.exp(-abs_diffs), 2)

    return tf.concat([input_, minibatch_features], axis=-1)

def nameop(op, name):

    return tf.identity(op, name=name)

def lrelu(x, leak=0.2, name="lrelu"):

    return tf.maximum(x, leak * x)

def bn(tensor, name, is_training):
    return tensor
    return tf.layers.batch_normalization(tensor,
                      momentum=.9,
                      training=True,
                      name=name)

def build_config(limit_gpu_fraction=0.2, limit_cpu_fraction=10):
    if limit_gpu_fraction > 0:
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=limit_gpu_fraction)
        config = tf.ConfigProto(gpu_options=gpu_options)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        config = tf.ConfigProto(device_count={'GPU': 0})
    if limit_cpu_fraction is not None:
        if limit_cpu_fraction <= 0:
            # -2 gives all CPUs except 2
            cpu_count = min(
                1, int(os.cpu_count() + limit_cpu_fraction))
        elif limit_cpu_fraction < 1:
            # 0.5 gives 50% of available CPUs
            cpu_count = min(
                1, int(os.cpu_count() * limit_cpu_fraction))
        else:
            # 2 gives 2 CPUs
            cpu_count = int(limit_cpu_fraction)
        config.inter_op_parallelism_threads = cpu_count
        config.intra_op_parallelism_threads = cpu_count
        os.environ['OMP_NUM_THREADS'] = str(1)
        os.environ['MKL_NUM_THREADS'] = str(cpu_count)
    return config

def get_layer(sess, intensor, data, outtensor, batch_size=100):
    out = []
    for batch in np.array_split(data, max(1, data.shape[0] / batch_size)):
        feed = {intensor: batch}
        batchout = sess.run(outtensor, feed_dict=feed)
        out.append(batchout)
    out = np.concatenate(out, axis=0)

    return out

def Generator(x, nfilt, outdim, activation=lrelu, is_training=True):
    h1 = tf.layers.dense(x, nfilt * 1, activation=None, name='h1')
    h1 = bn(h1, 'h1', is_training)
    h1 = activation(h1)

    h1b = tf.layers.dense(x, nfilt * 1, activation=None, name='h1b')
    h1b = tf.nn.relu(h1b)
    h1 = tf.concat([h1, h1b], axis=-1)

    h2 = tf.layers.dense(h1, nfilt * 2, activation=None, name='h2')
    h2 = bn(h2, 'h2', is_training)
    h2 = activation(h2)

    h2b = tf.layers.dense(h1, nfilt * 2, activation=None, name='h2b')
    h2b = tf.nn.relu(h2b)
    h2 = tf.concat([h2, h2b], axis=-1)

    h3 = tf.layers.dense(h2, nfilt * 4, activation=None, name='h3')
    h3 = bn(h3, 'h3', is_training)
    h3 = activation(h3)

    h3b = tf.layers.dense(h2, nfilt * 4, activation=None, name='h3b')
    h3b = tf.nn.relu(h3b)
    h3 = tf.concat([h3, h3b], axis=-1)

    # h4 = tf.layers.dense(h3, nfilt * 8, activation=None, name='h4')
    # h4 = bn(h4, 'h4', is_training)
    # h4 = activation(h4)

    # h5 = tf.layers.dense(h4, nfilt * 16, activation=None, name='h5')
    # h5 = bn(h5, 'h5', is_training)
    # h5 = activation(h5)

    out = tf.layers.dense(h3, outdim, activation=None, name='out')

    return out

def Discriminator(x, nfilt, outdim, activation=tf.nn.relu, is_training=True):
    h1 = tf.layers.dense(x, nfilt * 4, activation=None, name='h1')
    h1 = activation(h1)
    h1 = minibatch(h1)

    h1b = tf.layers.dense(x, nfilt * 4, activation=None, name='h1b')
    h1b = lrelu(h1b)
    h1b = minibatch(h1b)
    h1 = tf.concat([h1, h1b], axis=-1)

    h2 = tf.layers.dense(h1, nfilt * 2, activation=None, name='h2')
    h2 = bn(h2, 'h2', is_training)
    h2 = activation(h2)

    h2b = tf.layers.dense(h1, nfilt * 2, activation=None, name='h2b')
    h2b = lrelu(h2b)
    h2 = tf.concat([h2, h2b], axis=-1)

    h3 = tf.layers.dense(h2, nfilt * 1, activation=None, name='h3')
    h3 = bn(h3, 'h3', is_training)
    h3 = activation(h3)

    h3b = tf.layers.dense(h2, nfilt * 1, activation=None, name='h3b')
    h3b = lrelu(h3b)
    h3 = tf.concat([h3, h3b], axis=-1)

    # h4 = tf.layers.dense(h3, nfilt * 2, activation=None, name='h4')
    # h4 = bn(h4, 'h4', is_training)
    # h4 = activation(h4)

    # h5 = tf.layers.dense(h4, nfilt * 1, activation=None, name='h5')
    # h5 = bn(h5, 'h5', is_training)
    # h5 = activation(h5)

    out = tf.layers.dense(h3, outdim, activation=None, name='out')

    return out

def adversarial_loss(logits, labels):

    return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)

def compute_pairwise_distances(A):
    r = tf.reduce_sum(A*A, 1)

    # turn r into column vector
    r = tf.reshape(r, [-1, 1])
    D = r - 2*tf.matmul(A, tf.transpose(A)) + tf.transpose(r)

    return D

def make_legend(ax, labels, sort=False, s=20, scattercoords=[0, 0], cmap=mpl.cm.jet, **kwargs):
    if sort:
        uniquelabs = np.unique(labels)
    else:
        uniquelabs = labels
    numlabs = len(uniquelabs)
    for i, label in enumerate(uniquelabs):
        if numlabs > 1:
            ax.scatter(scattercoords[0], scattercoords[1], s=s, c=[cmap(1 * i / (numlabs - 1))], label=label)
        else:
            ax.scatter(scattercoords[0], scattercoords[1], s=s, c=[cmap(1.)], label=label)
    ax.scatter(scattercoords[0], scattercoords[1], s=2 * s, c='w')
    ax.legend(**kwargs)



tf.reset_default_graph()
loss_D = 0.
loss_G = 0.
tfis_training = tf.placeholder(tf.bool, [], name='tfis_training')

tfx1 = tf.placeholder(tf.float32, [None, outdim1], name='x1')
tfx2 = tf.placeholder(tf.float32, [None, outdim2], name='x2')

with tf.variable_scope('generator12', reuse=tf.AUTO_REUSE):
    fake2 = Generator(tfx1, nfilt, outdim=outdim2, is_training=tfis_training)
fake2 = nameop(fake2, 'fake2')

with tf.variable_scope('generator21', reuse=tf.AUTO_REUSE):
    fake1 = Generator(tfx2, nfilt, outdim=outdim1, is_training=tfis_training)
fake1 = nameop(fake1, 'fake1')

with tf.variable_scope('generator12', reuse=tf.AUTO_REUSE):
    cycle2 = Generator(fake1, nfilt, outdim=outdim2, is_training=tfis_training)
cycle2 = nameop(cycle2, 'cycle2')

with tf.variable_scope('generator21', reuse=tf.AUTO_REUSE):
    cycle1 = Generator(fake2, nfilt, outdim=outdim1, is_training=tfis_training)
cycle1 = nameop(cycle1, 'cycle1')


with tf.variable_scope('discriminator1', reuse=tf.AUTO_REUSE):
    d_real1 = Discriminator(tfx1, 2 * nfilt, 1, is_training=tfis_training)
    d_fake1 = Discriminator(fake1, 2 * nfilt, 1, is_training=tfis_training)

with tf.variable_scope('discriminator2', reuse=tf.AUTO_REUSE):
    d_real2 = Discriminator(tfx2, 2 * nfilt, 1, is_training=tfis_training)
    d_fake2 = Discriminator(fake2, 2 * nfilt, 1, is_training=tfis_training)

real = tf.concat([d_real1, d_real2], axis=0)
fake = tf.concat([d_fake1, d_fake2], axis=0)
##################################################



##################################################
loss_D_fake = tf.reduce_mean(adversarial_loss(logits=real, labels=tf.ones_like(real)))
loss_D_real = tf.reduce_mean(adversarial_loss(logits=fake, labels=tf.zeros_like(fake)))
loss_G_disc = tf.reduce_mean(adversarial_loss(logits=fake, labels=tf.ones_like(fake)))

loss_D += .5 * (loss_D_fake + loss_D_real)
loss_G += loss_G_disc

tf.add_to_collection('losses', nameop(loss_D_real, 'loss_D_real'))
tf.add_to_collection('losses', nameop(loss_D_fake, 'loss_D_fake'))
tf.add_to_collection('losses', nameop(loss_G_disc, 'loss_G_disc'))


loss_cycle = tf.reduce_mean((tfx1 - cycle1)**2) + tf.reduce_mean((tfx2 - cycle2)**2)
loss_G += lambda_cycle * loss_cycle
tf.add_to_collection('losses', nameop(loss_cycle, 'loss_cycle'))

########################################
##### define correspondence loss here

W1 = tf.constant(PCA1.components_)
W2 = tf.constant(PCA2.components_)
tfx1_ambient = tf.matmul(tfx1, W1)
tfx2_ambient = tf.matmul(tfx2, W2)
fake1_ambient = tf.matmul(fake1, W1)
fake2_ambient = tf.matmul(fake2, W2)

tfx1_ambient = tf.gather(tfx1_ambient, col_pairs[:, 0], axis=1)
tfx2_ambient = tf.gather(tfx2_ambient, col_pairs[:, 1], axis=1)
fake1_ambient = tf.gather(fake1_ambient, col_pairs[:, 0], axis=1)
fake2_ambient = tf.gather(fake2_ambient, col_pairs[:, 1], axis=1)

loss_correspondence = 0

# n_eigenvectors = 1
# loss_correspondence = []
# _, eigv1 = tf.linalg.eigh(compute_pairwise_distances(tfx1_ambient))
# _, eigv2 = tf.linalg.eigh(compute_pairwise_distances(tfx2_ambient))
# for i_eig in range(n_eigenvectors):
#     e1 = eigv1[:, i_eig]
#     e2 = eigv2[:, i_eig]

#     tfx1_e = tf.matmul(tf.transpose(tfx1_ambient), e1[:, np.newaxis])
#     tfg2_e = tf.matmul(tf.transpose(fake2_ambient), e1[:, np.newaxis])

#     tfg1_e = tf.matmul(tf.transpose(fake1_ambient), e2[:, np.newaxis])
#     tfx2_e = tf.matmul(tf.transpose(tfx2_ambient), e2[:, np.newaxis])

#     loss_correspondence.append(tf.reduce_mean((tfx1_e - tfg2_e)**2))
#     loss_correspondence.append(tf.reduce_mean((tfx2_e - tfg1_e)**2))
# loss_correspondence = tf.reduce_mean(loss_correspondence)

#loss_correspondence += tf.reduce_mean((tfx1_ambient - fake2_ambient)**2)
#loss_correspondence += tf.reduce_mean((tfx2_ambient - fake1_ambient)**2)
loss_correspondence += -1*pearson_r(tf.transpose(tfx1_ambient), tf.transpose(fake2_ambient))
loss_correspondence += -1*pearson_r(tf.transpose(tfx2_ambient), tf.transpose(fake1_ambient))
#loss_correspondence += -1*pearson_r(tfx1_ambient, fake2_ambient)
#loss_correspondence += -1*pearson_r(tfx2_ambient, fake1_ambient)

# n_eigenvectors = 1
# loss_correspondence = []
# _, eigv1 = tf.linalg.eigh(compute_pairwise_distances(tfx1))
# _, eigv2 = tf.linalg.eigh(compute_pairwise_distances(tfx2))
# for i_eig in range(n_eigenvectors):
#     e1 = eigv1[:, i_eig]
#     e2 = eigv2[:, i_eig]

#     tfx1_e = tf.matmul(tf.transpose(tfx1), e1[:, np.newaxis])
#     tfg2_e = tf.matmul(tf.transpose(fake2), e1[:, np.newaxis])

#     tfg1_e = tf.matmul(tf.transpose(fake1), e2[:, np.newaxis])
#     tfx2_e = tf.matmul(tf.transpose(tfx2), e2[:, np.newaxis])

#     loss_correspondence.append(tf.reduce_mean((tfx1_e - tfg2_e)**2))
#     loss_correspondence.append(tf.reduce_mean((tfx2_e - tfg1_e)**2))
# loss_correspondence = tf.reduce_mean(loss_correspondence)


# loss_correspondence = tf.reduce_mean((tfx1 - fake2)**2) + tf.reduce_mean((tfx2 - fake1)**2)
loss_G += lambda_correspondence * loss_correspondence
tf.add_to_collection('losses', nameop(loss_correspondence, 'loss_correspondence'))
########################################



Gvars = [tv for tv in tf.global_variables() if any(['generator' in tv.name or 'coder' in tv.name])]
Dvars = [tv for tv in tf.global_variables() if 'discriminator' in tv.name]
mul = lambda x, y: x * y
add = lambda x, y: x + y
red = functools.reduce
total_params_G = red(add, [red(mul, v.shape.as_list()) for v in Gvars if len(v.shape) > 0])
total_params_D = red(add, [red(mul, v.shape.as_list()) for v in Dvars if len(v.shape) > 0])
print('Generator vars: {}'.format(len(Gvars)))
print('Discriminator vars: {}'.format(len(Dvars)))
print('Generator params: {}'.format(total_params_G))
print('Discriminator params: {}'.format(total_params_D))

update_ops_D = [op for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'discriminator' in op.name]
update_ops_G = [op for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'generator' in op.name]
print('update ops G: {}'.format(len(update_ops_G)))
print('update ops D: {}'.format(len(update_ops_D)))

with tf.control_dependencies(update_ops_D):
    optD = tf.train.AdamOptimizer(learning_rate)
    train_op_D = optD.minimize(loss_D, var_list=Dvars)
with tf.control_dependencies(update_ops_G):
    optG = tf.train.AdamOptimizer(learning_rate)
    train_op_G = optG.minimize(loss_G, var_list=Gvars)
##################################################

sess = tf.Session(config=build_config(limit_gpu_fraction=.5))

sess.run(tf.global_variables_initializer())


pca_viz1 = sklearn.decomposition.PCA(3)
pca_viz2 = sklearn.decomposition.PCA(3)
pca_viz1.fit(x1)
pca_viz2.fit(x2)
pca_viz_x1 = pca_viz1.transform(x1)
pca_viz_x2 = pca_viz2.transform(x2)



t = time.time()
training_counter = 0
losses = [tns.name[:-2].replace('loss_', '').split('/')[-1] for tns in tf.get_collection('losses')]
print("Losses: {}".format(' '.join(losses)))
while training_counter < TRAINING_STEPS + 1:
    training_counter += 1
    batch_x1 = load1.next_batch(batch_size)
    batch_x2 = load2.next_batch(batch_size)

    feed = {tbn('x1:0'): batch_x1, tbn('x2:0'): batch_x2, tbn('tfis_training:0'): True}
    sess.run(train_op_G, feed_dict=feed)
    sess.run(train_op_D, feed_dict=feed)


    if training_counter % 100 == 0:
        losses = [tns.name[:-2].replace('loss_', '').split('/')[-1] for tns in tf.get_collection('losses')]
        losses_ = sess.run(tf.get_collection('losses'), feed_dict=feed)
        lstring = ' '.join(['{:.5f}'.format(loss) for loss in losses_])
        print("{} ({:.5f} s): {}".format(training_counter, time.time() - t, lstring))
        t = time.time()

    if training_counter in [100, 5000, 7500, 10000, 12500, TRAINING_STEPS]:
        
        col1 = 10122
        col2 = 6294
        
        eval_fake1 = get_layer(sess, tbn('x2:0'), x2_final, tbn('fake1:0'))
        pca_viz_fake1 = pca_viz1.transform(eval_fake1)

        fig, ax = plt.subplots(1,4, figsize=(14, 4))
        ax = ax.flatten()

        scprep.plot.scatter2d(pca_viz_x1, c='lightgray', ticks=None, ax=ax[0], alpha=0.1)
        scprep.plot.scatter2d(pca_viz_fake1, c=adata_ref.obs['Subset'] == 'T_CD4+_TfH_GC',
                              title='Fake Ref\nT_CD4+_TfH_GC', ticks=None, ax=ax[0])
        
        scprep.plot.scatter2d(pca_viz_x1, c='lightgray', ticks=None, ax=ax[1], alpha=0.1)
        scprep.plot.scatter2d(pca_viz_fake1, c=adata_ref.obs['Subset'] == 'B_GC_LZ',
                              title='Fake Ref\nB_GC_LZ', ticks=None, ax=ax[1])
        
        scprep.plot.scatter2d(pca_viz_x1, c='lightgray', ticks=None, ax=ax[2], alpha=0.1)
        scprep.plot.scatter2d(pca_viz_fake1, c=adata_ref.obs['Subset'] == 'T_CD4+_naive',
                              title='Fake Ref\nT_CD4+_naive', ticks=None, ax=ax[2])
        
        scprep.plot.scatter2d(pca_viz_x1, c='lightgray', ticks=None, ax=ax[3], alpha=0.1)
        scprep.plot.scatter2d(pca_viz_fake1, c=adata_ref.obs['Subset'] == 'B_IFN',
                              title='Fake Ref\nB_IFN', ticks=None, ax=ax[3])
        
        plt.tight_layout()

        fig.savefig(f'/home/aarthivenkat/Gene-Signal-Pattern-Analysis/analysis_spatial/results/scMMGAN/plot_{training_counter}.png')
        print('Plot saved.')
        
        output_fake1 = get_layer(sess, tbn('x2:0'), x2_final, tbn('fake1:0'))
        #output_fake1 = get_layer(sess, tbn('x2:0'), x2, tbn('fake1:0'))
        #output_fake1 = PCA1.inverse_transform(output_fake1)
        #output_fake1 = output_fake1 * 10
        
        with open(f'/home/aarthivenkat/Gene-Signal-Pattern-Analysis/analysis_spatial/results/scMMGAN/output_ref_to_spat_cycle_1_correspondence_lambda_5_corr_correspondence.npz', 'wb+') as f:
            np.savez(f, spots_pc=x1_final, generate_ref_to_spots_pc=output_fake1)

        print ('Mki67 correlation(fake1, tfx2)', scipy.stats.spearmanr(PCA1.inverse_transform(eval_fake1)[:, col1], PCA2.inverse_transform(x2_final)[:, col2]).correlation)

        correlations = []
        eval_fake1_ambient = PCA1.inverse_transform(eval_fake1)
        x2_ambient = PCA2.inverse_transform(x2_final)
        for cell in range(eval_fake1_ambient.shape[0]):
            correlations.append(scipy.stats.spearmanr(eval_fake1_ambient[cell, col_pairs[:, 0]], x2_ambient[cell, col_pairs[:, 1]]).correlation)
        
        print (f'Mean cell correlations {np.nanmean(correlations)}')

        correlations = []
        for pair in col_pairs:
            correlations.append(scipy.stats.spearmanr(eval_fake1_ambient[:, pair[0]], x2_ambient[:, pair[1]]).correlation)
        
        print (f'Mean gene correlations {np.nanmean(correlations)}')