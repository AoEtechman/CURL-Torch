################################################################################
# Copyright 2019 DeepMind Technologies Limited
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         https://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
################################################################################
"""Script to train CURL."""

import collections
import functools
from absl import logging


from arguments import get_args, print_args

import numpy as np
from sklearn import neighbors
# import sonnet as snt
# import tensorflow.compat.v1 as tf
# import tensorflow_datasets as tfds
# import tensorflow_probability as tfp
import torch
from data_loader import GeneralImgDataset, BinarizeTransform
import torchvision.transforms as transforms
from torcheval.metrics.functional import multiclass_confusion_matrix
import torch.distributions as distributions
import torch.optim as optim
from torch.utils.data import DataLoader

from curl import model
from curl import utils
from CURL.curl.data_loader import load_data

# tfc = tf.compat.v1

# pylint: disable=g-long-lambda

MainOps = collections.namedtuple('MainOps', [
    'elbo', 'll', 'log_p_x', 'kl_y', 'kl_z', 'elbo_supervised', 'll_supervised',
    'log_p_x_supervised', 'kl_y_supervised', 'kl_z_supervised',
    'cat_probs', 'confusion', 'purity', 'latents'
])

DatasetTuple = collections.namedtuple('DatasetTuple', [
    'train_data', 'train_iter_for_clf', 'train_data_for_clf',
    'valid_iter', 'valid_data', 'test_iter', 'test_data', 'ds_info'
])


def compute_purity(confusion):
  return np.sum(np.max(confusion, axis=0)).astype(float) / np.sum(confusion)



binarize_transform = transforms.compose([transforms.ToTensor(), BinarizeTransform()])

# def process_dataset(iterator,
#                     ops_to_run,
#                     sess,
#                     feed_dict=None,
#                     aggregation_ops=np.stack,
#                     processing_ops=None):
#   """Process a dataset by computing ops and accumulating batch by batch.

#   Args:
#     iterator: iterator through the dataset.
#     ops_to_run: dict, tf ops to run as part of dataset processing.
#     sess: tf.Session to use.
#     feed_dict: dict, required placeholders.
#     aggregation_ops: fn or dict of fns, aggregation op to apply for each op.
#     processing_ops: fn or dict of fns, extra processing op to apply for each op.

#   Returns:
#     Results accumulated over dataset.
#   """

#   if not isinstance(ops_to_run, dict):
#     raise TypeError('ops_to_run must be specified as a dict')

#   if not isinstance(aggregation_ops, dict):
#     aggregation_ops = {k: aggregation_ops for k in ops_to_run}
#   if not isinstance(processing_ops, dict):
#     processing_ops = {k: processing_ops for k in ops_to_run}

#   out_results = collections.OrderedDict()
#   sess.run(iterator.initializer)
#   while True:
#     # Iterate over the whole dataset and append the results to a per-key list.
#     try:
#       outs = sess.run(ops_to_run, feed_dict=feed_dict)
#       for key, value in outs.items():
#         out_results.setdefault(key, []).append(value)

#     except tf.errors.OutOfRangeError:  # end of dataset iterator
#       break

#   # Aggregate and process results.
#   for key, value in out_results.items():
#     if aggregation_ops[key]:
#       out_results[key] = aggregation_ops[key](value)
#     if processing_ops[key]:
#       out_results[key] = processing_ops[key](out_results[key], axis=0)

#   return out_results


# def get_data_sources(dataset, dataset_kwargs, batch_size, test_batch_size,
#                      training_data_type, n_concurrent_classes, image_key,
#                      label_key):
#   """Create and return data sources for training, validation, and testing.

#   Args:
#     dataset: str, name of dataset ('mnist', 'omniglot', etc).
#     dataset_kwargs: dict, kwargs used in tf dataset constructors.
#     batch_size: int, batch size used for training.
#     test_batch_size: int, batch size used for evaluation.
#     training_data_type: str, how training data is seen ('iid', or 'sequential').
#     n_concurrent_classes: int, # classes seen at a time (ignored for 'iid').
#     image_key: str, name if image key in dataset.
#     label_key: str, name of label key in dataset.

#   Returns:
#     A namedtuple containing all of the dataset iterators and batches.

#   """

#   # Load training data sources
#   ds_train, ds_info = tfds.load(
#       name=dataset,
#       split=tfds.Split.TRAIN,
#       with_info=True,
#       as_dataset_kwargs={'shuffle_files': False},
#       **dataset_kwargs)

#   # Validate assumption that data is in [0, 255]
#   assert ds_info.features[image_key].dtype == torch.uint8

#   n_classes = ds_info.features[label_key].num_classes
#   num_train_examples = ds_info.splits['train'].num_examples

#   def preprocess_data(x):
#     """Convert images from uint8 in [0, 255] to float in [0, 1]."""
#     x[image_key] = tf.image.convert_image_dtype(x[image_key], tf.float32)
#     return x

#   if training_data_type == 'sequential':
#     c = None  # The index of the class number, None for now and updated later
#     if n_concurrent_classes == 1:
#       filter_fn = lambda v: tf.equal(v[label_key], c)
#     else:
#       # Define the lowest and highest class number at each data period.
#       assert n_classes % n_concurrent_classes == 0, (
#           'Number of total classes must be divisible by '
#           'number of concurrent classes')
#       cmin = []
#       cmax = []
#       for i in range(int(n_classes / n_concurrent_classes)):
#         for _ in range(n_concurrent_classes):
#           cmin.append(i * n_concurrent_classes)
#           cmax.append((i + 1) * n_concurrent_classes)

#       filter_fn = lambda v: tf.logical_and(
#           tf.greater_equal(v[label_key], cmin[c]), tf.less(
#               v[label_key], cmax[c]))

#     # Set up data sources/queues (one for each class).
#     train_datasets = []
#     train_iterators = []
#     train_data = []

#     full_ds = ds_train.repeat().shuffle(num_train_examples, seed=0)
#     full_ds = full_ds.map(preprocess_data)
#     for c in range(n_classes):
#       filtered_ds = full_ds.filter(filter_fn).batch(
#           batch_size, drop_remainder=True)
#       train_datasets.append(filtered_ds)
#       train_iterators.append(train_datasets[-1].make_one_shot_iterator())
#       train_data.append(train_iterators[-1].get_next())

#   else:  # not sequential
#     full_ds = ds_train.repeat().shuffle(num_train_examples, seed=0)
#     full_ds = full_ds.map(preprocess_data)
#     train_datasets = full_ds.batch(batch_size, drop_remainder=True)
#     train_data = train_datasets.make_one_shot_iterator().get_next()

#   # Set up data source to get full training set for classifier training
#   full_ds = ds_train.repeat(1).shuffle(num_train_examples, seed=0)
#   full_ds = full_ds.map(preprocess_data)
#   train_datasets_for_classifier = full_ds.batch(
#       test_batch_size, drop_remainder=True)
#   train_iter_for_classifier = (
#       train_datasets_for_classifier.make_initializable_iterator())
#   train_data_for_classifier = train_iter_for_classifier.get_next()

#   # Load validation dataset.
#   try:
#     valid_dataset = tfds.load(
#         name=dataset, split=tfds.Split.VALIDATION, **dataset_kwargs)
#     num_valid_examples = ds_info.splits[tfds.Split.VALIDATION].num_examples
#     assert (num_valid_examples %
#             test_batch_size == 0), ('test_batch_size must be a divisor of %d' %
#                                     num_valid_examples)
#     valid_dataset = valid_dataset.repeat(1).batch(
#         test_batch_size, drop_remainder=True)
#     valid_dataset = valid_dataset.map(preprocess_data)
#     valid_iter = valid_dataset.make_initializable_iterator()
#     valid_data = valid_iter.get_next()
#   except (KeyError, ValueError):
#     logging.warning('No validation set!!')
#     valid_iter = None
#     valid_data = None

#   # Load test dataset.
#   test_dataset = tfds.load(
#       name=dataset, split=tfds.Split.TEST, **dataset_kwargs)
#   num_test_examples = ds_info.splits['test'].num_examples
#   assert (num_test_examples %
#           test_batch_size == 0), ('test_batch_size must be a divisor of %d' %
#                                   num_test_examples)
#   test_dataset = test_dataset.repeat(1).batch(
#       test_batch_size, drop_remainder=True)
#   test_dataset = test_dataset.map(preprocess_data)
#   test_iter = test_dataset.make_initializable_iterator()
#   test_data = test_iter.get_next()
#   logging.info('Loaded %s data', dataset)

#   return DatasetTuple(train_data, train_iter_for_classifier,
#                       train_data_for_classifier, valid_iter, valid_data,
#                       test_iter, test_data, ds_info)


def forward_pass(x, label, y, n_y, rgb, batch_size, curl_model, train_supervised,
                                   classify_with_samples, is_training, optimizer = None):
  """Set up the graph and return ops for training or evaluation.

  Args:
    x: tf placeholder for image.
    label: tf placeholder for ground truth label.
    y: tf placeholder for some self-supervised label/prediction.
    n_y: int, dimensionality of discrete latent variable y.
    curl_model: snt.AbstractModule representing the CURL model.
    classify_with_samples: bool, whether to *sample* latents for classification.
    is_training: bool, whether this graph is the training graph.
    name: str, graph name.

  Returns:
    A namedtuple with the required graph ops to perform training or evaluation.

  """
  # kl_y_supervised is -log q(y=y_true | x)

  dataset = GeneralImgDataset(x, y, label, binarize_transform, rgb )

  if is_training:
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = 2, drop_last = True)
  else:
    dataloader = DataLoader(dataset, batch_size = batch_size, num_workers = 2, drop_last = True)

  elbo_history = []
  ll_history = []
  ll_supervised_history = []
  elbo_history_supervised = []
  kl_y_history = []
  kl_z_history = []
  log_p_x_supervised_history = []
  kl_y_supervised_history = []
  kl_z_supervised_history = []
  hiddens_history = []
  cat_history = []
  cat_probs_history = []
  confusion_history = []
  latents_history = []
  for x,y,label in dataloader:
    if is_training:
      (log_p_x, kl_y, kl_z, log_p_x_supervised, kl_y_supervised,
      kl_z_supervised) = curl_model(x, y)

      ll = log_p_x - kl_y - kl_z
      ll_history.append(ll)
      elbo = -torch.mean(ll)
      elbo_history.append(elbo)

      # Supervised loss, either for SMGR, or adaptation to supervised benchmark.
      ll_supervised = log_p_x_supervised - kl_y_supervised - kl_z_supervised
      ll_supervised_history.append(ll_supervised)
      elbo_supervised = -torch.mean(ll_supervised)
      elbo_history_supervised.append(elbo_supervised)

      # Summaries
      kl_y = torch.mean(kl_y)
      kl_y_history.append(kl_y)
      kl_z = torch.mean(kl_z)
      kl_z_history.append(kl_z)
      log_p_x_supervised = torch.mean(log_p_x_supervised)
      log_p_x_supervised_history.append(log_p_x_supervised)
      kl_y_supervised = torch.mean(kl_y_supervised)
      kl_y_supervised_history.append(kl_y_supervised)
      kl_z_supervised = torch.mean(kl_z_supervised)
      kl_z_supervised_history.append(kl_z_supervised)

      optimizer.zero_grad()
      if train_supervised:
        elbo_supervised.backward()
      else:
        elbo.backward()

      optimizer.step()

      # Evaluation.
      curl_model.eval()
      with torch.no_grad():
        hiddens = curl_model.get_shared_rep(x)
        hiddens_history.append(hiddens)
        cat = curl_model.infer_cluster(hiddens)
        cat_history.append(cat)
        cat_probs = cat.probs
        cat_probs_history.append(cat_probs)

      
        confusion = multiclass_confusion_matrix(label, torch.argmax(cat_probs, axis=1),
                                        num_classes=n_y)
        confusion_history.append(confusion)

        if classify_with_samples:
          latents = curl_model.infer_latent(
              hiddens=hiddens, y= cat.sample().float()).sample()
        else:
          latents = curl_model.infer_latent(
              hiddens=hiddens, y= cat.mode().float()).mean()
        latents_history.append(latents)
          
      curl_model.train()
    else:
      curl_model.eval()
      with torch.no_grad():
        (log_p_x, kl_y, kl_z, log_p_x_supervised, kl_y_supervised,
        kl_z_supervised) = curl_model(x, y)

        ll = log_p_x - kl_y - kl_z
        elbo = -torch.mean(ll)
        elbo_history.append(elbo)

        # Supervised loss, either for SMGR, or adaptation to supervised benchmark.
        ll_supervised = log_p_x_supervised - kl_y_supervised - kl_z_supervised
        elbo_supervised = -torch.mean(ll_supervised)
        elbo_history_supervised.append(elbo_supervised)

        # Summaries
        kl_y = torch.mean(kl_y)
        kl_y_history.append(kl_y)
        kl_z = torch.mean(kl_z)
        kl_z_history.append(kl_z)
        log_p_x_supervised = torch.mean(log_p_x_supervised)
        log_p_x_supervised_history.append(log_p_x_supervised)
        kl_y_supervised = torch.mean(kl_y_supervised)
        kl_y_supervised_history.append(kl_y_supervised)
        kl_z_supervised = torch.mean(kl_z_supervised)
        kl_z_supervised_history.append(kl_z_supervised)


        hiddens = curl_model.get_shared_rep(x)
        hiddens_history.append(hiddens)
        cat = curl_model.infer_cluster(hiddens)
        cat_history.append(cat)
        cat_probs = cat.probs
        cat_probs_history.append(cat_probs)

      
        confusion = multiclass_confusion_matrix(label, torch.argmax(cat_probs, axis=1),
                                        num_classes=n_y)
        confusion_history.append(confusion)
        
        

        if classify_with_samples:
          latents = curl_model.infer_latent(
              hiddens=hiddens, y= cat.sample().float()).sample()
        else:
          latents = curl_model.infer_latent(
              hiddens=hiddens, y= cat.mode().float()).mean()
        latents_history.append(latents)
          
     

  confusion_agg = torch.zeros_like(confusion_history[0])
  

  for conf in confusion_history:
    confusion_agg += conf
  purity = (torch.sum(torch.max(confusion_agg, axis=0))
                  / torch.sum(confusion_agg))
  return MainOps(torch.mean(elbo_history), torch.mean(ll), torch.mean(log_p_x), torch.mean(kl_y), torch.mean(kl_z), torch.mean(elbo_supervised), torch.mean(ll_supervised),
                 torch.mean(log_p_x_supervised), torch.mean(kl_y_supervised), torch.mean(kl_z_supervised),
                 cat_probs, confusion_agg, purity, latents)


def get_generated_data(model, y_input, gen_buffer_size,
                       component_counts):
  """Get generated model data (in place of saving a model snapshot).

  Args:
    sess: tf.Session.
    gen_op: tf op representing a batch of generated data.
    y_input: tf placeholder for which mixture components to generate from.
    gen_buffer_size: int, number of data points to generate.
    component_counts: np.array, prior probabilities over components.

  Returns:
    A tuple of two numpy arrays
      The generated data
      The corresponding labels
  """

  batch_size, n_y = y_input.shape.tolist()

  # Sample based on the history of all components used.
  cluster_sample_probs = component_counts.astype(float)
  cluster_sample_probs = np.maximum(1e-12, cluster_sample_probs)
  cluster_sample_probs = cluster_sample_probs / np.sum(cluster_sample_probs)

  # Now generate the data based on the specified cluster prior.
  gen_buffer_images = []
  gen_buffer_labels = []
  for _ in range(gen_buffer_size):
    gen_label = np.random.choice(
        np.arange(n_y),
        size=(batch_size,),
        replace=True,
        p=cluster_sample_probs)
    y_gen_posterior_vals = np.zeros((batch_size, n_y))
    y_gen_posterior_vals[np.arange(batch_size), gen_label] = 1
    ### SHOULD GENERATE A DATASET WITH THE CORRECT TRANSFORMS
    gen_image = model.sample(y=y_gen_posterior_vals, mean=True)
    # gen_image = sess.run(gen_op, feed_dict={y_input: y_gen_posterior_vals})   # figure out what exactly gen op is 
    gen_buffer_images.append(gen_image)
    gen_buffer_labels.append(gen_label)

  gen_buffer_images = np.vstack(gen_buffer_images)
  gen_buffer_labels = np.concatenate(gen_buffer_labels)

  return gen_buffer_images, gen_buffer_labels


def get_encoder_params(n_y, model
                       ):
  """Set up ops to move / copy mixture component weights for dynamic expansion.

  Args:
    n_y: int, dimensionality of discrete latent variable y.

  Returns:
    A dict containing all of the ops required for dynamic updating.

  """
  # Set up graph ops to dynamically modify component params.
  # graph = tf.get_default_graph()

  # 1) Ops to get and set latent encoder params (entire tensors)
  latent_enc_tensors = {}
  named_parameters = model.named_parameters()
  named_parameters = dict(named_parameters)
  for k in range(n_y):
    latent_enc_tensors['latent_w_' + str(k)] = named_parameters[
        '_latent_encoder.mlp_latent_encoder.{}.weight'.format(k)]
    latent_enc_tensors['latent_b_' + str(k)] = named_parameters[
        '_latent_encoder.mlp_latent_encoder.{}.bias'.format(k)]

  cluster_w = named_parameters[
      '_cluster_encoder.mlp_cluster_encoder_final.weight']
  cluster_b = named_parameters[
      '_cluster_encoder.mlp_cluster_encoder_final.bias']


  latent_prior_mu_w = named_parameters[
      '_latent_decoder.latent_prior_mu.weight']
  latent_prior_sigma_w = named_parameters[
      '_latent_decoder.latent_prior_sigma.bias']

  encoder_params = {
      'latent_enc_tensors': latent_enc_tensors,
      'cluster_w': cluster_w,
      'cluster_b': cluster_b,
      'latent_prior_mu_w': latent_prior_mu_w,
      'latent_prior_sigma_w': latent_prior_sigma_w
  }

  return encoder_params





def copy_component_params(ind_from, ind_to, cluster_w, cluster_b, 
                          latent_enc_tensors, latent_prior_sigma_w, 
                          latent_prior_mu_w):
  """Copy parameters from component i to component j.

  Args:
    ind_from: int, component index to copy from.
    ind_to: int, component index to copy to.
    sess: tf.Session.
    ind_from_ph: tf placeholder for component to copy from.
    ind_to_ph: tf placeholder for component to copy to.
    latent_enc_tensors: dict, tensors in the latent posterior encoder.
    latent_enc_assign_ops: dict, assignment ops for latent posterior encoder.
    latent_enc_phs: dict, placeholders for assignment ops.
    cluster_w_update_op: op for updating weights of cluster encoder.
    cluster_b_update_op: op for updating biased of cluster encoder.
    mu_update_op: op for updating mu weights of latent prior.
    sigma_update_op: op for updating sigma weights of latent prior.

  """

  # Copy for latent encoder.
  new_w_val, new_b_val = [latent_enc_tensors['latent_w_' + str(ind_from)],latent_enc_tensors['latent_b_' + str(ind_from)]]
  with torch.no_grad(): 
    latent_enc_tensors['latent_w_' + str(ind_to)] = latent_enc_tensors['latent_w_' + str(ind_to)].copy_(new_w_val)
    latent_enc_tensors['latent_b_' + str(ind_to)] = latent_enc_tensors['latent_w_' + str(ind_to)].copy_(new_b_val)


  # Copy for cluster encoder softmax.
  w_indices = torch.stack([
          torch.range(cluster_w.shape[0], dtype=torch.int32),
          ind_to * torch.ones(shape=(cluster_w.shape[0],), dtype=torch.int32)
      ])   # shape cluster_w.shape[0], 2
  b_indices = ind_to

  cluster_w_updates = torch.squeeze(cluster_w[:, ind_from].unsqueeze(1)) # shape cluster_w.shape[0], 1
  cluster_b_updates = cluster_b[ind_from]
  with torch.no_grad():  # Disable gradient tracking for inplace operations
    # For updating cluster_w
    cluster_w[w_indices[0], w_indices[1]] = cluster_w_updates

    # For updating cluster_b
    cluster_b[b_indices] = cluster_b_updates

  mu_indices = torch.stack([
          ind_to * torch.ones(shape=(latent_prior_mu_w.shape[1],), dtype=torch.int32),
          torch.arange(latent_prior_mu_w.shape[1], dtype=torch.int32)
      ])
  mu_updates = latent_prior_mu_w[ind_from, :]

  with torch.no_grad():  # Disable gradient tracking for inplace operations
    latent_prior_mu_w[mu_indices[0], mu_indices[1]] = mu_updates

  sigma_indices = torch.stack([
          ind_to *
          torch.ones(shape=(latent_prior_sigma_w.shape[1],), dtype=torch.int32),
          torch.arange(latent_prior_sigma_w.shape[1], dtype=torch.int32)
      ])
  sigma_updates = latent_prior_sigma_w[ind_from, :]
  with torch.no_grad():  # Disable gradient tracking for inplace operations
    latent_prior_sigma_w[sigma_indices[0], sigma_indices[1]] = sigma_updates



def run_training(
    dataset,
    training_data_type,
    n_concurrent_classes,
    blend_classes,
    train_supervised,
    n_epochs,
    random_seed,
    lr_init,
    lr_factor,
    lr_schedule,
    output_type,
    n_y,
    n_y_active,
    n_z,
    encoder_kwargs,
    decoder_kwargs,
    dynamic_expansion,
    ll_thresh,
    classify_with_samples,
    report_interval,
    knn_values,
    gen_replay_type,
    use_supervised_replay):
  """Run training script.

  Args:
    dataset: str, name of the dataset.
    training_data_type: str, type of training run ('iid' or 'sequential').
    n_concurrent_classes: int, # of classes seen at a time (ignored for 'iid').
    blend_classes: bool, whether to blend in samples from the next class.
    train_supervised: bool, whether to use supervision during training.
    n_epochs: int, number of total training steps.
    random_seed: int, seed for tf and numpy RNG.
    lr_init: float, initial learning rate.
    lr_factor: float, learning rate decay factor.
    lr_schedule: float, epochs at which the decay should be applied.
    output_type: str, output distribution (currently only 'bernoulli').
    n_y: int, maximum possible dimensionality of discrete latent variable y.
    n_y_active: int, starting dimensionality of discrete latent variable y.
    n_z: int, dimensionality of continuous latent variable z.
    encoder_kwargs: dict, parameters to specify encoder.
    decoder_kwargs: dict, parameters to specify decoder.
    dynamic_expansion: bool, whether to perform dynamic expansion.
    ll_thresh: float, log-likelihood threshold below which to keep poor samples.
    classify_with_samples: bool, whether to sample latents when classifying.
    report_interval: int, number of steps after which to evaluate and report.
    knn_values: list of ints, k values for different k-NN classifiers to run
    (values of 3, 5, and 10 were used in different parts of the paper).
    gen_replay_type: str, 'fixed', 'dynamic', or None.
    use_supervised_replay: str, whether to use supervised replay (aka 'SMGR').
  """

  # Set tf random seed.
  
  args = get_args()
  print_args(args)
  torch.manual_seed(random_seed)
  np.set_printoptions(precision=2, suppress=True)

  # First set up the data source(s) and get dataset info.
  if dataset == 'mnist':
    batch_size = 100
    test_batch_size = 1000
  elif dataset == 'omniglot':
    batch_size = 15
    test_batch_size = 1318
  else:
    raise NotImplementedError

  # dataset_ops = get_data_sources(dataset, dataset_kwargs, batch_size,
  #                                test_batch_size, training_data_type,
  #                                n_concurrent_classes, image_key, label_key)
  dataset_tuple = load_data(dataset, training_data_type, n_concurrent_classes, batch_size, test_batch_size )

  train_dataset_list = dataset_tuple.train_dataset_list
  train_dataloader_for_clf = dataset_tuple.train_dataloader_for_clf
  valid_dataset = dataset_tuple.valid_dataset
  test_dataset = dataset_tuple.test_dataset

  output_shape = dataset_tuple.output_shape
  n_x = np.prod(output_shape)
  n_classes = dataset_tuple.n_classes
  num_train_examples = dataset_tuple.num_train_examples

  # Check that the number of classes is compatible with the training scenario
  assert n_classes % n_concurrent_classes == 0
  assert n_epochs % (n_classes / n_concurrent_classes) == 0

  # Set specific params depending on the type of gen replay
  if gen_replay_type == 'fixed':

    # data period seems to be the number of epochs belonging to one concurrent set of classes
    data_period = data_period = int(n_epochs /
                                    (n_classes / n_concurrent_classes))
    gen_every_n = 2  # Blend in a gen replay batch every 2 steps
    gen_refresh_period = data_period  # How often to refresh the batches of
    # generated data (equivalent to snapshotting a generative model)
    gen_refresh_on_expansion = False  # Don't refresh on dyn expansion
  elif gen_replay_type == 'dynamic':
    gen_every_n = 2  # Blend in a gen replay batch every 2 steps
    gen_refresh_period = 1e8  # Never refresh generated data periodically
    gen_refresh_on_expansion = True  # Refresh on dyn expansion instead
  elif gen_replay_type is None:
    gen_every_n = 0  # Don't use any gen replay batches
    gen_refresh_period = 1e8  # Never refresh generated data periodically
    gen_refresh_on_expansion = False  # Don't refresh on dyn expansion
  else:
    raise NotImplementedError

  max_gen_batches = 5000  # Max num of gen batches (proxy for storing a model)

  # Set dynamic expansion parameters
  exp_wait_epochs = 100  # Steps to wait after expansion before eligible again
  exp_burn_in = 100  # Steps to wait at start of learning before eligible
  exp_buffer_size = 100  # Size of the buffer of poorly explained data
  num_buffer_train_epochs = 10  # Num steps to train component on buffer

  # Define a global tf variable for the number of active components.
  # n_y_active = torch.tensor(n_y_active, dtype=torch.int32, requires_grad=False)

  logging.info('Starting CURL script on %s data.', dataset)

  # Set up placeholders for training.

  # x_train_raw = tfc.placeholder(
  #     dtype=tf.float32, shape=(batch_size,) + output_shape)
  # label_train = tfc.placeholder(dtype=tf.int32, shape=(batch_size,))



  # if dataset == 'mnist':
  #   x_train = binarize_fn(x_train_raw)
  #   x_valid = binarize_fn(valid_data[image_key]) if valid_data else None
  #   x_test = binarize_fn(test_data[image_key])
  #   x_train_for_clf = binarize_fn(train_data_for_clf[image_key])
  # elif 'cifar' in dataset or dataset == 'omniglot':
  #   x_train = x_train_raw
  #   x_valid = valid_data[image_key] if valid_data else None
  #   x_test = test_data[image_key]
  #   x_train_for_clf = train_data_for_clf[image_key]
  # else:
  #   raise ValueError('Unknown dataset {}'.format(dataset))

  # x_train, _ = train_dataloader.next()
  # x_valid, label_valid = valid_dataloader.next()
  # x_test, label_test = test_dataloader.next()
  # x_train_for_clf, _ = train_dataloader_for_clf.next()


  # label_valid = valid_data[label_key] if valid_data else None
  # label_test = test_data[label_key]

  
  # Uniform prior over y.
  prior_train_probs = utils.construct_prior_probs(batch_size, n_y, n_y_active)
    
  prior_train = distributions.OneHotCategorical(prior_train_probs)
  
  prior_test_probs = utils.construct_prior_probs(test_batch_size, n_y,
                                                 n_y_active)
  prior_test = distributions.OneHotCategorical(prior_test_probs, "prior_unconditional_test")

  CurlModel = model.Curl(
      prior_train,
      prior_test,
      output_type,
      output_shape,
      n_y,
      n_x,
      n_z,
      n_y_active,
      encoder_kwargs,
      decoder_kwargs,
      kly_over_batch=False,
      is_training=True,
      )
  


  # hiddens_for_clf = model_eval.get_shared_rep(x_train_for_clf,
  #                                             is_training=False)
  # cat_for_clf = model_eval.infer_cluster(hiddens_for_clf)

  # if classify_with_samples:
  #   latents_for_clf = model_eval.infer_latent(
  #       hiddens=hiddens_for_clf, y=cat_for_clf.sample().float()).sample()
  # else:
  #   latents_for_clf = model_eval.infer_latent(
  #       hiddens=hiddens_for_clf, y=cat_for_clf.mode().float()).mean()

  # This does val forward and computes val loss,
    # should use the same model, just prob turn to eval mode
    # 
  # if valid_dataloader is not None:
  #   valid_ops = setup_training_and_eval_graphs(
  #       x_valid,
  #       label_valid,
  #       y_valid,
  #       n_y,
  #       model_eval,
  #       classify_with_samples,
  #       is_training=False,
  #       name='valid')

  # does forward and then computes loss for test data
    # use the same model just turn on eval mode
  # test_ops = setup_training_and_eval_graphs(
  #     x_test,
  #     label_test,
  #     y_test,
  #     n_y,
  #     model_eval,
  #     classify_with_samples,
  #     is_training=False,
  #     name='test')


  # lr schedule gives epochs at which decay should happen
  
  # Set up optimizer (with scheduler).

  expansion_params = []
  for prefix in ["_cluster_encoder", "_latent_encoder", "_latent_decoder"]:
    curl_model_named_params = CurlModel.named_parameters(prefix = prefix)
    for _, param in curl_model_named_params:
      expansion_params.append(param)
  

  # will need to set up the optimizer with all of the correct things to track
  optimizer = optim.Adam(CurlModel.parameters(), lr=lr_init)

  ####will need to pass in only the expansion params
  expansion_optimizer = optim.Adam(expansion_params, lr = lr_init)
  
  scheduler = optim.lr_scheduler.MultiStepLR(optimizer, lr_schedule, gamma = lr_factor)




  # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
  #   train_step = optimizer.minimize(train_ops.elbo)
  #   train_step_supervised = optimizer.minimize(train_ops.elbo_supervised)

  #   # For dynamic expansion, we want to train only new-component-related params
  #   cat_params = tf.get_collection(
  #       tf.GraphKeys.TRAINABLE_VARIABLES,
  #       'cluster_encoder/mlp_cluster_encoder_final')
  #   component_params = tf.get_collection(
    #     tf.GraphKeys.TRAINABLE_VARIABLES,
    #     'latent_encoder/mlp_latent_encoder_*')
    # prior_params = tf.get_collection(
    #     tf.GraphKeys.TRAINABLE_VARIABLES,
    #     'latent_decoder/latent_prior*')

    # train_step_expansion = optimizer.minimize(
    #     train_ops.elbo_supervised,
    #     var_list=cat_params+component_params+prior_params)

  # Set up ops for generative replay
  if gen_every_n > 0:
    # How many generative batches will we use each period?
    gen_buffer_size = min(
        int(gen_refresh_period / gen_every_n), max_gen_batches)

    # Class each sample should be drawn from (default to uniform prior)
    y_gen = distributions.OneHotCategorical(
        probs=np.ones((batch_size, n_y)) / n_y,).sample()

    # ### SHOULD GENERATE A DATASET WITH THE CORRECT TRANSFORMS
    # gen_samples = CurlModel.sample(y=y_gen, mean=True)
    # gen_samples = list(torch.unbind(gen_samples, dim = 0))
    # gen_samples_targets = [None] * len(gen_samples)
    # gen_samples = GeneralImgDataset(gen_samples, gen_samples_targets, transform = binarize_transform, rgb = args.rgb)



    # if dataset == 'mnist' or dataset == 'omniglot':  pass
    #   gen_samples = binarize_fn(gen_samples)

  # Set up ops to dynamically modify parameters (for dynamic expansion)
  # encoder_params = get_encoder_params(n_y, CurlModel)

  logging.info('Created computation graph.')

  n_epochs_per_class = n_epochs / n_classes  # pylint: disable=invalid-name

  cumulative_component_counts = np.array([0] * n_y).astype(float)
  recent_component_counts = np.array([0] * n_y).astype(float)



  # Buffer of poorly explained data (if we're doing dynamic expansion).
  poor_data_buffer = []
  poor_data_labels = []
  all_full_poor_data_buffers = []
  all_full_poor_data_labels = []
  has_expanded = False
  epochs_since_expansion = 0
  gen_buffer_ind = 0
  eligible_for_expansion = False  # Flag to ensure we wait a bit after expansion

  # Set up basic ops to run and quantities to log. These are no longer ops but the actual values
  # ops_to_run = {
  #     'train_ELBO': train_ops.elbo,
  #     'train_log_p_x': train_ops.log_p_x,
  #     'train_kl_y': train_ops.kl_y,
  #     'train_kl_z': train_ops.kl_z,
  #     'train_ll': train_ops.ll,
  #     'train_batch_purity': train_ops.purity,
  #     'train_probs': train_ops.cat_probs,
  #     'n_y_active': n_y_active
  # }
  # if valid_dataloader is not None:
  #   valid_ops_to_run = {
  #       'valid_ELBO': valid_ops.elbo,
  #       'valid_kl_y': valid_ops.kl_y,
  #       'valid_kl_z': valid_ops.kl_z,
  #       'valid_confusion': valid_ops.confusion
  #   }
  # else:
  #   valid_ops_to_run = {}
  # test_ops_to_run = {
  #     'test_ELBO': test_ops.elbo,
  #     'test_kl_y': test_ops.kl_y,
  #     'test_kl_z': test_ops.kl_z,
  #     'test_confusion': test_ops.confusion
  # }
  # to_log = ['train_batch_purity']
  # to_log_eval = ['test_purity', 'test_ELBO', 'test_kl_y', 'test_kl_z']
  # if valid_dataloader is not None:
  #   to_log_eval += ['valid_ELBO', 'valid_purity']

  # if train_supervised:
  #   Loss_supervised = True
    # Track supervised losses, train on supervised loss.
  #   ops_to_run.update({
  #       'train_ELBO_supervised': train_ops.elbo_supervised,
  #       'train_log_p_x_supervised': train_ops.log_p_x_supervised,
  #       'train_kl_y_supervised': train_ops.kl_y_supervised,
  #       'train_kl_z_supervised': train_ops.kl_z_supervised,
  #       'train_ll_supervised': train_ops.ll_supervised
  #   })
  #   default_train_step = train_step_supervised
  #   to_log += [
  #       'train_ELBO_supervised', 'train_log_p_x_supervised',
  #       'train_kl_y_supervised', 'train_kl_z_supervised'
  #   ]
  # else:
  #   # Track unsupervised losses, train on unsupervised loss.
  #   ops_to_run.update({
  #       'train_ELBO': train_ops.elbo,
  #       'train_kl_y': train_ops.kl_y,
  #       'train_kl_z': train_ops.kl_z,
  #       'train_ll': train_ops.ll
  #   })
  #   default_train_step = train_step
  #   to_log += ['train_ELBO', 'train_kl_y', 'train_kl_z']

  # with tf.train.SingularMonitoredSession() as sess:

  for epoch in range(n_epochs):
    # feed_dict = {}

    # # Use the default training loss, but vary it each step depending on the
    # # training scenario (eg. for supervised gen replay, we alternate losses)
    # ops_to_run['train_step'] = default_train_step

    ### 1) PERIODICALLY TAKE SNAPSHOTS FOR GENERATIVE REPLAY ###
    if (gen_refresh_period and epoch % gen_refresh_period == 0 and
        gen_every_n > 0):

      # First, increment cumulative count and reset recent probs count.
      cumulative_component_counts += recent_component_counts
      recent_component_counts = np.zeros(n_y)

      # Generate enough samples for the rest of the next period
      # (Functionally equivalent to storing and sampling from the model).
      gen_buffer_images, gen_buffer_labels = get_generated_data(
      model = CurlModel,
      y_input=y_gen,
      gen_buffer_size=gen_buffer_size,
      component_counts=cumulative_component_counts)

    ### 2) DECIDE WHICH DATA SOURCE TO USE (GENERATIVE OR REAL DATA) ###
    periodic_refresh_started = (
        gen_refresh_period and epoch >= gen_refresh_period)
    refresh_on_expansion_started = (gen_refresh_on_expansion and has_expanded)
    if ((periodic_refresh_started or refresh_on_expansion_started) and
        gen_every_n > 0 and epoch % gen_every_n == 1):
      # Use generated data for the training batch
      used_real_data = False

      s = gen_buffer_ind * batch_size
      e = (gen_buffer_ind + 1) * batch_size

      gen_data_array = {
          'image': gen_buffer_images[s:e],
          'label': gen_buffer_labels[s:e]
      }
      gen_buffer_ind = (gen_buffer_ind + 1) % gen_buffer_size

      # Feed it as x_train because it's already reshaped and binarized.
      # feed_dict.update({
      #     x_train: gen_data_array['image'],
      #     label_train: gen_data_array['label']
      # })

      binarize_transform = transforms.compose([transforms.ToTensor(), BinarizeTransform()])
      train_dataset = GeneralImgDataset(gen_data_array['image'], gen_data_array['label'], binarize_transform, rgb = args.rgb)

      if use_supervised_replay:
        # Convert label to one-hot before feeding in.
        gen_label_onehot = np.eye(n_y)[gen_data_array['label']]
        CurlModel.y_label = gen_label_onehot
        # feed_dict.update({model_train.y_label: gen_label_onehot})
        # ops_to_run['train_step'] = train_step_supervised
        Loss_supervised = True

    else:
      # Else use the standard training data sources.
      used_real_data = True

      # Select appropriate data source for iid or sequential setup.
      if training_data_type == 'sequential':
        current_data_period = int(
            min(epoch / n_epochs_per_class, len(train_dataset_list) - 1))

        # If training supervised, set n_y_active directly based on how many
        # classes have been seen
        if train_supervised:
          assert not dynamic_expansion
          n_y_active = n_concurrent_classes * (
              current_data_period // n_concurrent_classes +1)
          # n_y_active.load(n_y_active_np, sess)
          # n_y_active = n_y_active_np.detach().clone()

        train_dataset = train_dataset_list[current_data_period]

        # If we are blending classes, figure out where we are in the data
        # period and add some fraction of other samples.
        if blend_classes:
          # If in the first quarter, blend in examples from the previous class
          if (epoch % n_epochs_per_class < n_epochs_per_class / 4 and
              current_data_period > 0):
            other_train_dataset = train_dataset_list[current_data_period -1]

            num_other = int(
                (n_epochs_per_class / 2 - 2 *
                  (epoch % n_epochs_per_class)) * batch_size / n_epochs_per_class)
            other_inds = np.random.permutation(batch_size)[:num_other]

            train_dataset.data[:num_other] = other_train_dataset.data[other_inds]
            train_dataset.targets[:num_other] = other_train_dataset.data[other_inds]
         
          # If in the last quarter, blend in examples from the next class
          elif (epoch % n_epochs_per_class > 3 * n_epochs_per_class / 4 and
                current_data_period < n_classes - 1):
            other_train_dataset = train_dataset_list[current_data_period  + 1]

            num_other = int(
                (2 * (epoch % n_epochs_per_class) - 3 * n_epochs_per_class / 2) *
                batch_size / n_epochs_per_class)
            other_inds = np.random.permutation(batch_size)[:num_other]

            train_dataset.data[:num_other] = other_train_dataset.data[other_inds]
            train_dataset.targets[:num_other] = other_train_dataset.data[other_inds]

          # Otherwise, just use the current class

      else:
        train_dataset = train_dataset_list[0]

      # feed_dict.update({
      #     x_train_raw: train_data_array[image_key],
      #     label_train: train_data_array[label_key]
      # })
        train_dataset = GeneralImgDataset(train_dataset.data, train_dataset.targets, binarize_transform, rgb = args.rgb)

    ### 3) PERFORM A GRADIENT STEP ###
    y_train = train_dataset.targets if train_supervised else None
    train_results = forward_pass(
    train_dataset.data,
    train_dataset.targets,
    y_train,
    n_y,
    args.rgb, 
    batch_size,
    CurlModel,
    train_supervised,
    classify_with_samples,
    is_training=True, optimizer = optimizer)

    

    
    # del train_results['train_step']

    ### 4) COMPUTE ADDITIONAL DIAGNOSTIC OPS ON VALIDATION/TEST SETS. ###
    if (epoch+1) % report_interval == 0:
      if valid_dataset is not None:
        
        logging.info('Evaluating on validation and test set!')
        # proc_ops = {
        #     k: (np.sum if 'confusion' in k
        #         else np.mean) for k in valid_ops_to_run
        # }
        # results.update(
        #     process_dataset(
        #         dataset_ops.valid_iter,
        #         valid_ops_to_run,
        #         sess,
        #         feed_dict=feed_dict,
        #         processing_ops=proc_ops))
        # results['valid_purity'] = compute_purity(results['valid_confusion'])


        # need to collect KL y, elbo, KL z, confusion
        y_valid = valid_dataset.targets if train_supervised else None
        valid_results = forward_pass(valid_dataset.data,
        valid_dataset.targets,
        y_valid,
        n_y,
        args.rgb,
        test_batch_size,
        CurlModel,
        classify_with_samples,
        is_training=False)
      # else:
        # logging.info('Evaluating on test set!')
        # proc_ops = {
        #     k: (np.sum if 'confusion' in k
        #         else np.mean) for k in test_ops_to_run
        # }
      y_test = test_dataset.targets if train_supervised else None
      test_results = forward_pass(test_dataset.data,
      test_dataset.targets,
      y_test,
      n_y,
      args.rgb, 
      batch_size,
      CurlModel,
      classify_with_samples,
      is_training=False)
      # results.update(process_dataset(dataset_ops.test_iter,
      #                                 test_ops_to_run,
      #                                 sess,
      #                                 feed_dict=feed_dict,
      #                                 processing_ops=proc_ops))
      # results['test_purity'] = compute_purity(results['test_confusion'])
      # curr_to_log = to_log + to_log_eval
    # else:
      # curr_to_log = list(to_log)  # copy to prevent in-place modifications

    ### 5) DYNAMIC EXPANSION ###
    if dynamic_expansion and used_real_data:
      # If we're doing dynamic expansion and below max capacity then add
      # poorly defined data points to a buffer.

      # First check whether the model is eligible for expansion (the model
      # becomes ineligible for a fixed time after each expansion, and when
      # it has hit max capacity).
      if (epochs_since_expansion >= exp_wait_epochs and epoch >= exp_burn_in and
          n_y_active < n_y):
        eligible_for_expansion = True

      steps_since_expansion += 1

      if eligible_for_expansion:
        # Add poorly explained data samples to a buffer.
        poor_inds = train_results.ll < ll_thresh
        poor_data_buffer.extend(list(np.array(train_dataset.data)[poor_inds]))
        poor_data_labels.extend(list(np.array(train_dataset.targets)[poor_inds]))

        n_poor_data = len(poor_data_buffer)

        # If buffer is big enough, then add a new component and train just the
        # new component with several steps of gradient descent.
        # (We just feed in a onehot cluster vector to indicate which
        # component).
        if n_poor_data >= exp_buffer_size:
          # Dump the buffers so we can log them.
          all_full_poor_data_buffers.append(poor_data_buffer)
          all_full_poor_data_labels.append(poor_data_labels)

          # Take a new generative snapshot if specified.
          if gen_refresh_on_expansion and gen_every_n > 0:
            # Increment cumulative count and reset recent probs count.
            cumulative_component_counts += recent_component_counts
            recent_component_counts = np.zeros(n_y)

            gen_buffer_images, gen_buffer_labels = get_generated_data(
            model = CurlModel,
            y_input=y_gen,
            gen_buffer_size=gen_buffer_size,
            component_counts=cumulative_component_counts)

          # Cull to a multiple of batch_size (keep the later data samples).
          poor_batches_dataset = GeneralImgDataset(poor_data_buffer, poor_data_labels,binarize_transform, rgb = args.rgb )
          # n_poor_batches = int(n_poor_data / batch_size)
          # poor_data_buffer = poor_data_buffer[-(n_poor_batches * batch_size):]
          # poor_data_labels = poor_data_labels[-(n_poor_batches * batch_size):]
          poor_batch_dataloader = DataLoader(poor_batches_dataset, batch_size = batch_size)

          # Find most probable component (on poor batch).
          poor_cprobs = []
          CurlModel.eval()
          with torch.no_grad():
            for x , _ in poor_batch_dataloader:
                hiddens = CurlModel.get_shared_rep(x)
                cat = CurlModel.infer_cluster(hiddens)
                cat_probs = cat.probs
                poor_cprobs.append(cat_probs)
          CurlModel.train()
          best_cluster = np.argmax(np.sum(np.vstack(poor_cprobs), axis=0))

          # Initialize parameters of the new component from most prob
          # existing.
          new_cluster = n_y_active

          encoder_params = get_encoder_params(n_y, CurlModel)
         
          copy_component_params(best_cluster, new_cluster, encoder_params['cluster_w'],
                                encoder_params['cluster_b'], encoder_params['latent_enc_tensors'],
                                encoder_params['latent_prior_sigma_w'], encoder_params['latent_prior_mu_w']
                                )

          # Increment mixture component count n_y_active.
          n_y_active += 1
          # n_y_active.load(n_y_active_np, sess)

          # Perform a number of steps of gradient descent on the data buffer,
          # training only the new component (supervised loss).
          for _ in range(num_buffer_train_epochs):
            for x,y in poor_batch_dataloader:
              label_onehot_batch = np.eye(n_y)[y]
              _, _, _, log_p_x_supervised, kl_y_supervised, kl_z_supervised = CurlModel(x, label_onehot_batch)
              ll_supervised = log_p_x_supervised - kl_y_supervised - kl_z_supervised
              elbo_supervised = torch.mean(ll_supervised)
              expansion_optimizer.zero_grad()
              elbo_supervised.backward()
              expansion_optimizer.step()





            # for bs in range(n_poor_batches):
            #   x_batch = poor_data_buffer[bs * batch_size:(bs + 1) *
            #                               batch_size]
            #   label_batch = [new_cluster] * batch_size
            #   label_onehot_batch = np.eye(n_y)[label_batch]
            #   _ = sess.run(
            #       train_step_expansion,
            #       feed_dict={
            #           x_train_raw: x_batch,
            #           model_train.y_label: label_onehot_batch
            #       })

          # Empty the buffer.
          poor_data_buffer = []
          poor_data_labels = []

          # Reset the threshold flag so we have a burn in before the next
          # component.
          eligible_for_expansion = False
          has_expanded = True
          steps_since_expansion = 0

    # Accumulate counts.
    if used_real_data:
      train_cat_probs_vals = train_results.cat_probs
      recent_component_counts += np.sum(
          train_cat_probs_vals, axis=0).astype(float)

    ### 6) LOGGING AND EVALUATION ###
    # cleanup_for_print = lambda x: ', {}: %.{}f'.format(
    #     x.capitalize().replace('_', ' '), 3)
    # log_str = 'Iteration %d'
    # log_str += ''.join([cleanup_for_print(el) for el in curr_to_log])
    # log_str += ' n_active: %d'
    # logging.info(
    #     log_str,
    #     *([step] + [results[el] for el in curr_to_log] + [n_y_active_np]))

    # Periodically perform evaluation
    # if (step + 1) % report_interval == 0:

    #   # Report test purity and related measures
    #   logging.info(
    #       'Iteration %d, Test purity: %.3f, Test ELBO: %.3f, Test '
    #       'KLy: %.3f, Test KLz: %.3f', step, results['test_purity'],
    #       results['test_ELBO'], results['test_kl_y'], results['test_kl_z'])
    #   # Flush data only once in a while to allow buffering of data for more
    #   # efficient writes.
    #   results['all_full_poor_data_buffers'] = all_full_poor_data_buffers
    #   results['all_full_poor_data_labels'] = all_full_poor_data_labels
    #   logging.info('Also training a classifier in latent space')

    #   # Perform knn classification from latents, to evaluate discriminability.

    #   # Get and encode training and test datasets.
    #   clf_train_vals = process_dataset(
    #       dataset_ops.train_iter_for_clf, {
    #           'latents': latents_for_clf,
    #           'labels': train_data_for_clf[label_key]
    #       },
    #       sess,
    #       feed_dict,
    #       aggregation_ops=np.concatenate)
    #   clf_test_vals = process_dataset(
    #       dataset_ops.test_iter, {
    #           'latents': test_ops.latents,
    #           'labels': test_data[label_key]
    #       },
    #       sess,
    #       aggregation_ops=np.concatenate)

    #   # Perform knn classification.
    #   knn_models = []
    #   for nval in knn_values:
    #     # Fit training dataset.
    #     clf = neighbors.KNeighborsClassifier(n_neighbors=nval)
    #     clf.fit(clf_train_vals['latents'], clf_train_vals['labels'])
    #     knn_models.append(clf)

    #     results['train_' + str(nval) + 'nn_acc'] = clf.score(
    #         clf_train_vals['latents'], clf_train_vals['labels'])

    #     # Get test performance.
    #     results['test_' + str(nval) + 'nn_acc'] = clf.score(
    #         clf_test_vals['latents'], clf_test_vals['labels'])

    #     logging.info(
    #         'Iteration %d %d-NN classifier accuracies, Training: '
    #         '%.3f, Test: %.3f', step, nval,
    #         results['train_' + str(nval) + 'nn_acc'],
    #         results['test_' + str(nval) + 'nn_acc'])
  scheduler.step()