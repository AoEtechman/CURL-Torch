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
"""Implementation of Continual Unsupervised Representation Learning model."""

from absl import logging
import numpy as np
# import sonnet as snt
# import tensorflow.compat.v1 as tf
# import tensorflow_probability as tfp

from layers import SharedConvModule, CustomBatchNorm1d
from utils import maybe_center_crop, generate_gaussian
import functools

# tfc = tf.compat.v1
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
import tensorflow as tf

# pylint: disable=g-long-lambda
# pylint: disable=redefined-outer-name

class SharedEncoder(nn.Module):
    """The shared encoder module, mapping input x to hiddens."""

    def __init__(self, encoder_type, n_enc, enc_strides):
        """The shared encoder function, mapping input x to hiddens.

        Args:
          encoder_type: str, type of encoder, either 'conv' or 'multi'
          n_enc: list, number of hidden units per layer in the encoder
          enc_strides: list, stride in each layer (only for 'conv' encoder_type)
        """
        super(SharedEncoder, self).__init__()
        self._encoder_type = encoder_type
        self.shared_encoder = None
        self.mlp_shared_encoder = None

        if encoder_type == 'conv':
            self.shared_encoder = SharedConvModule(
                filters=n_enc,
                strides=enc_strides,
                kernel_size=3,
                activation=nn.ReLU())
        elif encoder_type == 'multi':
            layers = [nn.Flatten()]
            for i in range(len(n_enc) - 1):
                layers.extend([
                    nn.Linear(in_features=n_enc[i], out_features=n_enc[i + 1]),
                    nn.ReLU(),
                ])
            self.mlp_shared_encoder = nn.Sequential(*layers)
        else:
            raise ValueError('Unknown encoder_type {}'.format(encoder_type))

    def forward(self, x):
        if self._encoder_type == 'multi':
            return self.mlp_shared_encoder(x)
        else:
            output = self.shared_encoder(x)
            self.conv_shapes = self.shared_encoder.conv_shapes
            return output

# class SharedEncoder(snt.AbstractModule):
#   """The shared encoder module, mapping input x to hiddens."""

#   def __init__(self, encoder_type, n_enc, enc_strides, name='shared_encoder'):
#     """The shared encoder function, mapping input x to hiddens.

#     Args:
#       encoder_type: str, type of encoder, either 'conv' or 'multi'
#       n_enc: list, number of hidden units per layer in the encoder
#       enc_strides: list, stride in each layer (only for 'conv' encoder_type)
#       name: str, module name used for tf scope.
#     """
#     super(SharedEncoder, self).__init__(name=name)
#     self._encoder_type = encoder_type

#     if encoder_type == 'conv':
#       self.shared_encoder = layers.SharedConvModule(
#           filters=n_enc,
#           strides=enc_strides,
#           kernel_size=3,
#           activation=tf.nn.relu)
#     elif encoder_type == 'multi':
#       self.shared_encoder = snt.nets.MLP(
#           name='mlp_shared_encoder',
#           output_sizes=n_enc,
#           activation=tf.nn.relu,
#           activate_final=True)
#     else:
#       raise ValueError('Unknown encoder_type {}'.format(encoder_type))

#   def _build(self, x, is_training):
#     if self._encoder_type == 'multi':
#       self.conv_shapes = None
#       x = snt.BatchFlatten()(x)
#       return self.shared_encoder(x)
#     else:
#       output = self.shared_encoder(x)
#       self.conv_shapes = self.shared_encoder.conv_shapes
#       return output



class ClusterEncoder(nn.Module):
    def __init__(self, input_size, n_y_active, n_y):
        super(ClusterEncoder, self).__init__()
        self.mlp_cluster_encoder_final = nn.Linear(input_size, n_y)
        self.n_y_active = n_y_active
        self.n_y = n_y

    def forward(self, hiddens):
        logits = self.mlp_cluster_encoder_final(hiddens)

        # Only use the first n_y_active components, and set the remaining to zero.
        if self.n_y > 1:
            logits_active = logits[:, :self.n_y_active]
            probs = F.softmax(logits_active, dim=1)
            paddings1 = (0, 0)
            paddings2 = (0, self.n_y - self.n_y_active)
            probs = F.pad(probs, (paddings1, paddings2), value=0.0) + 1e-12
        else:
            probs = torch.ones_like(logits)

        return torch.distributions.OneHotCategorical(probs=probs)


# def cluster_encoder_fn(hiddens, n_y_active, n_y, is_training=True):
#   """The cluster encoder function, modelling q(y | x).

#   Args:
#     hiddens: The shared encoder activations, 2D `Tensor` of size `[B, ...]`.
#     n_y_active: Tensor, the number of active components.
#     n_y: int, number of maximum components allowed (used for tensor size)
#     is_training: Boolean, whether to build the training graph or an evaluation
#       graph.

#   Returns:
#     The distribution `q(y | x)`.
#   """
#   del is_training  # unused for now
#   with tf.control_dependencies([tfc.assert_rank(hiddens, 2)]):
#     lin = snt.Linear(n_y, name='mlp_cluster_encoder_final')
#     logits = lin(hiddens)

#   # Only use the first n_y_active components, and set the remaining to zero.
#   if n_y > 1:
#     probs = tf.nn.softmax(logits[:, :n_y_active])
#     logging.info('Cluster softmax active probs shape: %s', str(probs.shape))
#     paddings1 = tf.stack([tf.constant(0), tf.constant(0)], axis=0)
#     paddings2 = tf.stack([tf.constant(0), n_y - n_y_active], axis=0)
#     paddings = tf.stack([paddings1, paddings2], axis=1)
#     probs = tf.pad(probs, paddings) + 0.0 * logits + 1e-12
#   else:
#     probs = tf.ones_like(logits)
#   logging.info('Cluster softmax probs shape: %s', str(probs.shape))

#   return tfp.distributions.OneHotCategorical(probs=probs)

class LatentEncoder(nn.Module):
    def __init__(self, input_size, n_y, n_z):
      super(LatentEncoder, self).__init__()
      self.n_y = n_y
      self.n_z = n_z

      self.mlp_latent_encoder = nn.ModuleList(nn.Linear(input_size, 2 * n_z) for i in range(n_y))

    def forward(self, hiddens, y):
        batch_size = hiddens.size(0)

        # Logits for both mean and variance
        all_logits = []
        for k in range(self.n_y):
          logits = self.mlp_latent_encoder[k](hiddens)
          all_logits.append(logits)

        all_logits = torch.stack(all_logits)
        

        # Sum over cluster components.
        logits = torch.einsum('ij,jik->ik', y, all_logits)

        # Compute distribution from logits.
        return generate_gaussian(logits=logits, sigma_nonlin='softplus', sigma_param='var')


# def latent_encoder_fn(hiddens, y, n_y, n_z, is_training=True):
#   """The latent encoder function, modelling q(z | x, y).

#   Args:
#     hiddens: The shared encoder activations, 2D `Tensor` of size `[B, ...]`.
#     y: Categorical cluster variable, `Tensor` of size `[B, n_y]`.
#     n_y: int, number of dims of y.
#     n_z: int, number of dims of z.
#     is_training: Boolean, whether to build the training graph or an evaluation
#       graph.

#   Returns:
#     The Gaussian distribution `q(z | x, y)`.
#   """
#   del is_training  # unused for now

#   with tf.control_dependencies([tfc.assert_rank(hiddens, 2)]):
#     # Logits for both mean and variance
#     n_logits = 2 * n_z

#     all_logits = []
#     for k in range(n_y):
#       lin = snt.Linear(n_logits, name='mlp_latent_encoder_' + str(k))
#       all_logits.append(lin(hiddens))

#   # Sum over cluster components.
#   all_logits = tf.stack(all_logits)  # [n_y, B, n_logits]
#   logits = tf.einsum('ij,jik->ik', y, all_logits)

#   # Compute distribution from logits.
#   return utils.generate_gaussian(
#       logits=logits, sigma_nonlin='softplus', sigma_param='var')

class DataDecoder(nn.Module):
  def __init__(self, output_type, output_shape, decoder_type, n_dec, dec_up_strides, n_x, n_y, shared_encoder_conv_shapes=None):
      super(DataDecoder, self).__init__()
      
      self.output_type = output_type
      self.output_shape = output_shape
      self.decoder_type = decoder_type
      self.n_dec = n_dec
      self.dec_up_strides = dec_up_strides
      self.n_x = n_x
      self.n_y = n_y
      self.shared_encoder_conv_shapes = shared_encoder_conv_shapes
      
      if output_type == 'bernoulli':
          self.output_dist = lambda x: torch.distributions.Bernoulli(logits=x)
          self.n_out_factor = 1
          self.out_shape = list(output_shape)
      else:
          raise NotImplementedError

      if shared_encoder_conv_shapes is None and decoder_type == 'deconv':
          raise ValueError('Shared encoder does not contain conv_shapes.')

      if decoder_type == 'deconv':
          num_output_channels = output_shape[-1]
          self.conv_decoder = UpsampleModule(
              filters=n_dec,
              kernel_size=3,
              activation=nn.ReLU(),
              dec_up_strides=dec_up_strides,
              enc_conv_shapes=shared_encoder_conv_shapes,
              n_c=num_output_channels * self.n_out_factor,
              method=decoder_type, 
              test_local_stats=True
          )
      
      elif decoder_type == 'multi':
          self.mlp_decoding_list = nn.ModuleList()
          for k in range(n_y):
              layers = []
              for i in range(len(n_dec) - 1):
                  layers.append(nn.Linear(n_dec[i], n_dec[i + 1]))
                  layers.append(nn.ReLU())
              layers.append(nn.Linear(n_dec[-1], n_x * self.n_out_factor))
              mlp_decoding = nn.Sequential(*layers)
              self.mlp_decoding_list.append(mlp_decoding)
      
      elif decoder_type == 'single':
          layers = []
          for i in range(len(n_dec) - 1):
              layers.append(nn.Linear(n_dec[i], n_dec[i + 1]))
              layers.append(nn.ReLU())
          layers.append(nn.Linear(n_dec[-1], n_x * self.n_out_factor))
          self.mlp_decoding = nn.Sequential(*layers)
      else:
          raise ValueError('Unknown decoder_type {}'.format(decoder_type))

  def forward(self, z, y):
    if self.decoder_type == 'deconv':
        logits = self.conv_decoder(z)
        logits = logits.view(-1, *self.out_shape)
    
    elif self.decoder_type == 'multi':
        all_logits = []
        for k in range(self.n_y):
            logits = self.mlp_decoding_list[k](z)
            all_logits.append(logits)
        all_logits = torch.stack(all_logits)
        logits = torch.einsum('ij,jik->ik', y, all_logits)
        logits = logits.view(-1, *self.out_shape)
    
    elif self.decoder_type == 'single':
        logits = self.mlp_decoding(z)
        logits = logits.view(-1, *self.out_shape)
    
    else:
        raise ValueError('Unknown decoder_type {}'.format(self.decoder_type))
    
    return self.output_dist(logits)

# def data_decoder_fn(z,
#                     y,
#                     output_type,
#                     output_shape,
#                     decoder_type,
#                     n_dec,
#                     dec_up_strides,
#                     n_x,
#                     n_y,
#                     shared_encoder_conv_shapes=None,
#                     is_training=True,
#                     test_local_stats=True):
#   """The data decoder function, modelling p(x | z).

#   Args:
#     z: Latent variables, `Tensor` of size `[B, n_z]`.
#     y: Categorical cluster variable, `Tensor` of size `[B, n_y]`.
#     output_type: str, output distribution ('bernoulli' or 'quantized_normal').
#     output_shape: list, shape of output (not including batch dimension).
#     decoder_type: str, 'single', 'multi', or 'deconv'.
#     n_dec: list, number of hidden units per layer in the decoder
#     dec_up_strides: list, stride in each layer (only for 'deconv' decoder_type).
#     n_x: int, number of dims of x.
#     n_y: int, number of dims of y.
#     shared_encoder_conv_shapes: the shapes of the activations of the
#       intermediate layers of the encoder,
#     is_training: Boolean, whether to build the training graph or an evaluation
#       graph.
#     test_local_stats: Boolean, whether to use the test batch statistics at test
#       time for batch norm (default) or the moving averages.

#   Returns:
#     The Bernoulli distribution `p(x | z)`.
#   """

#   if output_type == 'bernoulli':
#     output_dist = lambda x: tfp.distributions.Bernoulli(logits=x)
#     n_out_factor = 1
#     out_shape = list(output_shape)
#   else:
#     raise NotImplementedError
#   if len(z.shape) != 2:
#     raise NotImplementedError('The data decoder function expects `z` to be '
#                               '2D, but its shape was %s instead.' %
#                               str(z.shape))
#   if len(y.shape) != 2:
#     raise NotImplementedError('The data decoder function expects `y` to be '
#                               '2D, but its shape was %s instead.' %
#                               str(y.shape))

#   # Upsample layer (deconvolutional, bilinear, ..).
#   if decoder_type == 'deconv':

#     # First, check that the encoder is convolutional too (needed for batchnorm)
#     if shared_encoder_conv_shapes is None:
#       raise ValueError('Shared encoder does not contain conv_shapes.')

#     num_output_channels = output_shape[-1]
#     conv_decoder = UpsampleModule(
#         filters=n_dec,
#         kernel_size=3,
#         activation=tf.nn.relu,
#         dec_up_strides=dec_up_strides,
#         enc_conv_shapes=shared_encoder_conv_shapes,
#         n_c=num_output_channels * n_out_factor,
#         method=decoder_type)
#     logits = conv_decoder(
#         z, is_training=is_training, test_local_stats=test_local_stats)
#     logits = tf.reshape(logits, [-1] + out_shape)  # n_out_factor in last dim

#   # Multiple MLP decoders, one for each component.
#   elif decoder_type == 'multi':
#     all_logits = []
#     for k in range(n_y):
#       mlp_decoding = snt.nets.MLP(
#           name='mlp_latent_decoder_' + str(k),
#           output_sizes=n_dec + [n_x * n_out_factor],
#           activation=tf.nn.relu,
#           activate_final=False)
#       logits = mlp_decoding(z)
#       all_logits.append(logits)

#     all_logits = tf.stack(all_logits)
#     logits = tf.einsum('ij,jik->ik', y, all_logits)
#     logits = tf.reshape(logits, [-1] + out_shape)  # Back to 4D

#   # Single (shared among components) MLP decoder.
#   elif decoder_type == 'single':
#     mlp_decoding = snt.nets.MLP(
#         name='mlp_latent_decoder',
#         output_sizes=n_dec + [n_x * n_out_factor],
#         activation=tf.nn.relu,
#         activate_final=False)
#     logits = mlp_decoding(z)
#     logits = tf.reshape(logits, [-1] + out_shape)  # Back to 4D
#   else:
#     raise ValueError('Unknown decoder_type {}'.format(decoder_type))

#   return output_dist(logits)

class LatentDecoder(nn.Module):
    def __init__(self, n_y, n_z):
        super(LatentDecoder, self).__init__()
        self.latent_prior_mu = nn.Linear(n_y, n_z)
        self.latent_prior_sigma = nn.Linear(n_y, n_z)

    def forward(self, y):
        if len(y.shape) != 2:
            raise NotImplementedError('The latent decoder function expects `y` to be '
                                      '2D, but its shape was {} instead.'.format(str(y.shape)))

        mu = self.latent_prior_mu(y)
        sigma = self.latent_prior_sigma(y)

        logits = torch.cat([mu, sigma], dim=1)

        # Assuming utils.generate_gaussian is a function generating Gaussian distribution
        return generate_gaussian(logits=logits, sigma_nonlin='softplus', sigma_param='var')

# def latent_decoder_fn(y, n_z, is_training=True):
#   """The latent decoder function, modelling p(z | y).

#   Args:
#     y: Categorical cluster variable, `Tensor` of size `[B, n_y]`.
#     n_z: int, number of dims of z.
#     is_training: Boolean, whether to build the training graph or an evaluation
#       graph.

#   Returns:
#     The Gaussian distribution `p(z | y)`.
#   """
#   del is_training  # Unused for now.
#   if len(y.shape) != 2:
#     raise NotImplementedError('The latent decoder function expects `y` to be '
#                               '2D, but its shape was %s instead.' %
#                               str(y.shape))

#   lin_mu = snt.Linear(n_z, name='latent_prior_mu')
#   lin_sigma = snt.Linear(n_z, name='latent_prior_sigma')

#   mu = lin_mu(y)
#   sigma = lin_sigma(y)

#   logits = tf.concat([mu, sigma], axis=1)

#   return utils.generate_gaussian(
#       logits=logits, sigma_nonlin='softplus', sigma_param='var')


class Curl(nn.Module):
  """CURL model class."""

  def __init__(self,
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
               ):
    self._shared_encoder = SharedEncoder(**encoder_kwargs)
    self._prior_train = prior_train
    self._prior_test = prior_test
    self._latent_decoder = functools.partial(LatentDecoder, n_z=n_z)
    encoder_conv_shapes = getattr(self._shared_encoder, 'conv_shapes', None)
    self._data_decoder =  functools.partial(
        DataDecoder,
        output_type=output_type,
        output_shape=output_shape,
        n_x=n_x,
        n_y=n_y,
        shared_encoder_conv_shapes = encoder_conv_shapes,
        **decoder_kwargs)
    self._cluster_encoder = functools.partial(ClusterEncoder, n_y_active=n_y_active, n_y=n_y)
    self._latent_encoder = functools.partial(LatentEncoder, n_y=n_y, n_z=n_z)
    self._n_y_active = n_y_active
    self._kly_over_batch = kly_over_batch
    self._is_training = is_training
    self._cache = {}

    
   

  def sample(self, sample_shape=(), y=None, mean=False):
    """Draws a sample from the learnt distribution p(x).

    Args:
      sample_shape: `int` or 0D `Tensor` giving the number of samples to return.
        If  empty tuple (default value), 1 sample will be returned.
      y: Optional, the one hot label on which to condition the sample.
      mean: Boolean, if True the expected value of the output distribution is
        returned, otherwise samples from the output distribution.

    Returns:
      Sample tensor of shape `[B * N, ...]` where `B` is the batch size of
      the prior, `N` is the number of samples requested, and `...` represents
      the shape of the observations.

    Raises:
      ValueError: If both `sample_shape` and `n` are provided.
      ValueError: If `sample_shape` has rank > 0 or if `sample_shape`
      is an int that is < 1.
    """
    with torch.no_grad():
      if y is None:
          y = self.compute_prior().sample(sample_shape).float()

      if y.dim() > 2:
          y = torch.flatten(y, start_dim=0, end_dim=-2)


      z = self._latent_decoder(y)
      if mean:
          samples = self.predict(z.sample(), y).mean(dim=0)
      else:
          samples = self.predict(z.sample(), y).sample()

    return samples

  def reconstruct(self, x, use_mode=True, use_mean=False):
    """Reconstructs the given observations.

    Args:
      x: Observed `Tensor`.
      use_mode: Boolean, if true, take the argmax over q(y|x)
      use_mean: Boolean, if true, use pixel-mean for reconstructions.

    Returns:
      The reconstructed samples x ~ p(x | y~q(y|x), z~q(z|x, y)).
    """

    hiddens = self._shared_encoder(x)
    qy = self.infer_cluster(hiddens)
    y_sample = qy.mode() if use_mode else qy.sample()
    y_sample = tf.to_float(y_sample)
    qz = self.infer_latent(hiddens, y_sample)
    p = self.predict(qz.sample(), y_sample)

    if use_mean:
      return p.mean()
    else:
      return p.sample()

  def log_prob(self, x):
    """Redirects to log_prob_elbo with a warning."""
    logging.warn('log_prob is actually a lower bound')
    return self.log_prob_elbo(x)

  def log_prob_elbo(self, x):
    """Returns evidence lower bound."""
    log_p_x, kl_y, kl_z = self.forward(x)[:3]
    return log_p_x - kl_y - kl_z

  def forward(self, x, y=None, reduce_op=torch.sum):
    """Returns the components used in calculating the evidence lower bound.

    Args:
      x: Observed variables, `Tensor` of size `[B, I]` where `I` is the size of
        a flattened input.
      y: Optional labels, `Tensor` of size `[B, I]` where `I` is the size of a
        flattened input.
      reduce_op: The op to use for reducing across non-batch dimensions.
        Typically either `tf.reduce_sum` or `tf.reduce_mean`.

    Returns:
      `log p(x|y,z)` of shape `[B]` where `B` is the batch size.
      `KL[q(y|x) || p(y)]` of shape `[B]` where `B` is the batch size.
      `KL[q(z|x,y) || p(z|y)]` of shape `[B]` where `B` is the batch size.
    """
    cache_key = (x,)

    # Checks if the output graph for this inputs has already been computed.
    if cache_key in self._cache:
      return self._cache[cache_key]

    # with tf.name_scope('{}_log_prob_elbo'.format(self.scope_name)):

    hiddens = self._shared_encoder(x)
    # 1) Compute KL[q(y|x) || p(y)] from x, and keep distribution q_y around
    kl_y, q_y = self._kl_and_qy(hiddens)  # [B], distribution

    # For the next two terms, we need to marginalise over all y.

    # First, construct every possible y indexing (as a one hot) and repeat it
    # for every element in the batch [n_y_active, B, n_y].
    # Note that the onehot have dimension of all y, while only the codes
    # corresponding to active components are instantiated
    bs, n_y = q_y.probs.shape
    all_y = torch.tile(
          torch.unsqueeze(F.one_hot(torch.arange(self._n_y_active), n_y), dim=1),
          (1, bs, 1))

    # 2) Compute KL[q(z|x,y) || p(z|y)] (for all possible y), and keep z's
    # around [n_y, B] and [n_y, B, n_z]
    for y in all_y:
      kl_z, z = self._kl_and_z(hiddens, y)
      kl_z_all.append(kl_z)
      z_all.append(z)
    kl_z_all = torch.stack(kl_z_all)
    z_all = torch.stack(z_all)
    kl_z_all = torch.transpose(kl_z_all)

    # Now take the expectation over y (scale by q(y|x))
    y_logits = q_y.logits[:, :self._n_y_active]  # [B, n_y]
    y_probs = q_y.probs[:, :self._n_y_active]  # [B, n_y]
    y_probs = y_probs / torch.sum(y_probs, dim=1, keepdim=True)
    kl_z = torch.sum(y_probs * kl_z_all, dim=1)

    # 3) Evaluate logp and recon, i.e., log and mean of p(x|z,[y])
    # (conditioning on y only in the `multi` decoder_type case, when
    # train_supervised is True). Here we take the reconstruction from each
    # possible component y and take its log prob. [n_y, B, Ix, Iy, Iz]
    log_p_x_all = []
    for y, z in zip(all_y, z_all):
      log_p_x_all.append(self.predict(z, y).log_prob(x))
    log_p_x_all = torch.stack(log_p_x_all)

    # Sum log probs over all dimensions apart from the first two (n_y, B),
    # i.e., over I.
    log_p_x_all = log_p_x_all.flatten(start_dim=2)  # [n_y, B, I]
    # Note, this is E_{q(y|x)} [ log p(x | z, y)], i.e., we scale log_p_x_all
    # by q(y|x).
    log_p_x = torch.einsum('ij,jik->ik', y_probs, log_p_x_all)  # [B, I]
    
    
    # We may also use a supervised loss for some samples [B, n_y]
    if y is not None:
      self.y_label = F.one_hot(y, n_y)
    else:
      self.y_label = torch.zeros((bs, n_y), dtype=torch.float32)


    # This is computing log p(x | z, y=true_y)], which is basically equivalent
    # to indexing into the correct element of `log_p_x_all`.
    log_p_x_sup = torch.einsum('ij,jik->ik',
                                self.y_label[:, :self._n_y_active],
                                log_p_x_all)  # [B, I]
    kl_z_sup = torch.einsum('ij,ij->i',
                              self.y_label[:, :self._n_y_active],
                              kl_z_all)  # [B]
    # -log q(y=y_true | x)
    kl_y_sup = F.cross_entropy(y_logits, torch.argmax(self.y_label[:, :self._n_y_active], dim=1))

    dims_x = tuple(range(1, log_p_x.dim()))
    log_p_x = reduce_op(log_p_x, dims_x)
    log_p_x_sup = reduce_op(log_p_x_sup, dims_x)

    # Store values needed externally
    self.q_y = q_y
    self.log_p_x_all = log_p_x_all.permute(1, 0, 2)
    self.kl_z_all = kl_z_all
    self.y_probs = y_probs

    self._cache[cache_key] = (log_p_x, kl_y, kl_z, log_p_x_sup, kl_y_sup,
                              kl_z_sup)
    return log_p_x, kl_y, kl_z, log_p_x_sup, kl_y_sup, kl_z_sup

  def _kl_and_qy(self, hiddens):
    """Returns analytical or sampled KL div and the distribution q(y | x).

    Args:
      hiddens: The shared encoder activations, 2D `Tensor` of size `[B, ...]`.

    Returns:
      Pair `(kl, y)`, where `kl` is the KL divergence (a `Tensor` with shape
      `[B]`, where `B` is the batch size), and `y` is a sample from the
      categorical encoding distribution.
    """
    if hiddens.ndim == 2:
      q = self.infer_cluster(hiddens)  # q(y|x)
    p = self.compute_prior()  # p(y)
    try:
      if self._kly_over_batch:
        # Take the average proportions over the whole batch then repeat it in each row
        # before computing the KL
        probs = q.probs.mean(dim=0, keepdim=True).repeat(q.probs.size(0), 1)
        qmean = torch.distributions.OneHotCategorical(probs=probs)
        kl = torch.distributions.kl.kl_divergence(qmean, p)
      else:
          kl = torch.distributions.kl.kl_divergence(q, p)
    except NotImplementedError:
      y = q.sample(name='y_sample')
      logging.warn('Using sampling KLD for y')
      log_p_y = p.log_prob(y)
      log_q_y = q.log_prob(y)

      # Reduce over all dimension except batch.
      sum_axis_p = [k for k in range(1, log_p_y.ndims)]
      log_p_y = torch.sum(log_p_y, sum_axis_p)
      sum_axis_q = [k for k in range(1, log_q_y.ndims)]
      log_q_y = torch.sum(log_q_y, sum_axis_q)

      kl = log_q_y - log_p_y

    # Reduce over all dimension except batch.
    sum_axis_kl = [k for k in range(1, kl.ndims)]
    kl = torch.sum(kl, sum_axis_kl)
    return kl, q

  def _kl_and_z(self, hiddens, y):
    """Returns KL[q(z|y,x) || p(z|y)] and a sample for z from q(z|y,x).

    Returns the analytical KL divergence KL[q(z|y,x) || p(z|y)] if one is
    available (as registered with `kullback_leibler.RegisterKL`), or a sampled
    KL divergence otherwise (in this case the returned sample is the one used
    for the KL divergence).

    Args:
      hiddens: The shared encoder activations, 2D `Tensor` of size `[B, ...]`.
      y: Categorical cluster random variable, `Tensor` of size `[B, n_y]`.

    Returns:
      Pair `(kl, z)`, where `kl` is the KL divergence (a `Tensor` with shape
      `[B]`, where `B` is the batch size), and `z` is a sample from the encoding
      distribution.
    """
    if hiddens.ndim == 2:
      q = self.infer_latent(hiddens, y)  # q(z|x,y)
    p = self.generate_latent(y)  # p(z|y)
    z = q.sample()
    try:
      kl = torch.distributions.kl.kl_divergence(q, p)
    except NotImplementedError:
      logging.warn('Using sampling KLD for z')
      log_p_z = p.log_prob(z)
      log_q_z = q.log_prob(z)

      # Reduce over all dimension except batch.
      sum_axis_p = [k for k in range(1, log_p_z.ndims)]
      log_p_z = torch.sum(log_p_z, sum_axis_p)
      sum_axis_q = [k for k in range(1, log_q_z.ndims)]
      log_q_z = torch.sum(log_q_z, sum_axis_q)

      kl = log_q_z - log_p_z

    # Reduce over all dimension except batch.
    sum_axis_kl = [k for k in range(1, kl.ndims)]
    kl = torch.sum(kl, sum_axis_kl, name='kl')
    return kl, z

  def infer_latent(self, hiddens, y=None, use_mean_y=False):
    """Performs inference over the latent variable z.

    Args:
      hiddens: The shared encoder activations, 4D `Tensor` of size `[B, ...]`.
      y: Categorical cluster variable, `Tensor` of size `[B, ...]`.
      use_mean_y: Boolean, whether to take the mean encoding over all y.

    Returns:
      The distribution `q(z|x, y)`, which on sample produces tensors of size
      `[N, B, ...]` where `B` is the batch size of `x` and `y`, and `N` is the
      number of samples and `...` represents the shape of the latent variables.
    """

    if hiddens.ndim == 2:
      if y is None:
        # Convert to float
        y = torch.tensor(self.infer_cluster(hiddens).mode()).float()

    if use_mean_y:
        # If use_mean_y, then y must be probabilities
        all_y = torch.tile(
            torch.unsqueeze(torch.eye(y.size(1)), dim=1),
            (1, y.size(0), 1)
        )

        # Compute z KL from x (for all possible y), and keep z's around
        z_all = torch.stack([
            self._latent_encoder(hiddens, y_elem).mean(dim=0)
            for y_elem in all_y
        ])

        return torch.einsum('ij,jik->ik', y, z_all)
    else:
        return self._latent_encoder(hiddens, y)


  def generate_latent(self, y):
    """Use the generative model to compute latent variable z, given a y.

    Args:
      y: Categorical cluster variable, `Tensor` of size `[B, ...]`.

    Returns:
      The distribution `p(z|y)`, which on sample produces tensors of size
      `[N, B, ...]` where `B` is the batch size of `x`, and `N` is the number of
      samples asked and `...` represents the shape of the latent variables.
    """
    return self._latent_decoder(y)

  def get_shared_rep(self, x):
    """Gets the shared representation from a given input x.

    Args:
      x: Observed variables, `Tensor` of size `[B, I]` where `I` is the size of
        a flattened input.
      is_training: bool, whether this constitutes training data or not.

    Returns:
      `log p(x|y,z)` of shape `[B]` where `B` is the batch size.
      `KL[q(y|x) || p(y)]` of shape `[B]` where `B` is the batch size.
      `KL[q(z|x,y) || p(z|y)]` of shape `[B]` where `B` is the batch size.
    """
    return self._shared_encoder(x)

  def infer_cluster(self, hiddens):
    """Performs inference over the categorical variable y.

    Args:
      hiddens: The shared encoder activations, 2D `Tensor` of size `[B, ...]`.

    Returns:
      The distribution `q(y|x)`, which on sample produces tensors of size
      `[N, B, ...]` where `B` is the batch size of `x`, and `N` is the number of
      samples asked and `...` represents the shape of the latent variables.
    """
    assert hiddens.ndim == 2
    return self._cluster_encoder(hiddens)

  def predict(self, z, y):
    """Computes prediction over the observed variables.

    Args:
      z: Latent variables, `Tensor` of size `[B, ...]`.
      y: Categorical cluster variable, `Tensor` of size `[B, ...]`.

    Returns:
      The distribution `p(x|z)`, which on sample produces tensors of size
      `[N, B, ...]` where `N` is the number of samples asked.
    """
    
    return self._data_decoder(
        z,
        y,
        )

  def compute_prior(self):
    """Computes prior over the latent variables.

    Returns:
      The distribution `p(y)`, which on sample produces tensors of size
      `[N, ...]` where `N` is the number of samples asked and `...` represents
      the shape of the latent variables.
    """

    if self.training:
      return self._prior_train
    else:
       return self._prior_test


# class UpsampleModule(nn.Module):
#   """Convolutional decoder.

#   If `method` is 'deconv' apply transposed convolutions with stride 2,
#   otherwise apply the `method` upsampling function and then smooth with a
#   stride 1x1 convolution.

#   Params:
#   -------
#   filters: list, where the first element is the number of filters of the initial
#     MLP layer and the remaining elements are the number of filters of the
#     upsampling layers.
#   kernel_size: the size of the convolutional kernels. The same size will be
#     used in all convolutions.
#   activation: an activation function, applied to all layers but the last.
#   dec_up_strides: list, the upsampling factors of each upsampling convolutional
#     layer.
#   enc_conv_shapes: list, the shapes of the input and of all the intermediate
#     feature maps of the convolutional layers in the encoder.
#   n_c: the number of output channels.
#   """

#   def __init__(self,
#                filters,
#                kernel_size,
#                activation,
#                dec_up_strides,
#                enc_conv_shapes,
#                n_c,
#                method='nn',
#                name='upsample_module'):
#     super(UpsampleModule, self).__init__(name=name)

#     assert len(filters) == len(dec_up_strides) + 1, (
#         'The decoder\'s filters should contain one element more than the '
#         'decoder\'s up stride list, but has %d elements instead of %d.\n'
#         'Decoder filters: %s\nDecoder up strides: %s' %
#         (len(filters), len(dec_up_strides) + 1, str(filters),
#          str(dec_up_strides)))

#     self._filters = filters
#     self._kernel_size = kernel_size
#     self._activation = activation

#     self._dec_up_strides = dec_up_strides
#     self._enc_conv_shapes = enc_conv_shapes
#     self._n_c = n_c
#     if method == 'deconv':
#       self._conv_layer = tf.layers.Conv2DTranspose
#       self._method = method
#     else:
#       self._conv_layer = tf.layers.Conv2D
#       self._method = getattr(tf.image.ResizeMethod, method.upper())
#     self._method_str = method.capitalize()

#   def _build(self, z, is_training=True, test_local_stats=True, use_bn=False):
#     batch_norm_args = {
#         'is_training': is_training,
#         'test_local_stats': test_local_stats
#     }

#     method = self._method
#     # Cycle over the encoder shapes backwards, to build a symmetrical decoder.
#     enc_conv_shapes = self._enc_conv_shapes[::-1]
#     strides = self._dec_up_strides
#     # We store the heights and widths of the encoder feature maps that are
#     # unique, i.e., the ones right after a layer with stride != 1. These will be
#     # used as a target to potentially crop the upsampled feature maps.
#     unique_hw = np.unique([(el[1], el[2]) for el in enc_conv_shapes], axis=0)
#     unique_hw = unique_hw.tolist()[::-1]
#     unique_hw.pop()  # Drop the initial shape

#     # The first filter is an MLP.
#     mlp_filter, conv_filters = self._filters[0], self._filters[1:]
#     # The first shape is used after the MLP to go to 4D.

#     layers = [z]
#     # The shape of the first enc is used after the MLP to go back to 4D.
#     dec_mlp = snt.nets.MLP(
#         name='dec_mlp_projection',
#         output_sizes=[mlp_filter, np.prod(enc_conv_shapes[0][1:])],
#         use_bias=not use_bn,
#         activation=self._activation,
#         activate_final=True)

#     upsample_mlp_flat = dec_mlp(z)
#     if use_bn:
#       upsample_mlp_flat = snt.BatchNorm(scale=True)(upsample_mlp_flat,
#                                                     **batch_norm_args)
#     layers.append(upsample_mlp_flat)
#     upsample = tf.reshape(upsample_mlp_flat, enc_conv_shapes[0])
#     layers.append(upsample)

#     for i, (filter_i, stride_i) in enumerate(zip(conv_filters, strides), 1):
#       if method != 'deconv' and stride_i > 1:
#         upsample = tf.image.resize_images(
#             upsample, [stride_i * el for el in upsample.shape.as_list()[1:3]],
#             method=method,
#             name='upsample_' + str(i))
#       upsample = self._conv_layer(
#           filters=filter_i,
#           kernel_size=self._kernel_size,
#           padding='same',
#           use_bias=not use_bn,
#           activation=self._activation,
#           strides=stride_i if method == 'deconv' else 1,
#           name='upsample_conv_' + str(i))(
#               upsample)
#       if use_bn:
#         upsample = snt.BatchNorm(scale=True)(upsample, **batch_norm_args)
#       if stride_i > 1:
#         hw = unique_hw.pop()
#         upsample = utils.maybe_center_crop(upsample, hw)
#       layers.append(upsample)

#     # Final layer, no upsampling.
#     x_logits = tf.layers.Conv2D(
#         filters=self._n_c,
#         kernel_size=self._kernel_size,
#         padding='same',
#         use_bias=not use_bn,
#         activation=None,
#         strides=1,
#         name='logits')(
#             upsample)
#     if use_bn:
#       x_logits = snt.BatchNorm(scale=True)(x_logits, **batch_norm_args)
#     layers.append(x_logits)

#     logging.info('%s upsampling module layer shapes', self._method_str)
#     logging.info('\n'.join([str(v.shape.as_list()) for v in layers]))

#     return x_logits



class UpsampleModule(nn.Module):
  def __init__(self, filters, kernel_size, activation, dec_up_strides, enc_conv_shapes, n_c, test_local_stats, use_bn = False, method=None):
      super(UpsampleModule, self).__init__()

      assert len(filters) == len(dec_up_strides) + 1, (
          'The decoder\'s filters should contain one element more than the '
          'decoder\'s up stride list, but has %d elements instead of %d.\n'
          'Decoder filters: %s\nDecoder up strides: %s' %
          (len(filters), len(dec_up_strides) + 1, str(filters),
          str(dec_up_strides)))

      self._filters = filters
      self._kernel_size = kernel_size
      self._activation = activation
      self._dec_up_strides = dec_up_strides
      self._enc_conv_shapes = enc_conv_shapes
      self._n_c = n_c
      self._method = method
      self._method_str = method.capitalize()
      self.use_bn = use_bn

      if method == 'deconv':
          self._conv_layer = nn.ConvTranspose2d
      else:
          self._method = method
          self._conv_layer = nn.Conv2d


      enc_conv_shapes = self._enc_conv_shapes[::-1]
      strides = self._dec_up_strides


      mlp_filter, conv_filters = self._filters[0], self._filters[1:]
      first_linear_output_size = np.prod(enc_conv_shapes[0][1:])

      # Define the MLP projection layer
      self.dec_mlp = nn.Sequential(
          nn.Linear(mlp_filter, first_linear_output_size, bias=not use_bn),
          self._activation
      )

      if use_bn:
        self.dec_mlp.append(CustomBatchNorm1d(first_linear_output_size, test_local_stats=test_local_stats))

      conv_layer_list = []

      input_channels = None
      for i, (filter_i, stride_i) in enumerate(zip(conv_filters, strides), 1):

        pad = ((filter_i - 1) * stride_i + kernel_size - input_channels) / 2
        if pad % 1 == 0:
            
            padding = (pad, pad, pad, pad)
        else:
            pad_low = np.floor(pad)
            pad_high = np.ceil(pad)
            padding = (pad_high, pad_low, pad_high, pad_low)

        conv_layer = self._conv_layer(
        in_channels=100,
        out_channels=input_channels,
        kernel_size=self._kernel_size,
        bias = not use_bn,
        stride = stride_i if method == 'deconv' else 1,
        padding= padding)
        input_channels = filter_i
        conv_layer_list.append(conv_layer)
        if use_bn:
            conv_layer_list.append(CustomBatchNorm1d(filter_i, test_local_stats=test_local_stats))
            
      self.upsample_one = nn.ModuleList(conv_layer_list)


      pad = ((self._n_c - 1) * stride_i + kernel_size - filter_i) / 2
      if pad % 1 == 0:
          
          padding = (pad, pad, pad, pad)
      else:
          pad_low = np.floor(pad)
          pad_high = np.ceil(pad)
          padding = (pad_high, pad_low, pad_high, pad_low)
      self.upsample_two = nn.Sequential(nn.Conv2d(filter_i, self._n_c, kernel_size = self._kernel_size
      ,padding = padding, bias = not use_bn, stride = 1))
      if use_bn:
        self.upsample_two.append(CustomBatchNorm1d(self._n_c, test_local_stats=test_local_stats))

  def forward(self, z):
     
    method = self._method
    enc_conv_shapes = self._enc_conv_shapes[::-1]
    strides = self._dec_up_strides
    mlp_filter, conv_filters = self._filters[0], self._filters[1:]
    # We store the heights and widths of the encoder feature maps that are
    # unique, i.e., the ones right after a layer with stride != 1. These will be
    # used as a target to potentially crop the upsampled feature maps.
    unique_hw = np.unique([(el[1], el[2]) for el in enc_conv_shapes], axis=0)
    unique_hw = unique_hw.tolist()[::-1]
    unique_hw.pop()  # Drop the initial shape

    upsample = self.dec_mlp(z)
    upsample = torch.reshape(upsample,  enc_conv_shapes[0])


    upsample_one_idx = 0
    for i, (filter_i, stride_i) in enumerate(zip(conv_filters, strides), 1):
      if method != 'deconv' and stride_i > 1:
        upsample = upsample.numpy()
        upsample = tf.convert_to_tensor(upsample)
        upsample = tf.image.resize_images(
            upsample, [stride_i * el for el in upsample.shape.as_list()[1:3]],
            method=method,)
        upsample = upsample.numpy()
        upsample = torch.from_numpy(upsample)
        upsample = self.upsample_one[upsample_one_idx](upsample)
        upsample_one_idx += 1
        if self.use_bn:
          upsample = self.upsample_one[upsample_one_idx](upsample)
          upsample_one_idx += 1
        if stride_i > 1:
          hw = unique_hw.pop()
          upsample = maybe_center_crop(upsample, hw)
    
    x_logits = self.upsample_two(upsample)
    return x_logits
       

      
