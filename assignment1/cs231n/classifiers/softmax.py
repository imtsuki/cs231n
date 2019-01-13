import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  batch_size = X.shape[0] # batch size
  S = np.dot(X, W) # softmax values of the batch
  for i in range(batch_size):
    # First compute softmax and its loss
    s = S[i, :]
    s -= np.max(s) # for numeric stability
    d = np.sum(np.exp(s)) # the denominator in softmax expression
    s = np.exp(s) / d # compute softmax
    loss += -np.log(s[y[i]]) # compute loss
    # Then compute dW
    # Difference is the "-1"
    for j in range(s.shape[0]):
      if j == y[i]:
        dW[:, j] += X[i] * (s[j] - 1)
      else:
        dW[:, j] += X[i] * s[j]
  # divided by batch size, plus L2 regularization
  loss = loss / batch_size + 0.5 * reg * np.sum(W*W) # 0.5: L2 范数求导后产生系数 2, 为使得梯度表达式简洁在这里配 0.5 的系数
  dW = dW / batch_size + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  batch_size = X.shape[0]
  S = np.dot(X, W)
  S -= np.reshape(np.max(S, axis=1), (batch_size, 1)) # broadcasting
  S = np.exp(S) / np.sum(np.exp(S), axis=1, keepdims=True)
  ones = np.zeros_like(S)
  ones[np.arange(batch_size), y] = 1.0
  loss = -np.sum(np.multiply(ones, np.log(S))) / batch_size + 0.5 * reg * np.sum(W*W)
  dW = np.dot(X.T, S - ones) / batch_size + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

