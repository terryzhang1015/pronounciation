import sys
import time
import numpy as np
import tensorflow as tf

hidden_dim = 64
layer_number = 2
target_number = 68
train_data = "train_small.dat.npy"
test_data = "test.dat.npy"
learning_rate = 0.01
batch_size = 20

def build_model(x1, hidden_dim, layer_number, target_number, eps=1e-8):
  with tf.variable_scope("model"):
    rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [hidden_dim] * layer_number]
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
    x1_rnn, _ = tf.nn.dynamic_rnn(cell=multi_rnn_cell, inputs=x1, time_major=False, dtype=tf.float32)
    predict = tf.layers.dense(x1_rnn, target_number)
  return predict

def get_loss(predict, label, seq_len):
  # predict = tf.transpose(predict, (1, 0, 2))
  loss = tf.nn.ctc_loss(label, predict, seq_len, time_major=False)
  loss = tf.reduce_mean(loss)
  return loss

def get_rec_results(predict, label, seq_len):
  predict = tf.transpose(predict, (1, 0, 2))
  results, _ = tf.nn.ctc_greedy_decoder(predict, seq_len)
  error_rate = tf.reduce_mean(tf.edit_distance(tf.cast(results[0], tf.int32), label))
  return results, error_rate

def sparse_tuple(sequences, dtype=np.int32):
  indices = []
  values = []
  for n, seq in enumerate(sequences):
    indices.extend(zip([n] * len(seq), range(len(seq))))
    values.extend(seq)
  indices = np.asarray(indices, dtype=np.int32)
  values = np.asarray(values, dtype=dtype)
  shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int32)
  return indices, values, shape

def data_process(input_data):
  len_data = len(input_data[0])
  len_seq = [0] * len_data
  output_data = []
  output_label = []
  for i in range(len_data):
    len_seq[i] = len(input_data[0][i])
  len_seq = np.array(len_seq)
  sort_index = np.argsort(len_seq)
  for i in range(len_data):
    tmpindex = sort_index[i]
    tmpdata = input_data[0][tmpindex]
    tmplabel = input_data[1][tmpindex]
    output_data.append(tmpdata)
    output_label.append(tmplabel)
  return output_data, output_label

def get_a_batch(input_mfcc, input_label, i, bs = 20):
  batch_mfcc = []
  batch_label = []
  batch_seq_len = []
  len_data = len(input_mfcc)
  maxlen = 0
  for j in range(i*bs, (i+1)*bs):
    tmpmfcc = input_mfcc[j % len_data]
    tmplen = len(tmpmfcc)
    if tmplen > maxlen:
      maxlen = tmplen
  for j in range(i*bs, (i+1)*bs):
    tmpmfcc = input_mfcc[j % len_data]
    tmpmfcc_pad = np.zeros([maxlen, 39])
    tmpmfcc_pad[:len(tmpmfcc),:] = tmpmfcc
    batch_mfcc.append(tmpmfcc_pad)
    tmplabel = input_label[j % len_data]
    batch_label.append(tmplabel)
    batch_seq_len.append(len(tmpmfcc))
  batch_mfcc = np.array(batch_mfcc)
  batch_label = sparse_tuple(batch_label)
  batch_seq_len = np.array(batch_seq_len, dtype=np.int32)
  return batch_mfcc, batch_label, batch_seq_len

def train(train_data, test_data):
  x1 = tf.placeholder(tf.float32, [None, None, 39])
  label = tf.sparse_placeholder(tf.int32)
  seq_len = tf.placeholder(tf.int32, [None])
  predict = build_model(x1, hidden_dim, layer_number, target_number)
  loss = get_loss(predict, label, seq_len)
  (rec_results, error_rate) = get_rec_results(predict, label, seq_len)
  rec_results_dense = tf.sparse.to_dense(rec_results[0])

  params = tf.trainable_variables()
  for a_p in params:
    print(a_p)
  opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
  # opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
  gradients = tf.gradients(loss, params)
  clipped_gradients, norm = tf.clip_by_global_norm(gradients, 5.0)
  train_op = opt.apply_gradients(zip(clipped_gradients, params))

  saver = tf.train.Saver(max_to_keep=100)
  checkpoint_path = './models/model.ckpt'
  init = tf.global_variables_initializer()
  sess = tf.Session()
  sess.run(init)
  
  train_all=np.load(train_data, allow_pickle=True)
  test_all=np.load(test_data, allow_pickle=True)
  (train_mfcc, train_label) = data_process(train_all)
  (test_mfcc, test_label) = data_process(test_all)

  phones={}
  with open('phone_mapping.txt') as f:
    for line in f:
      line=line.strip()
      if len(line) > 0:
        linesplits = line.split()
        phones[int(linesplits[1])] = linesplits[0]
  
  num_batches = 1000000
  i = 0
  t0 = time.time()
  while(i < num_batches):
    (x1_tmp, label_tmp, seq_len_tmp) = get_a_batch(train_mfcc, train_label, i)
    (_, loss_value, predict_value, ler) = sess.run([train_op, loss, predict, error_rate], feed_dict={x1:x1_tmp, label:label_tmp, seq_len:seq_len_tmp})
    if i % 50 == 0:
      print('training loss and error rate', i, loss_value, ler)
      sys.stdout.flush()
    if i % 300 == 0:
      saver.save(sess, checkpoint_path, global_step=i)
      loss_mean = 0.0
      error_mean = 0.0
      for j in range(20):  # test 20 batches
        (x1_tmp, label_tmp, seq_len_tmp) = get_a_batch(test_mfcc, test_label, j)
        (loss_value, predict_value, rec_results_value, ler) = sess.run([loss, predict, rec_results_dense, error_rate], feed_dict={x1:x1_tmp, label:label_tmp, seq_len:seq_len_tmp})
        loss_mean += loss_value
        error_mean += ler
      loss_mean = loss_mean / 20.0
      error_mean = error_mean / 20.0
      print('test loss and error rate', i, loss_mean, error_mean)
      if i % 3000 == 0:
        print('last test batch rec results')
        for _ in range(20):
          tmprec = rec_results_value[_]
          tmprec_new = []
          for k in range(len(tmprec)):
            if tmprec[k] != -1:
              tmprec_new.append(phones[tmprec[k]])
          print(' '.join(tmprec_new))
    i+=1

if __name__ == '__main__':
  train(train_data, test_data)
