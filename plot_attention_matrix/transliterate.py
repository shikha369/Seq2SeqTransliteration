#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from __future__ import unicode_literals
import math
import os
import random
import sys
import time
import codecs


#matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
import seq2seq_model


tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 7.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 128,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 256, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("en_vocab_size", 32, "English vocabulary size.")
tf.app.flags.DEFINE_integer("hn_vocab_size", 86, "Hindi vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "/home/shikha/Desktop/pa3/data/", "Data directory")
tf.app.flags.DEFINE_string("transliterate_file_dir", "/home/shikha/Desktop/pa3/data/", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/home/shikha/Desktop/pa3/data/", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("transliterate_file", True,
                            "Set to True for evaluating.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_integer("pat", 5,
                            "patience parameter to use during training.")
tf.app.flags.DEFINE_integer("max_steps", 60000,
                            "max steps to use during training.")
tf.app.flags.DEFINE_integer("drop_out", 0,
                            "drop out to use during training.")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 10), (10, 15), (15, 25),(25,35),(35,60),(60,70)]



def read_data(source_path, target_path, max_size=None):
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 10000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(data_utils.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set


def create_model(session, forward_only):
  """Create transliteration model and initialize or load parameters in session."""
  model = seq2seq_model.Seq2SeqModel(
      FLAGS.en_vocab_size, FLAGS.hn_vocab_size, _buckets,
      FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
      forward_only=forward_only,use_lstm=True,drop_out = FLAGS.drop_out)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and ckpt.model_checkpoint_path:
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())
  return model


def train():
  """Train a en->hn transliteration model using REV_brandnames data."""
  # Prepare news_2012 data.
  print("Preparing News_2012 data in %s" % FLAGS.data_dir)
  en_train, hn_train, en_dev, hn_dev, _, _ = data_utils.prepare_rev_data(
      FLAGS.data_dir, FLAGS.en_vocab_size, FLAGS.hn_vocab_size)

  with tf.Session() as sess:
    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, False)  # forward_only = False

    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)."
           % FLAGS.max_train_data_size)
    dev_set = read_data(en_dev, hn_dev)
    train_set = read_data(en_train, hn_train, FLAGS.max_train_data_size)
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes)) #size of dataset

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    validation_loss = []
    validation_acc = []
    file_logs = open(FLAGS.train_dir+"/log_loss.txt", "w")
    acc_logs = open(FLAGS.train_dir+"/accuracy.txt", "w")
    while current_step <= FLAGS.max_steps:
      # Choose a bucket according to data distribution. We pick a random number
      # in [0, 1] and use the corresponding interval in train_buckets_scale.
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

      # Get a batch and make a step.
      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights,_ = model.get_batch(
          train_set, bucket_id)
      _, step_loss, _, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False)
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.steps_per_checkpoint == 0:
        # Print statistics for the previous epoch.
        perplexity = math.exp(loss) if loss < 300 else float('inf')
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))



        # Decrease learning rate if no improvement was seen over last 3 times.

        '''if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op)'''
        previous_losses.append(loss)

        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(FLAGS.train_dir, "transliterate.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)

        # Run evals on development set and print their perplexity.
        valid_loss = []
        correct_prediction = 0
        itera =0
        for bucket_id in xrange(len(_buckets)):
          if len(dev_set[bucket_id]) == 0:
            print("  eval: empty bucket %d" % (bucket_id))
            continue
          encoder_inputs, decoder_inputs, target_weights,dec_input = model.get_batch(
              dev_set, bucket_id)

          _, eval_loss, output_logits,_ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)

          eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
          print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
          valid_loss.append(eval_loss)

          output_logits= np.asarray(output_logits)
          #print (np.shape(output_logits))
          j=0
          for x in range(np.shape(output_logits)[1]):
              outputs = []
              for y in range(np.shape(output_logits)[0]):
                  logit = output_logits[y][x][:]
                  outputs.append([int(np.argmax(logit))])
              #print (np.shape(np.asarray(outputs)))
              #outputs = [int(np.argmax(logit,axis=1))for logit in output_logits[:][x][:]]
              # If there is an EOS symbol in outputs, cut them at that point.



              # Print out Hindi word corresponding to outputs.
              #print (outputs)
              output_ = []
              for li in outputs:
                  output_.extend(li)
              #print (output_)
              if data_utils.EOS_ID in output_:
                  output_ = output_[:output_.index(data_utils.EOS_ID)+1]

              #print (output_)
              #print (dec_input[j])
              itera = itera +1
              if output_ == dec_input[j]:
                  correct_prediction = correct_prediction + 1
              j += 1

        acc = correct_prediction/itera
        print (acc)
        print (itera)

        '''if len(validation_loss) > (FLAGS.pat - 1) and valid_loss_current_epoch > max(validation_loss[-1*FLAGS.pat:]):
            current_step = FLAGS.max_steps '''
        '''if len(validation_acc) > (FLAGS.pat - 1) and acc < min(validation_acc[-1*FLAGS.pat:]):
            current_step = FLAGS.max_steps'''
        valid_loss_current_epoch = np.mean(np.asarray(valid_loss))
        validation_loss.append(valid_loss_current_epoch)
        validation_acc.append(acc)
        file_logs.write("Step %i, train_Loss: % .5f, valid_Loss: % .5f \n" %(current_step, loss, valid_loss_current_epoch))
        acc_logs.write("Step %i, accuracy: % .5f\n" %(current_step,acc ))
        print ("validation loss % .3f" % (valid_loss_current_epoch))
        print ("Train loss % .3f" % (step_loss)) #changed it to step_loss instead of running avg loss



        #reset params
        step_time, loss = 0.0, 0.0
        sys.stdout.flush()
    file_logs.close()
    np.savetxt(FLAGS.train_dir+'log_loss_train.csv',np.asarray(previous_losses),delimiter = ',')
    np.savetxt(FLAGS.train_dir+'log_loss_valid.csv',np.asarray(validation_loss),delimiter = ',')


def evaluate(input_fname,output_fname):
  """Generate an evaluation output of the  model.
     Takes the directory path for evaluation from FLAGS and writes an output file to the same directory"""
  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, True)
    model.batch_size = 1 # We decode one word at a time.
    correct_prediction=0
    # Load vocabularies.
    en_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.en" % FLAGS.en_vocab_size)
    hn_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.hn" % FLAGS.hn_vocab_size)
    en_vocab, _ = data_utils.initialize_vocabulary(en_vocab_path)
    _, rev_hn_vocab = data_utils.initialize_vocabulary(hn_vocab_path)

    #path for loading the evaluation file
    en_eval_path = os.path.join(FLAGS.transliterate_file_dir,input_fname)
    hn_target_path = os.path.join(FLAGS.transliterate_file_dir,output_fname)
    ft = open(hn_target_path)
    print ('reading target output for the word')

    print('Transliterating '+en_eval_path)
    #Path to save the output file
    result_path = os.path.join(FLAGS.transliterate_file_dir,'result.txt')
    print('Results will be stored in '+result_path)

    en_eval_list = []
    file_content_output = []
    print('reading input file')

    with open(en_eval_path) as fp:
      for line in fp:
          char_list = list(line)
          #en_eval_list.append(char_list)
          #en_eval_list.append(' ')
          space_separated = ' '.join(char_list)
          en_eval_list.append(space_separated)

    print('decoding input file')

    for i,word in enumerate(en_eval_list):
      #word = word.lower()

      char_list_new = list(word)
      #print (np.shape(np.asarray(char_list_new)))
      word = " ".join(char_list_new)
      # Get token-ids for the input word.
      token_ids = data_utils.word_to_token_ids(tf.compat.as_bytes(word), en_vocab)
      # Which bucket does it belong to?
      bucket_list = [b for b in xrange(len(_buckets))
                       if _buckets[b][0] > len(token_ids)]
      #bucket_id = min([b for b in xrange(len(_buckets))
      #                 if _buckets[b][0] > len(token_ids)])
      if len(bucket_list) == 0:
        print('could not find bucket')
        continue
      bucket_id = min(bucket_list)
      # Get a 1-element batch to feed the word to the model.
      encoder_inputs, decoder_inputs, target_weights,_ = model.get_batch(
          {bucket_id: [(token_ids, [])]}, bucket_id)
      #print (token_ids)
      # Get output logits for the word.
      #print (len(encoder_inputs))
      _, loss, output_logits,attn_matrix = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)

      #print (loss)
      #print (np.shape(np.asarray(output_logits)))
      # This is a greedy decoder - outputs are just argmaxes of output_logits.

      outputs = [int(np.argmax(logit,axis=1))for logit in output_logits]
      # If there is an EOS symbol in outputs, cut them at that point.
      if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
      # Print out Hindi word corresponding to outputs.
      #print (outputs)
      hindi_target_ids = ft.readline().split()
      hindi_target_ids = map(int, hindi_target_ids)
      #print (hindi_target_ids)
      #print ("done")
      #print (outputs)
      #print (hindi_target_ids)
      if outputs==hindi_target_ids:
          correct_prediction = correct_prediction + 1
          #print ('*'*80)
          #print (outputs)
          #print (hindi_target_ids)
          hn_output = "".join([tf.compat.as_str(rev_hn_vocab[output]) for output in outputs])
          #print (hn_output)
          #print word
          wordH = [rev_hn_vocab[output] for output in outputs]
          plot_attention(attn_matrix,bucket_id,word,wordH)
      #print (([tf.compat.as_str(rev_hn_vocab[output]) for output in outputs]))

      if i%100 == 0:
        #print(str(i)+' out of ' + str(len(en_eval_list)) +' words decoded\n English Input: ' + word + '\t Hindi Output: ' + hn_output)

          print (i)


      #file_content_output.append([word,hn_output])

    #print('done generating the output file!!!')
    #fc_str = '\n'.join(['\t'.join(row) for row in file_content_output])
    #f = codecs.open(result_path, encoding='utf-8', mode='wb')
    #f.write(fc_str.decode('utf-8'))
    accuracy = correct_prediction/1000
    print (correct_prediction)
    print(type(correct_prediction))
    print (accuracy)
    return accuracy


def plot_attention(data,bucket_id, X_label=None, Y_label=None):
  fig, ax = plt.subplots(1) # set figure size
  heatmap = ax.pcolor(data, cmap=plt.cm.Blues, alpha=0.9)
  print (X_label)
  #print (Y_label)
  fig.canvas.draw()
  plt.rc('font',**{'family':'serif','serif':['Times']})
  # Set axis labels
  if X_label != None and Y_label != None:
    #X_label = [x_label.decode('utf-8') for x_label in X_label]
    Y_label = [y_label.decode('utf-8') for y_label in Y_label]


    x_lab = X_label.split()
    diff = _buckets[bucket_id][0]-len(x_lab)
    print (diff)
    xticks = range(diff,len(x_lab)+diff)
    ax.set_xticks(xticks) # major ticks
    print (x_lab)
    print (len(x_lab))
    ax.set_xticklabels(x_lab)

    print (len(Y_label))
    yticks = range(0,len(Y_label))
    ax.set_yticks(yticks)
    y_lab = [Y_label[i] for i in range(0,len(Y_label))]
    print (y_lab)
    print (len(y_lab))
    #hfont = {'fontname':'Comic Sans MS'}
    ax.set_yticklabels(y_lab)

    ax.grid(True)

  # plot Figure
  plt.title('Attention Heatmap')
  plt.show()
  # OR plt.matshow(attn_matrix,cmap=plt.cm.gray)
  #plt.show()

def main(_):
  if FLAGS.self_test:
    self_test()
  elif FLAGS.decode:
    decode()
  elif FLAGS.transliterate_file:
    evaluate('test.en','test.ids86.hn')
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
