
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import codecs

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
tf.app.flags.DEFINE_integer("batch_size", 200,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 256, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("en_vocab_size", 32, "English vocabulary size.")
tf.app.flags.DEFINE_integer("hn_vocab_size", 86, "Hindi vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "/home/shikha/temp/data", "Data directory")
tf.app.flags.DEFINE_string("transliterate_file_dir", "/home/shikha/temp/data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/home/shikha/temp/data", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("transliterate_file", False,
                            "Set to True for evaluating.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_integer("pat", 0,
                            "patience parameter to use during training.")
tf.app.flags.DEFINE_integer("max_steps", 42000,
                            "max steps to use during training.")
tf.app.flags.DEFINE_integer("drop_out", 0,
                            "drop out to use during training.")
tf.app.flags.DEFINE_integer("init_", 1,
                            "initlisae to use during training.")
tf.app.flags.DEFINE_boolean("bi_lstm", True,
                            "Set to True for bilstm .")
tf.app.flags.DEFINE_boolean("attent", True,
                            "Set to True for attention .")
                            

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
      forward_only=forward_only,use_lstm=True,drop_out = FLAGS.drop_out,bi_lstm = FLAGS.bi_lstm,init_ = FLAGS.init_,attent=FLAGS.attent)
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
        print ("Accuracy on validation is : "+ str(acc) )
        print (itera)
        
        # patience parameter on validation accuracy 
        if FLAGS.pat > 0:
        
           if len(validation_acc) > (FLAGS.pat - 1) and acc < min(validation_acc[-1*FLAGS.pat:]):
            current_step = FLAGS.max_steps

        valid_loss_current_epoch = np.mean(np.asarray(valid_loss))
        validation_loss.append(valid_loss_current_epoch)
        validation_acc.append(acc)

        file_logs.write("Step %i, train_Loss: % .5f, valid_Loss: % .5f \n" %(current_step, loss, valid_loss_current_epoch))
        acc_logs.write("Step %i, accuracy: % .5f\n" %(current_step,acc ))

        print ("validation loss % .3f" % (valid_loss_current_epoch))
        print ("Train loss % .3f" % (step_loss))			 #changed it to step_loss instead of running avg loss



        #reset params
        step_time, loss = 0.0, 0.0
        sys.stdout.flush()
    file_logs.close()
    np.savetxt(FLAGS.train_dir+'log_loss_train.csv',np.asarray(previous_losses),delimiter = ',')
    np.savetxt(FLAGS.train_dir+'log_loss_valid.csv',np.asarray(validation_loss),delimiter = ',')

def decode():
  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, True)
    model.batch_size = 10  # We decode one word at a time.

    # Load vocabularies.
    en_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.en" % FLAGS.en_vocab_size)
    hn_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.hn" % FLAGS.hn_vocab_size)
    en_vocab, _ = data_utils.initialize_vocabulary(en_vocab_path)
    _, rev_hn_vocab = data_utils.initialize_vocabulary(hn_vocab_path)

    # Decode from standard input.
    sys.stdout.write("> ")
    sys.stdout.flush()
    word = sys.stdin.readline()
    while word:
      #word = word.lower()
      char_list_new = list(word)
      word = " ".join(char_list_new)
      # Get token-ids for the input word.
      token_ids = data_utils.word_to_token_ids(tf.compat.as_bytes(word), en_vocab)
      # Which bucket does it belong to?
      bucket_id = min([b for b in xrange(len(_buckets))
                       if _buckets[b][0] > len(token_ids)])
      # Get a 1-element batch to feed the word to the model.
      encoder_inputs, decoder_inputs, target_weights,_ = model.get_batch(
          {bucket_id: [(token_ids, [])]}, bucket_id)
      # Get output logits for the word.
      _, _, output_logits,out = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
      # This is a greedy decoder - outputs are just argmaxes of output_logits.
      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      # If there is an EOS symbol in outputs, cut them at that point.
      if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
      # Print out Hindi word corresponding to outputs.
      print("".join([tf.compat.as_str(rev_hn_vocab[output]) for output in outputs]))
      
      print("> ", end="")
      sys.stdout.flush()
      word = sys.stdin.readline()


def self_test():
  """Test the translation model."""
  with tf.Session() as sess:
    print("Self-test for neural transliteration model.")
    # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
    model = seq2seq_model.Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 32, 2,
                                       5.0, 32, 0.3, 0.99, num_samples=8)
    sess.run(tf.initialize_all_variables())

    # Fake data set for both the (3, 3) and (6, 6) bucket.
    data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
    for _ in xrange(5):  # Train the fake model for 5 steps.
      bucket_id = random.choice([0, 1])
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          data_set, bucket_id)
      model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                 bucket_id, False)

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
      # Get output logits for the word.
      #print (len(encoder_inputs))
      _, _, output_logits,_ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)

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
      if outputs==hindi_target_ids:
          correct_prediction = correct_prediction + 1
      hn_output = "".join([tf.compat.as_str(rev_hn_vocab[output]) for output in outputs])
      if i%100 == 0:
        #print(str(i)+' out of ' + str(len(en_eval_list)) +' words decoded\n English Input: ' + word + '\t Hindi Output: ' + hn_output)
      #print (hn_output)
          print (i)
      file_content_output.append([word,hn_output])

    #print('done generating the output file!!!')
    #fc_str = '\n'.join(['\t'.join(row) for row in file_content_output])
    #f = codecs.open(result_path, encoding='utf-8', mode='wb')
    #f.write(fc_str.decode('utf-8'))
    accuracy = float(correct_prediction)/1000
    print (accuracy)
    return accuracy
    #Check
def main_called(args):
  FLAGS.init_ = args.init
  FLAGS.learning_rate = args.lr
  FLAGS.batch_size = args.batch_size
  FLAGS.size = args.cell_size
  FLAGS.data_dir = args.data_dir
  FLAGS.transliterate_file_dir = args.data_dir
  FLAGS.steps_per_checkpoint = args.checkpoint
  if args.evaluate == 1:
      FLAGS.transliterate_file = True
  else:
      FLAGS.transliterate_file = False
  print (FLAGS.transliterate_file)
  FLAGS.train_dir = args.data_dir
  FLAGS.drop_out = args.drop_out
  if args.attent == 1:
      FLAGS.attent = True
  else :
      FLAGS.attent = False
    
  print (FLAGS.attent)
  if args.bi_lstm == 1:
      FLAGS.bi_lstm = True
  else:
      FLAGS.bi_lstm = False
  FLAGS.pat = args.pat
  
  if FLAGS.transliterate_file == False:
      train()
  else:
      _=evaluate('test.en','test.ids86.hn')
  

def main(_):

  if FLAGS.self_test:
    self_test()
  elif FLAGS.decode:
    decode()
  elif FLAGS.transliterate_file:
    _=evaluate('test.en','test.ids86.hn')
  else:
    train()
    


if __name__ == "__main__":
  tf.app.run()
