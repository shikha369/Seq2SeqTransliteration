
"""Utilities for downloading  data , tokenizing, vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile

from six.moves import urllib

from tensorflow.python.platform import gfile

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_CHAR_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")


def maybe_download(directory, filename, url):
  """Download filename from url unless it's already in directory."""
  if not os.path.exists(directory):
    print("Creating directory %s" % directory)
    os.mkdir(directory)
  filepath = os.path.join(directory, filename)
  if not os.path.exists(filepath):
    print("Downloading %s to %s" % (url, filepath))
    filepath, _ = urllib.request.urlretrieve(url, filepath)
    statinfo = os.stat(filepath)
    print("Succesfully downloaded", filename, statinfo.st_size, "bytes")
  return filepath


def gunzip_file(gz_path, new_path):
  """Unzips from gz_path into new_path."""
  print("Unpacking %s to %s" % (gz_path, new_path))
  with gzip.open(gz_path, "rb") as gz_file:
    with open(new_path, "wb") as new_file:
      for line in gz_file:
        new_file.write(line)


def get_rev_enhn_train_set(directory):
  """Check whether training files exist"""
  print(directory)
  train_path = os.path.join(directory, "train")
  if not (gfile.Exists(train_path +".hn") and gfile.Exists(train_path +".en")):
    raise ValueError("Training files %s not found.", train_path)
  return train_path


def get_rev_enhn_dev_set(directory):
  """Check whether Development files exist."""
  dev_name = "valid"
  dev_path = os.path.join(directory, dev_name)
  if not (gfile.Exists(dev_path + ".hn") and gfile.Exists(dev_path + ".en")):
    raise ValueError("Development files %s not found.", dev_path)
  return dev_path


def basic_tokenizer(sentence):
  """Very basic tokenizer: split the word into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(re.split('_', space_separated_fragment))
  list1 =  [w for w in words if w]
  return list1


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=False):
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="rb") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 10000 == 0:
          print("  processing line %d" % counter)
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
        for w in tokens:
          word = re.sub(_DIGIT_RE, b"0", w) if normalize_digits else w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      print (vocab)
      print (".................")
      print (vocab.get)
      sorted(vocab, key=vocab.get, reverse=True)
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + b"\n")


def initialize_vocabulary(vocabulary_path):
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def word_to_token_ids(word, vocabulary,
                          tokenizer=None, normalize_digits=False):
  
  if tokenizer:
    chars = tokenizer(word)
  else:
    chars = basic_tokenizer(word)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in chars]
  # Normalize digits by 0 before looking chars up in the vocabulary.
  return [vocabulary.get(re.sub(_DIGIT_RE, b"0", w), UNK_ID) for w in chars]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=False):
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="rb") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 10000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = word_to_token_ids(line, vocab, tokenizer,
                                            normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_rev_data(data_dir, en_vocabulary_size, hn_vocabulary_size, tokenizer=None):
  
  # Get REV data to the specified directory.
  train_path = get_rev_enhn_train_set(data_dir)
  dev_path = get_rev_enhn_dev_set(data_dir)

  # Create vocabularies of the appropriate sizes.
  hn_vocab_path = os.path.join(data_dir, "vocab%d.hn" % hn_vocabulary_size)
  en_vocab_path = os.path.join(data_dir, "vocab%d.en" % en_vocabulary_size)
  create_vocabulary(hn_vocab_path, train_path + ".hn", hn_vocabulary_size, tokenizer)
  create_vocabulary(en_vocab_path, train_path + ".en", en_vocabulary_size, tokenizer)

  # Create token ids for the training data.
  hn_train_ids_path = train_path + (".ids%d.hn" % hn_vocabulary_size)
  en_train_ids_path = train_path + (".ids%d.en" % en_vocabulary_size)
  data_to_token_ids(train_path + ".hn", hn_train_ids_path, hn_vocab_path, tokenizer)
  data_to_token_ids(train_path + ".en", en_train_ids_path, en_vocab_path, tokenizer)

  # Create token ids for the development data.
  hn_dev_ids_path = dev_path + (".ids%d.hn" % hn_vocabulary_size)
  en_dev_ids_path = dev_path + (".ids%d.en" % en_vocabulary_size)
  data_to_token_ids(dev_path + ".hn", hn_dev_ids_path, hn_vocab_path, tokenizer)
  data_to_token_ids(dev_path + ".en", en_dev_ids_path, en_vocab_path, tokenizer)

  return (en_train_ids_path, hn_train_ids_path,
          en_dev_ids_path, hn_dev_ids_path,
          en_vocab_path, hn_vocab_path)
