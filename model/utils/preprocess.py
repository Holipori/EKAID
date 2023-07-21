"""
Utilities for preprocessing sequence data.

Special tokens that are in all dictionaries:

<NULL>: Extra parts of the sequence that we should ignore
<START>: Goes at the start of a sequence
<END>: Goes at the end of a sequence, before <NULL> tokens
<UNK>: Out-of-vocabulary words
"""

SPECIAL_TOKENS = {
  '<NULL>': 0,
  '<UNK>': 1,
  '<START>': 2,
  '<END>': 3,
}


def tokenize(s, delim=' ',
      add_start_token=True, add_end_token=True,
      punct_to_keep=None, punct_to_remove=None):
  """
  Tokenize a sequence, converting a string s into a list of (string) tokens by
  splitting on the specified delimiter. Optionally keep or remove certain
  punctuation marks and add start and end tokens.
  """
  if punct_to_keep is not None:
    for p in punct_to_keep:
      s = s.replace(p, '%s%s' % (delim, p))

  if punct_to_remove is not None:
    for p in punct_to_remove:
      s = s.replace(p, '')

  tokens = s.split(delim)
  if add_start_token:
    tokens.insert(0, '<START>')
  if add_end_token:
    tokens.append('<END>')
  return tokens

def filt_word_category(pos_token, words):
    # load the category file

    # give each category a ID
    category_name_un = ['FW', '-LRB-', '-RRB-', 'LS']
    category_name_vb = ['VB', 'VBD', 'VBP', 'VBG', 'VBN', 'VBZ']
    category_name_nn = ['NN', 'NNS', 'NNP']
    category_name_jj = ['JJ', 'JJR', 'JJS']
    category_name_rb = ['RB', 'RBS', 'RBR', 'WRB', 'EX']
    category_name_cc = ['CC']
    category_name_pr = ['PRP', 'PRP$', 'WP', 'POS', 'WP$']
    category_name_in = ['IN', 'TO']
    category_name_dt = ['DT', 'WDT', 'PDT']
    category_name_rp = ['RP', 'MD']
    category_name_cd = ['CD']
    category_name_sy = ['SYM', ':', '``', '#', '$']
    category_name_uh = ['UH']

    all_category = pos_token.values()
    category_id = {}  # {VB:2, VBS:2, NN:3, NNS:3 ...}
    for category in all_category:
        if category in category_name_vb:
            category_id[category] = 4
        elif category in category_name_nn:
            category_id[category] = 5
        elif category in category_name_jj:
            category_id[category] = 6
        elif category in category_name_rb:
            category_id[category] = 7
        elif category in category_name_cc:
            category_id[category] = 8
        elif category in category_name_pr:
            category_id[category] = 9
        elif category in category_name_in:
            category_id[category] = 10
        elif category in category_name_dt:
            category_id[category] = 11
        elif category in category_name_rp:
            category_id[category] = 12
        elif category in category_name_cd:
            category_id[category] = 13
        elif category in category_name_sy:
            category_id[category] = 14
        elif category in category_name_uh:
            category_id[category] = 15
        else:
            category_id[category] = 1
    # turn words' category from str to ID
    all_words_in_category = pos_token.keys()
    filted_words_categoryid = {}  # {'<EOS>':0, '<UNK>':1, 'cat':3, 'take':2, 'log_vir':1}
    for key in words:
        if key in all_words_in_category:
            the_key_category = pos_token[key]
            filted_words_categoryid[key] = category_id[the_key_category]
        else:
            filted_words_categoryid[key] = 1

    filted_words_categoryid['<NULL>'] = 0
    filted_words_categoryid['<UNK>'] = 1
    filted_words_categoryid['<START>'] = 2
    filted_words_categoryid['<END>'] = 3
    # take out the unmasked category ids
    unmasked_categoryid = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # VB, NN, JJ, and RB needn't be masked
    return filted_words_categoryid, pos_token, category_id, unmasked_categoryid


def build_vocab(sequences, min_token_count=1, delim=' ',
                punct_to_keep=None, punct_to_remove=None):
  token_to_count = {}
  tokenize_kwargs = {
    'delim': delim,
    'punct_to_keep': punct_to_keep,
    'punct_to_remove': punct_to_remove,
  }
  for seq in sequences:
    seq_tokens = tokenize(seq, **tokenize_kwargs,
                    add_start_token=False, add_end_token=False)
    for token in seq_tokens:
      if token not in token_to_count:
        token_to_count[token] = 0
      token_to_count[token] += 1

  token_to_idx = {}
  for token, idx in SPECIAL_TOKENS.items():
    token_to_idx[token] = idx
  for token, count in sorted(token_to_count.items()):
    if count >= min_token_count:
      token_to_idx[token] = len(token_to_idx)

  return token_to_idx


def encode(seq_tokens, token_to_idx, allow_unk=False):
  seq_idx = []
  for token in seq_tokens:
    if token not in token_to_idx:
      if allow_unk:
        token = '<UNK>'
      else:
        raise KeyError('Token "%s" not in vocab' % token)
    seq_idx.append(token_to_idx[token])
  return seq_idx


def decode(seq_idx, idx_to_token, delim=None, stop_at_end=True):
  tokens = []
  for idx in seq_idx:
    tokens.append(idx_to_token[idx])
    if stop_at_end and tokens[-1] == '<END>':
      break
  if delim is None:
    return tokens
  else:
    return delim.join(tokens)
