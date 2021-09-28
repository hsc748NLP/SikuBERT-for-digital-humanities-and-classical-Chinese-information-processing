# -*- coding: utf-8 -*-
from functools import partial

import six
import torch
from torchtext.data import Field, RawField

from onmt.inputters.datareader_base import DataReaderBase

#from konlpy.tag import Mecab
#import sentencepiece as spm
import re


class TextDataReader(DataReaderBase):
    def read(self, sequences, side, _dir=None):
        """Read text data from disk.

        Args:
            sequences (str or Iterable[str]):
                path to text file or iterable of the actual text data.
            side (str): Prefix used in return dict. Usually
                ``"src"`` or ``"tgt"``.
            _dir (NoneType): Leave as ``None``. This parameter exists to
                conform with the :func:`DataReaderBase.read()` signature.

        Yields:
            dictionaries whose keys are the names of fields and whose
            values are more or less the result of tokenizing with those
            fields.
        """
        assert _dir is None or _dir == "", \
            "Cannot use _dir with TextDataReader."
        if isinstance(sequences, str):
            sequences = DataReaderBase._read_file(sequences)
        for i, seq in enumerate(sequences):
            if isinstance(seq, six.binary_type):
                seq = seq.decode("utf-8")
            yield {side: seq, "indices": i}


def text_sort_key(ex):
    """Sort using the number of tokens in the sequence."""
    if hasattr(ex, "tgt"):
        return len(ex.src[0]), len(ex.tgt[0])
    return len(ex.src[0])


def isHangul(text):
    hanCount = len(re.findall(u'[\u3130-\u318F\uAC00-\uD7A3]+', text))
    return hanCount > 0

def Korean_tokenizer(x):
    sp = spm.SentencePieceProcessor()
    sp.Load('data/korean_tok.model')
    token = sp.EncodeAsPieces(x)
    b = []
    for word in token:
        c = word.replace("▁", "")
        if c != '':
            b.append(c)
    return b

def English_tokenizer(x):
    sp = spm.SentencePieceProcessor()
    sp.Load('data/english_tok.model')
    token= sp.EncodeAsPieces(x)
    b = []
    for word in token:
        c = word.replace("▁", "")
        if c != '':
            b.append(c)
    return b

# mix this with partial
def _feature_tokenize(
        string, layer=0, feat_delim=None, truncate=None):
    
    # use Mecab tokenizer
    #mecab = Mecab()
    #tokens = mecab.morphs(string)
    
    # use own tokenizer
    """
    if isHangul(string):
        tokens=Korean_tokenizer(string)
    else:
        tokens=English_tokenizer(string)
    """
    # use original tokenizer
    tokens = string.split(feat_delim)
    
    #print("tokens: ",tokens)
    if truncate is not None:
        tokens = tokens[:truncate]
    if feat_delim is not None:
        tokens = [t.split(feat_delim)[layer] for t in tokens]
    return tokens

class TextMultiField(RawField):
    """Container for subfields.

    Text data might use POS/NER/etc labels in addition to tokens.
    This class associates the "base" :class:`Field` with any subfields.
    It also handles padding the data and stacking it.

    Args:
        base_name (str): Name for the base field.
        base_field (Field): The token field.
        feats_fields (Iterable[Tuple[str, Field]]): A list of name-field
            pairs.

    Attributes:
        fields (Iterable[Tuple[str, Field]]): A list of name-field pairs.
            The order is defined as the base field first, then
            ``feats_fields`` in alphabetical order.
    """

    def __init__(self, base_name, base_field, feats_fields):
        super(TextMultiField, self).__init__()
        print("TextMultiField")
        self.fields = [(base_name, base_field)]
        for name, ff in sorted(feats_fields, key=lambda kv: kv[0]):
            self.fields.append((name, ff))

    @property
    def base_field(self):
        return self.fields[0][1]

    def process(self, batch, device=None):
        """Convert outputs of preprocess into Tensors.

        Args:
            batch (List[List[List[str]]]): A list of length batch size.
                Each element is a list of the preprocess results for each
                field (which are lists of str "words" or feature tags.
            device (torch.device or str): The device on which the tensor(s)
                are built.

        Returns:
            torch.LongTensor or Tuple[LongTensor, LongTensor]:
                A tensor of shape ``(seq_len, batch_size, len(self.fields))``
                where the field features are ordered like ``self.fields``.
                If the base field returns lengths, these are also returned
                and have shape ``(batch_size,)``.
        """

        # batch (list(list(list))): batch_size x len(self.fields) x seq_len
        batch_by_feat = list(zip(*batch))
        base_data = self.base_field.process(batch_by_feat[0], device=device)
        if self.base_field.include_lengths:
            # lengths: batch_size
            base_data, lengths = base_data

        feats = [ff.process(batch_by_feat[i], device=device)
                 for i, (_, ff) in enumerate(self.fields[1:], 1)]
        levels = [base_data] + feats
        # data: seq_len x batch_size x len(self.fields)
        data = torch.stack(levels, 2)
        if self.base_field.include_lengths:
            return data, lengths
        else:
            return data

    def preprocess(self, x):
        """Preprocess data.

        Args:
            x (str): A sentence string (words joined by whitespace).

        Returns:
            List[List[str]]: A list of length ``len(self.fields)`` containing
                lists of tokens/feature tags for the sentence. The output
                is ordered like ``self.fields``.
        """

        return [f.preprocess(x) for _, f in self.fields]

    def __getitem__(self, item):
        return self.fields[item]


def text_fields(**kwargs):
    """Create text fields.

    Args:
        base_name (str): Name associated with the field.
        n_feats (int): Number of word level feats (not counting the tokens)
        include_lengths (bool): Optionally return the sequence lengths.
        pad (str, optional): Defaults to ``"<blank>"``.
        bos (str or NoneType, optional): Defaults to ``"<s>"``.
        eos (str or NoneType, optional): Defaults to ``"</s>"``.
        truncate (bool or NoneType, optional): Defaults to ``None``.

    Returns:
        TextMultiField
    """
    #print("text_fields")
    n_feats = kwargs["n_feats"]
    include_lengths = kwargs["include_lengths"]
    base_name = kwargs["base_name"]
    pad = kwargs.get("pad", "<blank>")
    bos = kwargs.get("bos", "<s>")
    eos = kwargs.get("eos", "</s>")
    truncate = kwargs.get("truncate", None)
    fields_ = []
    feat_delim = u"￨" if n_feats > 0 else None
    for i in range(n_feats + 1):
        name = base_name + "_feat_" + str(i - 1) if i > 0 else base_name
        tokenize = partial(
            _feature_tokenize,
            layer=i,
            truncate=truncate,
            feat_delim=feat_delim)
        use_len = i == 0 and include_lengths
        feat = Field(
            init_token=bos, eos_token=eos,
            pad_token=pad, tokenize=tokenize,
            include_lengths=use_len)
        fields_.append((name, feat))
    assert fields_[0][0] == base_name  # sanity check
    field = TextMultiField(fields_[0][0], fields_[0][1], fields_[1:])
    return field
