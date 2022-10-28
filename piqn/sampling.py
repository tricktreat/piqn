import pdb
import random

import torch

from piqn import util


def create_train_sample(doc, random_mask = False, tokenizer = None, repeat_gt_entities = -1):
    pos_encoding = [t.pos_id for t in doc.tokens]
    encodings = doc.encoding
    seg_encoding = doc.seg_encoding
    # if len(doc.encoding) > 512:
    #     return None
    token_count = len(doc.tokens)
    context_size = len(encodings)

    gt_seq_labels = [0] * len(encodings)
    special_tokens_map = tokenizer.special_tokens_map
    if random_mask:
        if random.random() < 0.5:
            for i in range(len(gt_seq_labels) -1):
                replace_rnd = random.random()
                if replace_rnd < 0.15 and i != 0:
                    gt_seq_labels[i] = encodings[i]
                    strategy_rnd = random.random()
                    if strategy_rnd < 0.8:
                        encodings[i] = tokenizer.convert_tokens_to_ids(special_tokens_map['mask_token'])
                    elif strategy_rnd < 0.9:
                        encodings[i] = random.randint(0, tokenizer.vocab_size - 1)
        # else:
        #     gt_seq_labels[0] = encodings[0]

    char_encodings = doc.char_encoding
    char_encoding = []
    char_count = []
    for char_encoding_token in char_encodings:
        char_count.append(len(char_encoding_token))
        char_encoding.append(torch.tensor(char_encoding_token,dtype=torch.long))
    char_encoding = util.padded_stack(char_encoding)
    token_masks_char = (char_encoding!=0).long()
    char_count = torch.tensor(char_count, dtype = torch.long)

    wordvec_encoding = [t.wordinx for t in doc.tokens]
    
    # import pdb; pdb.set_trace()
    # all tokens
    context2token_masks = []
    for t in doc.tokens:
        context2token_masks.append(create_entity_mask(*t.span, context_size))

    gt_entities_spans_token = []
    gt_entity_types = []
    gt_entity_masks = []
    
    for e in doc.entities:
        gt_entities_spans_token.append(e.span_token)
        gt_entity_types.append(e.entity_type.index)
        gt_entity_masks.append(1)

    if repeat_gt_entities != -1:
        if len(doc.entities)!=0:
            k = repeat_gt_entities//len(doc.entities)
            m = repeat_gt_entities%len(doc.entities)
            gt_entities_spans_token = gt_entities_spans_token*k + gt_entities_spans_token[:m]
            gt_entity_types = gt_entity_types*k + gt_entity_types[:m]
            gt_entity_masks = gt_entity_masks*k + gt_entity_masks[:m]
            assert len(gt_entities_spans_token) == len(gt_entity_types) == len(gt_entity_masks) == repeat_gt_entities

    # create tensors
    # token indices
    encodings = torch.tensor(encodings, dtype=torch.long)
    seg_encoding = torch.tensor(seg_encoding, dtype=torch.long)
    gt_seq_labels = torch.tensor(gt_seq_labels, dtype=torch.long)
    pos_encoding = torch.tensor(pos_encoding, dtype=torch.long)
    wordvec_encoding = torch.tensor(wordvec_encoding, dtype=torch.long)

    # masking of tokens
    context_masks = torch.ones(context_size, dtype=torch.bool)
    # context_masks = torch.tensor(seg_encoding, dtype=torch.bool)
    token_masks = torch.ones(token_count, dtype=torch.bool)

    context2token_masks = torch.stack(context2token_masks)

    if len(gt_entity_types) > 0:
        gt_entity_types = torch.tensor(gt_entity_types, dtype=torch.long)
        # gt_entity_spans_token = torch.tensor(gt_entities_spans_token, dtype=torch.float) / len(doc.tokens)
        gt_entity_spans_token = torch.tensor(gt_entities_spans_token, dtype=torch.long)
        # gt_entity_spans_token[:, 1] = gt_entity_spans_token[:, 1] - 1
        gt_entity_masks = torch.tensor(gt_entity_masks, dtype=torch.bool)
    else:
        gt_entity_types = torch.zeros([1], dtype=torch.long)
        gt_entity_spans_token = torch.zeros([1, 2], dtype=torch.long)
        gt_entity_masks = torch.zeros([1], dtype=torch.bool)
    
    return dict(encodings=encodings, context_masks=context_masks, seg_encoding = seg_encoding, context2token_masks=context2token_masks, token_masks=token_masks, 
                pos_encoding = pos_encoding, wordvec_encoding = wordvec_encoding, char_encoding = char_encoding, token_masks_char = token_masks_char, char_count = char_count,
                gt_types=gt_entity_types, gt_spans=gt_entity_spans_token, entity_masks=gt_entity_masks, gt_seq_labels = gt_seq_labels, meta_doc = doc)


def create_eval_sample(doc):
    # if len(doc.encoding) > 512:
    #     return None
    pos_encoding = [t.pos_id for t in doc.tokens]
    encodings = doc.encoding
    seg_encoding = doc.seg_encoding
    token_count = len(doc.tokens)
    context_size = len(encodings)

    char_encodings = doc.char_encoding
    char_encoding = []
    char_count = []
    for char_encoding_token in char_encodings:
        char_count.append(len(char_encoding_token))
        char_encoding.append(torch.tensor(char_encoding_token,dtype=torch.long))
    char_encoding = util.padded_stack(char_encoding)
    token_masks_char = (char_encoding!=0).long()
    char_count = torch.tensor(char_count, dtype = torch.long)

    wordvec_encoding = [t.wordinx for t in doc.tokens]
    
    # import pdb; pdb.set_trace()
    # all tokens
    context2token_masks = []
    for t in doc.tokens:
        context2token_masks.append(create_entity_mask(*t.span, context_size))

    # create tensors
    # token indices
    encodings = torch.tensor(encodings, dtype=torch.long)
    seg_encoding = torch.tensor(seg_encoding, dtype=torch.long)
    
    pos_encoding = torch.tensor(pos_encoding, dtype=torch.long)
    wordvec_encoding = torch.tensor(wordvec_encoding, dtype=torch.long)

    # masking of tokens
    context_masks = torch.ones(context_size, dtype=torch.bool)
    token_masks = torch.ones(token_count, dtype=torch.bool)

    # also create samples_masks:
    # tensors to mask entity/relation samples of batch
    # since samples are stacked into batches, "padding" entities/relations possibly must be created
    # these are later masked during loss computation
    context2token_masks = torch.stack(context2token_masks)

    return dict(encodings=encodings, context_masks=context_masks, seg_encoding = seg_encoding, context2token_masks=context2token_masks, token_masks=token_masks, 
                pos_encoding = pos_encoding, wordvec_encoding = wordvec_encoding, char_encoding = char_encoding, token_masks_char = token_masks_char, char_count = char_count, meta_doc = doc)

def create_entity_mask(start, end, context_size):
    mask = torch.zeros(context_size, dtype=torch.bool)
    mask[start:end] = 1
    return mask

def collate_fn_padding(batch):
    batch = list(filter(lambda x: x is not None, batch))
    padded_batch = dict()
    keys = batch[0].keys()

    for key in keys:
        samples = [s[key] for s in batch]
        if key.startswith("meta"):
            padded_batch[key] = samples
            continue

        if not batch[0][key].shape:
            padded_batch[key] = torch.stack(samples)
        else:
            padded_batch[key] = util.padded_stack([s[key] for s in batch])

    return padded_batch
