'''Binary layout definitions for echoframe keys.

Defines the output type rank map and struct format strings for all
echoframe_key layouts. All formats are big-endian and fixed-width.

Field widths:
    model_id:         uint32  (4 bytes)
    output_type_id:   uint8   (1 byte)
    layer:            uint8   (1 byte)
    collar:           uint16  (2 bytes)
    phraser_key:      22 bytes (fixed)
    model_name_hash:  8 bytes  (fixed)
    tag_hash:         8 bytes  (fixed)
'''

import struct

OUTPUT_TYPE_RANK_MAP = {
    'hidden_state':     0,
    'attention':        1,
    'codebook_indices': 2,
    'codebook_matrix':  3,
}

RANK_OUTPUT_TYPE_MAP = {v: k for k, v in OUTPUT_TYPE_RANK_MAP.items()}

PHRASER_KEY_LEN   = 22
MODEL_NAME_HASH_LEN = 8
TAG_HASH_LEN        = 8


def make_key_fmt(output_type):
    '''Return the struct fmt string for one echoframe key type.'''
    if output_type in ('hidden_state', 'attention'):
        return f'>IBB{PHRASER_KEY_LEN}sH'
    if output_type == 'codebook_indices':
        return f'>IB{PHRASER_KEY_LEN}sH'
    if output_type == 'codebook_matrix':
        return '>IB'
    raise ValueError(f'unknown output type: {output_type!r}')


def key_len(output_type):
    '''Return the byte length of the echoframe_key for the given output type.'''
    return struct.calcsize(make_key_fmt(output_type))
