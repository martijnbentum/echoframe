'''Binary layout definitions for echoframe keys.

Defines the output type rank map and struct format strings for all
echoframe_key layouts. All formats are big-endian and fixed-width.

Field widths:
    model_id:         uint16  (2 bytes)
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
    'model_metadata':   4,
}

RANK_OUTPUT_TYPE_MAP = {v: k for k, v in OUTPUT_TYPE_RANK_MAP.items()}

PHRASER_KEY_LEN   = 22
MODEL_NAME_HASH_LEN = 8
TAG_HASH_LEN        = 8


def make_key_fmt(output_type):
    '''Return the struct fmt string for the given output type echoframe_key.

    hidden_state / attention:
        uint16 model_id, uint8 output_type_id, uint8 layer,
        22-byte phraser_key, uint16 collar
    codebook_indices:
        uint16 model_id, uint8 output_type_id,
        22-byte phraser_key, uint16 collar
    codebook_matrix:
        uint16 model_id, uint8 output_type_id
    model_metadata:
        8-byte model_name_hash, uint8 output_type_id
    '''
    if output_type in ('hidden_state', 'attention'):
        return f'>HBB{PHRASER_KEY_LEN}sH'
    if output_type == 'codebook_indices':
        return f'>HB{PHRASER_KEY_LEN}sH'
    if output_type == 'codebook_matrix':
        return '>HB'
    if output_type == 'model_metadata':
        return f'>{MODEL_NAME_HASH_LEN}sB'
    raise ValueError(f'unknown output type: {output_type!r}')


def key_len(output_type):
    '''Return the byte length of the echoframe_key for the given output type.'''
    return struct.calcsize(make_key_fmt(output_type))
