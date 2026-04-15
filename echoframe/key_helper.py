'''Pack, unpack, and scan-key builders for echoframe binary keys.

All keys are big-endian fixed-width byte strings as defined in struct_helper.
Key building and interpretation live here; writing to the store is a store
responsibility.
'''

import hashlib
import struct

from . import struct_helper
from .struct_helper import (
    OUTPUT_TYPE_RANK_MAP,
    RANK_OUTPUT_TYPE_MAP,
    MODEL_NAME_HASH_LEN,
    TAG_HASH_LEN,
    PHRASER_KEY_LEN,
)

# pre-computed fmt strings and lengths
_HS_FMT  = struct_helper.make_key_fmt('hidden_state')
_AT_FMT  = struct_helper.make_key_fmt('attention')
_CI_FMT  = struct_helper.make_key_fmt('codebook_indices')
_CM_FMT  = struct_helper.make_key_fmt('codebook_matrix')
_MM_FMT  = struct_helper.make_key_fmt('model_metadata')

HIDDEN_STATE_KEY_LEN      = struct.calcsize(_HS_FMT)   # 28
ATTENTION_KEY_LEN         = struct.calcsize(_AT_FMT)   # 28
CODEBOOK_INDICES_KEY_LEN  = struct.calcsize(_CI_FMT)   # 27
CODEBOOK_MATRIX_KEY_LEN   = struct.calcsize(_CM_FMT)   # 3
MODEL_METADATA_KEY_LEN    = struct.calcsize(_MM_FMT)   # 9


# -------- hashing --------

def model_name_hash(model_name):
    '''Return the canonical 8-byte blake2b hash for a model name.
    model_name:   str
    '''
    return hashlib.blake2b(
        model_name.encode('utf-8'), digest_size=MODEL_NAME_HASH_LEN
    ).digest()


def tag_hash(tag):
    '''Return the canonical 8-byte blake2b hash for a tag string.
    tag:   str
    '''
    return hashlib.blake2b(
        tag.encode('utf-8'), digest_size=TAG_HASH_LEN
    ).digest()


# -------- pack --------

def pack_hidden_state_key(model_id, layer, phraser_key, collar):
    '''Pack an echoframe_key for a hidden_state record.
    model_id:     int  (uint16)
    layer:        int  (uint8)
    phraser_key:  bytes (22 bytes)
    collar:       int  (uint16)
    '''
    output_type_id = OUTPUT_TYPE_RANK_MAP['hidden_state']
    return struct.pack(_HS_FMT, model_id, output_type_id, layer,
                       phraser_key, collar)


def pack_attention_key(model_id, layer, phraser_key, collar):
    '''Pack an echoframe_key for an attention record.
    model_id:     int  (uint16)
    layer:        int  (uint8)
    phraser_key:  bytes (22 bytes)
    collar:       int  (uint16)
    '''
    output_type_id = OUTPUT_TYPE_RANK_MAP['attention']
    return struct.pack(_AT_FMT, model_id, output_type_id, layer,
                       phraser_key, collar)


def pack_codebook_indices_key(model_id, phraser_key, collar):
    '''Pack an echoframe_key for a codebook_indices record.
    model_id:     int  (uint16)
    phraser_key:  bytes (22 bytes)
    collar:       int  (uint16)
    '''
    output_type_id = OUTPUT_TYPE_RANK_MAP['codebook_indices']
    return struct.pack(_CI_FMT, model_id, output_type_id, phraser_key, collar)


def pack_codebook_matrix_key(model_id):
    '''Pack an echoframe_key for a codebook_matrix record.
    model_id:   int  (uint16)
    '''
    output_type_id = OUTPUT_TYPE_RANK_MAP['codebook_matrix']
    return struct.pack(_CM_FMT, model_id, output_type_id)


def pack_model_metadata_key(model_name):
    '''Pack an echoframe_key for a model_metadata record.
    model_name:   str
    '''
    name_hash = model_name_hash(model_name)
    output_type_id = OUTPUT_TYPE_RANK_MAP['model_metadata']
    return struct.pack(_MM_FMT, name_hash, output_type_id)


# -------- unpack --------

def unpack_hidden_state_key(key_bytes):
    '''Unpack a hidden_state echoframe_key into its component fields.
    Returns dict with model_id, output_type, layer, phraser_key, collar.
    '''
    model_id, output_type_id, layer, phraser_key, collar = struct.unpack(
        _HS_FMT, key_bytes)
    return {
        'model_id':    model_id,
        'output_type': RANK_OUTPUT_TYPE_MAP[output_type_id],
        'layer':       layer,
        'phraser_key': phraser_key,
        'collar':      collar,
    }


def unpack_attention_key(key_bytes):
    '''Unpack an attention echoframe_key into its component fields.
    Returns dict with model_id, output_type, layer, phraser_key, collar.
    '''
    model_id, output_type_id, layer, phraser_key, collar = struct.unpack(
        _AT_FMT, key_bytes)
    return {
        'model_id':    model_id,
        'output_type': RANK_OUTPUT_TYPE_MAP[output_type_id],
        'layer':       layer,
        'phraser_key': phraser_key,
        'collar':      collar,
    }


def unpack_codebook_indices_key(key_bytes):
    '''Unpack a codebook_indices echoframe_key into its component fields.
    Returns dict with model_id, output_type, phraser_key, collar.
    '''
    model_id, output_type_id, phraser_key, collar = struct.unpack(
        _CI_FMT, key_bytes)
    return {
        'model_id':    model_id,
        'output_type': RANK_OUTPUT_TYPE_MAP[output_type_id],
        'phraser_key': phraser_key,
        'collar':      collar,
    }


def unpack_codebook_matrix_key(key_bytes):
    '''Unpack a codebook_matrix echoframe_key into its component fields.
    Returns dict with model_id, output_type.
    '''
    model_id, output_type_id = struct.unpack(_CM_FMT, key_bytes)
    return {
        'model_id':    model_id,
        'output_type': RANK_OUTPUT_TYPE_MAP[output_type_id],
    }


def unpack_model_metadata_key(key_bytes):
    '''Unpack a model_metadata echoframe_key into its component fields.
    Returns dict with model_name_hash, output_type.
    '''
    name_hash, output_type_id = struct.unpack(_MM_FMT, key_bytes)
    return {
        'model_name_hash': name_hash,
        'output_type':     RANK_OUTPUT_TYPE_MAP[output_type_id],
    }


# -------- secondary scan-key builders --------

def make_tag_scan_key(tag, echoframe_key):
    '''Build a tag secondary scan key: tag_hash || echoframe_key.
    tag:            str
    echoframe_key:  bytes
    '''
    return tag_hash(tag) + echoframe_key


def make_output_type_scan_key(output_type, echoframe_key):
    '''Build an output-type secondary scan key: output_type_id || echoframe_key.
    output_type:    str
    echoframe_key:  bytes
    '''
    output_type_id = OUTPUT_TYPE_RANK_MAP[output_type]
    return struct.pack('>B', output_type_id) + echoframe_key


def make_tag_scan_prefix(tag):
    '''Return the prefix for scanning all records with a given tag.'''
    return tag_hash(tag)


def make_output_type_scan_prefix(output_type):
    '''Return the prefix for scanning all records of a given output type.'''
    output_type_id = OUTPUT_TYPE_RANK_MAP[output_type]
    return struct.pack('>B', output_type_id)
