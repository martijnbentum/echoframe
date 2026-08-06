'''Pack, unpack, and scan-key builders for echoframe binary keys.

All keys are big-endian fixed-width byte strings as defined in struct_helper.
Key building and interpretation live here; writing to the store is a store
responsibility.
'''

import hashlib
import struct

from . import struct_helper

# pre-computed fmt strings and lengths
_HS_FMT = struct_helper.make_key_fmt('hidden_state')
_AT_FMT = struct_helper.make_key_fmt('attention')
_CI_FMT = struct_helper.make_key_fmt('codebook_indices')
_CM_FMT = struct_helper.make_key_fmt('codebook_matrix')
_AF_FMT = struct_helper.make_key_fmt('acoustic_feature')
_CNN_FMT = struct_helper.make_key_fmt('cnn')

HIDDEN_STATE_KEY_LEN = struct.calcsize(_HS_FMT)   # 30
ATTENTION_KEY_LEN = struct.calcsize(_AT_FMT)   # 30
CODEBOOK_INDICES_KEY_LEN = struct.calcsize(_CI_FMT)   # 29
CODEBOOK_MATRIX_KEY_LEN = struct.calcsize(_CM_FMT)   # 5
ACOUSTIC_FEATURE_KEY_LEN = struct.calcsize(_AF_FMT)   # 31
CNN_KEY_LEN = struct.calcsize(_CNN_FMT)   # 29


# -------- hashing --------

def model_name_hash(model_name):
    '''Return the canonical 8-byte blake2b hash for a model name.
    model_name:   str
    '''
    return hashlib.blake2b(
        model_name.encode('utf-8'),
        digest_size=struct_helper.MODEL_NAME_HASH_LEN
    ).digest()


def feature_name_hash(feature_name):
    '''Return the canonical 8-byte blake2b hash for a feature name.
    feature_name:   str
    '''
    encoded_name = feature_name.encode('utf-8')
    hasher = hashlib.blake2b(encoded_name,
        digest_size=struct_helper.FEATURE_NAME_HASH_LEN)
    return hasher.digest()


def tag_hash(tag):
    '''Return the canonical 8-byte blake2b hash for a tag string.
    tag:   str
    '''
    return hashlib.blake2b(
        tag.encode('utf-8'),
        digest_size=struct_helper.TAG_HASH_LEN
    ).digest()


# -------- validation --------

def validate_segment_phraser_key(phraser_key):
    '''Return a validated segment phraser_key as bytes.

    Segment-based echoframe keys embed the raw fixed-width phraser segment
    key. The value must therefore already be binary and exactly
    ``PHRASER_KEY_LEN`` bytes long.
    '''
    if not isinstance(phraser_key, (bytes, bytearray)):
        raise ValueError('phraser_key must be bytes for segment-based keys')
    if len(phraser_key) != struct_helper.PHRASER_KEY_LEN:
        raise ValueError(
            'phraser_key must be exactly '
            f'{struct_helper.PHRASER_KEY_LEN} bytes')
    return bytes(phraser_key)


# -------- pack --------

def pack_hidden_state_key(model_id, layer, phraser_key, collar):
    '''Pack an echoframe_key for a hidden_state record.
    model_id:     int  (uint32)
    layer:        int  (uint8)
    phraser_key:  bytes (22 bytes)
    collar:       int  (uint16)
    '''
    _require_key_fields('hidden_state', model_id=model_id, layer=layer,
        collar=collar)
    phraser_key = validate_segment_phraser_key(phraser_key)
    output_type_id = struct_helper.OUTPUT_TYPE_RANK_MAP['hidden_state']
    return struct.pack(_HS_FMT, model_id, output_type_id, layer,
        phraser_key, collar)


def pack_attention_key(model_id, layer, phraser_key, collar):
    '''Pack an echoframe_key for an attention record.
    model_id:     int  (uint32)
    layer:        int  (uint8)
    phraser_key:  bytes (22 bytes)
    collar:       int  (uint16)
    '''
    _require_key_fields('attention', model_id=model_id, layer=layer,
        collar=collar)
    phraser_key = validate_segment_phraser_key(phraser_key)
    output_type_id = struct_helper.OUTPUT_TYPE_RANK_MAP['attention']
    return struct.pack(_AT_FMT, model_id, output_type_id, layer,
        phraser_key, collar)


def pack_codebook_indices_key(model_id, phraser_key, collar):
    '''Pack an echoframe_key for a codebook_indices record.
    model_id:     int  (uint32)
    phraser_key:  bytes (22 bytes)
    collar:       int  (uint16)
    '''
    _require_key_fields('codebook_indices', model_id=model_id, collar=collar)
    phraser_key = validate_segment_phraser_key(phraser_key)
    output_type_id = struct_helper.OUTPUT_TYPE_RANK_MAP['codebook_indices']
    return struct.pack(_CI_FMT, model_id, output_type_id, phraser_key,
        collar)


def pack_codebook_matrix_key(model_id):
    '''Pack an echoframe_key for a codebook_matrix record.
    model_id:   int  (uint32)
    '''
    _require_key_fields('codebook_matrix', model_id=model_id)
    output_type_id = struct_helper.OUTPUT_TYPE_RANK_MAP['codebook_matrix']
    return struct.pack(_CM_FMT, model_id, output_type_id)


def pack_acoustic_feature_key(feature_name, phraser_key):
    '''Pack an echoframe_key for an acoustic_feature record.
    feature_name:  acoustic feature identifier, such as mfcc
    phraser_key:   bytes (22 bytes)
    '''
    _require_key_fields('acoustic_feature', feature_name=feature_name)
    if not isinstance(feature_name, str) or not feature_name.strip():
        message = 'feature_name must be a non-empty string'
        raise ValueError(message)
    phraser_key = validate_segment_phraser_key(phraser_key)
    output_type_id = struct_helper.OUTPUT_TYPE_RANK_MAP['acoustic_feature']
    name_hash = feature_name_hash(feature_name)
    return struct.pack(_AF_FMT, name_hash, output_type_id, phraser_key)


def pack_cnn_key(model_id, phraser_key, collar):
    '''Pack an echoframe_key for a cnn record.
    model_id:     int  (uint32)
    phraser_key:  bytes (22 bytes)
    collar:       int  (uint16)
    '''
    _require_key_fields('cnn', model_id=model_id, collar=collar)
    phraser_key = validate_segment_phraser_key(phraser_key)
    output_type_id = struct_helper.OUTPUT_TYPE_RANK_MAP['cnn']
    return struct.pack(_CNN_FMT, model_id, output_type_id, phraser_key,
        collar)


def pack_echoframe_key(output_type, model_id=None, phraser_key=None,
    layer=None, collar=None, feature_name=None):
    '''Pack one echoframe_key by output type.
    output_type:  one of the supported echoframe output types
    model_id:     model identifier
    phraser_key:  segment phraser key when required
    layer:        layer when required
    collar:       collar when required
    feature_name:  acoustic feature identifier when required
    '''
    if output_type == 'hidden_state':
        _reject_unused_key_fields(output_type, feature_name=feature_name)
        return pack_hidden_state_key(model_id, layer, phraser_key, collar)
    if output_type == 'attention':
        _reject_unused_key_fields(output_type, feature_name=feature_name)
        return pack_attention_key(model_id, layer, phraser_key, collar)
    if output_type == 'codebook_indices':
        _reject_unused_key_fields(output_type, layer=layer,
            feature_name=feature_name)
        return pack_codebook_indices_key(model_id, phraser_key, collar)
    if output_type == 'codebook_matrix':
        _reject_unused_key_fields(output_type, phraser_key=phraser_key,
            layer=layer, collar=collar, feature_name=feature_name)
        return pack_codebook_matrix_key(model_id)
    if output_type == 'acoustic_feature':
        _reject_unused_key_fields(output_type, model_id=model_id,
            layer=layer, collar=collar)
        return pack_acoustic_feature_key(feature_name, phraser_key)
    if output_type == 'cnn':
        _reject_unused_key_fields(output_type, layer=layer,
            feature_name=feature_name)
        return pack_cnn_key(model_id, phraser_key, collar)
    raise ValueError(f'unknown output type: {output_type!r}')


def _reject_unused_key_fields(output_type, **fields):
    '''Raise when fields that this output type does not encode are set.'''
    for name, value in fields.items():
        if value is not None:
            message = f'{output_type} keys do not encode {name}, '
            message += f'got {name}={value!r}'
            raise ValueError(message)


def _require_key_fields(output_type, **fields):
    '''Raise when fields that this output type encodes are missing.'''
    for name, value in fields.items():
        if value is None:
            message = f'{output_type} keys require {name}, got None'
            raise ValueError(message)


# -------- unpack --------

def unpack_hidden_state_key(key_bytes):
    '''Unpack a hidden_state echoframe_key into its component fields.
    Returns dict with model_id, output_type, layer, phraser_key, collar.
    '''
    model_id, output_type_id, layer, phraser_key, collar = struct.unpack(
        _HS_FMT, key_bytes)
    return {
        'model_id': model_id,
        'output_type': struct_helper.RANK_OUTPUT_TYPE_MAP[output_type_id],
        'layer': layer,
        'phraser_key': phraser_key,
        'collar': collar,
    }


def unpack_attention_key(key_bytes):
    '''Unpack an attention echoframe_key into its component fields.
    Returns dict with model_id, output_type, layer, phraser_key, collar.
    '''
    model_id, output_type_id, layer, phraser_key, collar = struct.unpack(
        _AT_FMT, key_bytes)
    return {
        'model_id': model_id,
        'output_type': struct_helper.RANK_OUTPUT_TYPE_MAP[output_type_id],
        'layer': layer,
        'phraser_key': phraser_key,
        'collar': collar,
    }


def unpack_codebook_indices_key(key_bytes):
    '''Unpack a codebook_indices echoframe_key into its component fields.
    Returns dict with model_id, output_type, phraser_key, collar.
    '''
    model_id, output_type_id, phraser_key, collar = struct.unpack(
        _CI_FMT, key_bytes)
    return {
        'model_id': model_id,
        'output_type': struct_helper.RANK_OUTPUT_TYPE_MAP[output_type_id],
        'phraser_key': phraser_key,
        'collar': collar,
    }


def unpack_codebook_matrix_key(key_bytes):
    '''Unpack a codebook_matrix echoframe_key into its component fields.
    Returns dict with model_id, output_type.
    '''
    model_id, output_type_id = struct.unpack(_CM_FMT, key_bytes)
    return {
        'model_id': model_id,
        'output_type': struct_helper.RANK_OUTPUT_TYPE_MAP[output_type_id],
    }


def unpack_acoustic_feature_key(key_bytes):
    '''Unpack an acoustic feature key into its component fields.'''
    name_hash, output_type_id, phraser_key = struct.unpack(_AF_FMT, key_bytes)
    return {
        'feature_name_hash': name_hash,
        'output_type': struct_helper.RANK_OUTPUT_TYPE_MAP[output_type_id],
        'phraser_key': phraser_key,
    }


def unpack_cnn_key(key_bytes):
    '''Unpack a cnn echoframe_key into its component fields.
    Returns dict with model_id, output_type, phraser_key, collar.
    '''
    model_id, output_type_id, phraser_key, collar = struct.unpack(
        _CNN_FMT, key_bytes)
    return {
        'model_id': model_id,
        'output_type': struct_helper.RANK_OUTPUT_TYPE_MAP[output_type_id],
        'phraser_key': phraser_key,
        'collar': collar,
    }


def unpack_echoframe_key(key_bytes):
    '''Unpack one echoframe_key by inferring its output type.'''
    output_type = output_type_from_echoframe_key(key_bytes)
    if output_type == 'hidden_state':
        return unpack_hidden_state_key(key_bytes)
    if output_type == 'attention':
        return unpack_attention_key(key_bytes)
    if output_type == 'codebook_indices':
        return unpack_codebook_indices_key(key_bytes)
    if output_type == 'codebook_matrix':
        return unpack_codebook_matrix_key(key_bytes)
    if output_type == 'acoustic_feature':
        return unpack_acoustic_feature_key(key_bytes)
    if output_type == 'cnn': return unpack_cnn_key(key_bytes)
    raise ValueError(f'unknown output type: {output_type!r}')


def output_type_from_echoframe_key(key_bytes):
    '''Infer the output type from a fixed-width echoframe_key.'''
    n_bytes = len(key_bytes)
    if n_bytes == ACOUSTIC_FEATURE_KEY_LEN:
        output_type_id = key_bytes[struct_helper.FEATURE_NAME_HASH_LEN]
    elif n_bytes in {HIDDEN_STATE_KEY_LEN, ATTENTION_KEY_LEN,
            CODEBOOK_INDICES_KEY_LEN, CODEBOOK_MATRIX_KEY_LEN}:
        output_type_id = key_bytes[4]
    else:
        raise ValueError(f'unknown echoframe_key length: {n_bytes}')

    try:
        output_type = struct_helper.RANK_OUTPUT_TYPE_MAP[output_type_id]
    except KeyError as exc:
        raise ValueError(
            f'unknown output_type_id in echoframe_key: {output_type_id}'
        ) from exc

    if n_bytes == HIDDEN_STATE_KEY_LEN and output_type not in {
            'hidden_state', 'attention'}:
        raise ValueError(
            f'invalid output_type_id for 30-byte echoframe_key: '
            f'{output_type_id}')
    if n_bytes == CODEBOOK_INDICES_KEY_LEN and output_type not in {
            'codebook_indices', 'cnn'}:
        raise ValueError(
            f'invalid output_type_id for 29-byte echoframe_key: '
            f'{output_type_id}')
    if n_bytes == CODEBOOK_MATRIX_KEY_LEN and output_type != (
            'codebook_matrix'):
        raise ValueError(
            f'invalid output_type_id for 5-byte echoframe_key: '
            f'{output_type_id}')
    if n_bytes == ACOUSTIC_FEATURE_KEY_LEN and output_type != (
            'acoustic_feature'):
        raise ValueError(
            f'invalid output_type_id for 31-byte echoframe_key: '
            f'{output_type_id}')
    return output_type


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
    output_type_id = struct_helper.OUTPUT_TYPE_RANK_MAP[output_type]
    return struct.pack('>B', output_type_id) + echoframe_key


def make_tag_scan_prefix(tag):
    '''Return the prefix for scanning all records with a given tag.'''
    return tag_hash(tag)


def make_output_type_scan_prefix(output_type):
    '''Return the prefix for scanning all records of a given output type.'''
    output_type_id = struct_helper.OUTPUT_TYPE_RANK_MAP[output_type]
    return struct.pack('>B', output_type_id)
