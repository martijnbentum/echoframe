'''Tests for key_helper pack, unpack, hash, and scan-key builders.'''

import struct
import unittest

from echoframe import key_helper
from echoframe.key_helper import (
    model_name_hash,
    tag_hash,
    pack_hidden_state_key,
    pack_attention_key,
    pack_codebook_indices_key,
    pack_codebook_matrix_key,
    pack_model_metadata_key,
    unpack_hidden_state_key,
    unpack_attention_key,
    unpack_codebook_indices_key,
    unpack_codebook_matrix_key,
    unpack_model_metadata_key,
    make_tag_scan_key,
    make_output_type_scan_key,
    make_tag_scan_prefix,
    make_output_type_scan_prefix,
    HIDDEN_STATE_KEY_LEN,
    ATTENTION_KEY_LEN,
    CODEBOOK_INDICES_KEY_LEN,
    CODEBOOK_MATRIX_KEY_LEN,
    MODEL_METADATA_KEY_LEN,
)
from echoframe.struct_helper import OUTPUT_TYPE_RANK_MAP

SAMPLE_PHRASER_KEY = b'x' * 22
MODEL_ID = 7
LAYER = 3
COLLAR = 100


class TestHashing(unittest.TestCase):

    def test_model_name_hash_returns_8_bytes(self):
        result = model_name_hash('bert-base-uncased')
        self.assertIsInstance(result, bytes)
        self.assertEqual(len(result), 8)

    def test_model_name_hash_is_deterministic(self):
        self.assertEqual(
            model_name_hash('bert-base-uncased'),
            model_name_hash('bert-base-uncased'),
        )

    def test_model_name_hash_differs_for_different_names(self):
        self.assertNotEqual(
            model_name_hash('bert-base-uncased'),
            model_name_hash('wav2vec2-base'),
        )

    def test_tag_hash_returns_8_bytes(self):
        result = tag_hash('train')
        self.assertIsInstance(result, bytes)
        self.assertEqual(len(result), 8)

    def test_tag_hash_is_deterministic(self):
        self.assertEqual(tag_hash('train'), tag_hash('train'))

    def test_tag_hash_differs_for_different_tags(self):
        self.assertNotEqual(tag_hash('train'), tag_hash('test'))


class TestPackUnpackHiddenState(unittest.TestCase):

    def setUp(self):
        self.key = pack_hidden_state_key(
            MODEL_ID, LAYER, SAMPLE_PHRASER_KEY, COLLAR)

    def test_pack_returns_bytes(self):
        self.assertIsInstance(self.key, bytes)

    def test_pack_returns_correct_length(self):
        self.assertEqual(len(self.key), HIDDEN_STATE_KEY_LEN)
        self.assertEqual(len(self.key), 28)

    def test_pack_is_big_endian(self):
        model_id_bytes = self.key[:2]
        self.assertEqual(struct.unpack('>H', model_id_bytes)[0], MODEL_ID)

    def test_unpack_roundtrip(self):
        fields = unpack_hidden_state_key(self.key)
        self.assertEqual(fields['model_id'], MODEL_ID)
        self.assertEqual(fields['output_type'], 'hidden_state')
        self.assertEqual(fields['layer'], LAYER)
        self.assertEqual(fields['phraser_key'], SAMPLE_PHRASER_KEY)
        self.assertEqual(fields['collar'], COLLAR)


class TestPackUnpackAttention(unittest.TestCase):

    def setUp(self):
        self.key = pack_attention_key(
            MODEL_ID, LAYER, SAMPLE_PHRASER_KEY, COLLAR)

    def test_pack_returns_correct_length(self):
        self.assertEqual(len(self.key), ATTENTION_KEY_LEN)
        self.assertEqual(len(self.key), 28)

    def test_unpack_roundtrip(self):
        fields = unpack_attention_key(self.key)
        self.assertEqual(fields['model_id'], MODEL_ID)
        self.assertEqual(fields['output_type'], 'attention')
        self.assertEqual(fields['layer'], LAYER)
        self.assertEqual(fields['phraser_key'], SAMPLE_PHRASER_KEY)
        self.assertEqual(fields['collar'], COLLAR)

    def test_attention_and_hidden_state_differ_by_output_type_id(self):
        hs_key = pack_hidden_state_key(
            MODEL_ID, LAYER, SAMPLE_PHRASER_KEY, COLLAR)
        at_key = pack_attention_key(
            MODEL_ID, LAYER, SAMPLE_PHRASER_KEY, COLLAR)
        self.assertNotEqual(hs_key, at_key)


class TestPackUnpackCodebookIndices(unittest.TestCase):

    def setUp(self):
        self.key = pack_codebook_indices_key(
            MODEL_ID, SAMPLE_PHRASER_KEY, COLLAR)

    def test_pack_returns_correct_length(self):
        self.assertEqual(len(self.key), CODEBOOK_INDICES_KEY_LEN)
        self.assertEqual(len(self.key), 27)

    def test_unpack_roundtrip(self):
        fields = unpack_codebook_indices_key(self.key)
        self.assertEqual(fields['model_id'], MODEL_ID)
        self.assertEqual(fields['output_type'], 'codebook_indices')
        self.assertEqual(fields['phraser_key'], SAMPLE_PHRASER_KEY)
        self.assertEqual(fields['collar'], COLLAR)


class TestPackUnpackCodebookMatrix(unittest.TestCase):

    def setUp(self):
        self.key = pack_codebook_matrix_key(MODEL_ID)

    def test_pack_returns_correct_length(self):
        self.assertEqual(len(self.key), CODEBOOK_MATRIX_KEY_LEN)
        self.assertEqual(len(self.key), 3)

    def test_unpack_roundtrip(self):
        fields = unpack_codebook_matrix_key(self.key)
        self.assertEqual(fields['model_id'], MODEL_ID)
        self.assertEqual(fields['output_type'], 'codebook_matrix')


class TestPackUnpackModelMetadata(unittest.TestCase):

    def setUp(self):
        self.model_name = 'bert-base-uncased'
        self.key = pack_model_metadata_key(self.model_name)

    def test_pack_returns_correct_length(self):
        self.assertEqual(len(self.key), MODEL_METADATA_KEY_LEN)
        self.assertEqual(len(self.key), 9)

    def test_unpack_roundtrip(self):
        fields = unpack_model_metadata_key(self.key)
        self.assertEqual(fields['model_name_hash'],
                         model_name_hash(self.model_name))
        self.assertEqual(fields['output_type'], 'model_metadata')

    def test_same_name_produces_same_key(self):
        self.assertEqual(
            pack_model_metadata_key(self.model_name),
            pack_model_metadata_key(self.model_name),
        )

    def test_different_names_produce_different_keys(self):
        self.assertNotEqual(
            pack_model_metadata_key('bert-base-uncased'),
            pack_model_metadata_key('wav2vec2-base'),
        )


class TestSecondaryTagScanKey(unittest.TestCase):

    def setUp(self):
        self.echoframe_key = pack_hidden_state_key(
            MODEL_ID, LAYER, SAMPLE_PHRASER_KEY, COLLAR)

    def test_make_tag_scan_key_starts_with_tag_hash(self):
        scan_key = make_tag_scan_key('train', self.echoframe_key)
        self.assertTrue(scan_key.startswith(tag_hash('train')))

    def test_make_tag_scan_key_ends_with_echoframe_key(self):
        scan_key = make_tag_scan_key('train', self.echoframe_key)
        self.assertTrue(scan_key.endswith(self.echoframe_key))

    def test_make_tag_scan_key_length(self):
        scan_key = make_tag_scan_key('train', self.echoframe_key)
        self.assertEqual(len(scan_key), 8 + len(self.echoframe_key))

    def test_make_tag_scan_prefix_returns_tag_hash(self):
        prefix = make_tag_scan_prefix('train')
        self.assertEqual(prefix, tag_hash('train'))
        self.assertEqual(len(prefix), 8)

    def test_different_tags_produce_different_prefixes(self):
        self.assertNotEqual(
            make_tag_scan_prefix('train'),
            make_tag_scan_prefix('test'),
        )


class TestSecondaryOutputTypeScanKey(unittest.TestCase):

    def setUp(self):
        self.echoframe_key = pack_hidden_state_key(
            MODEL_ID, LAYER, SAMPLE_PHRASER_KEY, COLLAR)

    def test_make_output_type_scan_key_starts_with_output_type_id(self):
        scan_key = make_output_type_scan_key('hidden_state', self.echoframe_key)
        expected_byte = struct.pack(
            '>B', OUTPUT_TYPE_RANK_MAP['hidden_state'])
        self.assertTrue(scan_key.startswith(expected_byte))

    def test_make_output_type_scan_key_ends_with_echoframe_key(self):
        scan_key = make_output_type_scan_key('hidden_state', self.echoframe_key)
        self.assertTrue(scan_key.endswith(self.echoframe_key))

    def test_make_output_type_scan_key_length(self):
        scan_key = make_output_type_scan_key('hidden_state', self.echoframe_key)
        self.assertEqual(len(scan_key), 1 + len(self.echoframe_key))

    def test_make_output_type_scan_prefix(self):
        prefix = make_output_type_scan_prefix('hidden_state')
        expected = struct.pack('>B', OUTPUT_TYPE_RANK_MAP['hidden_state'])
        self.assertEqual(prefix, expected)
        self.assertEqual(len(prefix), 1)

    def test_different_output_types_produce_different_prefixes(self):
        self.assertNotEqual(
            make_output_type_scan_prefix('hidden_state'),
            make_output_type_scan_prefix('attention'),
        )


class TestKeyOrdering(unittest.TestCase):
    '''Verify that keys sort in lexicographic order as expected.'''

    def test_hidden_state_keys_sort_by_model_id_first(self):
        key_1 = pack_hidden_state_key(1, 0, SAMPLE_PHRASER_KEY, 0)
        key_2 = pack_hidden_state_key(2, 0, SAMPLE_PHRASER_KEY, 0)
        self.assertLess(key_1, key_2)

    def test_hidden_state_keys_sort_by_layer_within_model(self):
        key_l0 = pack_hidden_state_key(MODEL_ID, 0, SAMPLE_PHRASER_KEY, 0)
        key_l1 = pack_hidden_state_key(MODEL_ID, 1, SAMPLE_PHRASER_KEY, 0)
        self.assertLess(key_l0, key_l1)

    def test_hidden_state_keys_sort_by_collar_within_layer(self):
        key_c0 = pack_hidden_state_key(MODEL_ID, LAYER, SAMPLE_PHRASER_KEY, 0)
        key_c1 = pack_hidden_state_key(MODEL_ID, LAYER, SAMPLE_PHRASER_KEY, 1)
        self.assertLess(key_c0, key_c1)


if __name__ == '__main__':
    unittest.main()
