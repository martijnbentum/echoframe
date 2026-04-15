'''Tests for struct_helper binary layout definitions.'''

import struct
import unittest

from echoframe import struct_helper
from echoframe.struct_helper import (
    OUTPUT_TYPE_RANK_MAP,
    RANK_OUTPUT_TYPE_MAP,
    PHRASER_KEY_LEN,
    MODEL_NAME_HASH_LEN,
    TAG_HASH_LEN,
    make_key_fmt,
    key_len,
)


class TestRankMaps(unittest.TestCase):

    def test_output_type_rank_map_contains_all_types(self):
        expected = {
            'hidden_state', 'attention', 'codebook_indices',
            'codebook_matrix', 'model_metadata',
        }
        self.assertEqual(set(OUTPUT_TYPE_RANK_MAP.keys()), expected)

    def test_rank_output_type_map_is_inverse(self):
        for name, rank in OUTPUT_TYPE_RANK_MAP.items():
            self.assertEqual(RANK_OUTPUT_TYPE_MAP[rank], name)

    def test_ranks_are_unique(self):
        ranks = list(OUTPUT_TYPE_RANK_MAP.values())
        self.assertEqual(len(ranks), len(set(ranks)))


class TestKeyFmt(unittest.TestCase):

    def test_hidden_state_fmt(self):
        fmt = make_key_fmt('hidden_state')
        self.assertTrue(fmt.startswith('>'))
        self.assertEqual(struct.calcsize(fmt), 28)

    def test_attention_fmt_matches_hidden_state(self):
        self.assertEqual(make_key_fmt('attention'), make_key_fmt('hidden_state'))

    def test_codebook_indices_fmt(self):
        fmt = make_key_fmt('codebook_indices')
        self.assertEqual(struct.calcsize(fmt), 27)

    def test_codebook_matrix_fmt(self):
        fmt = make_key_fmt('codebook_matrix')
        self.assertEqual(struct.calcsize(fmt), 3)

    def test_model_metadata_fmt(self):
        fmt = make_key_fmt('model_metadata')
        self.assertEqual(struct.calcsize(fmt), 9)

    def test_unknown_output_type_raises(self):
        with self.assertRaises(ValueError):
            make_key_fmt('unknown_type')


class TestKeyLen(unittest.TestCase):

    def test_key_lengths(self):
        self.assertEqual(key_len('hidden_state'), 28)
        self.assertEqual(key_len('attention'), 28)
        self.assertEqual(key_len('codebook_indices'), 27)
        self.assertEqual(key_len('codebook_matrix'), 3)
        self.assertEqual(key_len('model_metadata'), 9)


class TestConstants(unittest.TestCase):

    def test_phraser_key_len(self):
        self.assertEqual(PHRASER_KEY_LEN, 22)

    def test_model_name_hash_len(self):
        self.assertEqual(MODEL_NAME_HASH_LEN, 8)

    def test_tag_hash_len(self):
        self.assertEqual(TAG_HASH_LEN, 8)


if __name__ == '__main__':
    unittest.main()
