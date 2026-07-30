'''Tests for acoustic-feature storage without model metadata fiction.'''

from __future__ import annotations

import tempfile
import unittest
from unittest import mock

import numpy as np

import echoframe
from echoframe.acoustic_features import (
    make_acoustic_feature_item,
    store_mfcc,
    store_mfcc_batch,
)
from echoframe.metadata import EchoframeMetadata, filter_metadata
from tests.helpers import make_fake_store, pk


class FakeSegment:
    def __init__(self, key, matrix):
        self.key = key
        self._mfcc = np.asarray(matrix)
        self.mfcc_load_count = 0

    @property
    def mfcc(self):
        self.mfcc_load_count += 1
        return self._mfcc


class TestAcousticFeatureKeys(unittest.TestCase):
    def test_store_builds_key_without_registered_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            key = store.make_echoframe_key('acoustic_feature',
                feature_name='mfcc', phraser_key=pk('segment-1'))

        self.assertIsInstance(key, bytes)
        self.assertEqual(store.model_registry.model_metadatas, [])

    def test_store_rejects_model_fields_for_acoustic_feature(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            with self.assertRaisesRegex(ValueError,
                'does not accept model_name'):
                store.make_echoframe_key('acoustic_feature',
                    model_name='mfcc', feature_name='mfcc',
                    phraser_key=pk('segment-1'))

    def test_metadata_rejects_feature_name_that_does_not_match_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            key = store.make_echoframe_key('acoustic_feature',
                feature_name='mfcc', phraser_key=pk('segment-1'))
            with self.assertRaisesRegex(ValueError,
                'does not match echoframe_key'):
                EchoframeMetadata(key, feature_name='pitch')


class TestMfccStorage(unittest.TestCase):
    def test_store_helpers_reject_none_store(self):
        segment = FakeSegment(pk('segment-1'), np.ones((3, 4)))
        for helper, value in (
                (store_mfcc, segment),
                (store_mfcc_batch, [segment])):
            with self.subTest(helper=helper.__name__):
                with self.assertRaisesRegex(ValueError, 'echoframe Store'):
                    helper(value, None)

    def test_store_mfcc_uses_honest_metadata_and_storage_paths(self):
        matrix = np.arange(12, dtype='float32').reshape(3, 4)
        segment = FakeSegment(pk('segment-1'), matrix)
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            with mock.patch.object(store.phraser_registry,
                'segment_to_source_id', return_value='cgn-main'):
                store_mfcc(segment, store, tags=['acoustic'])
            key = store.make_echoframe_key('acoustic_feature',
                feature_name='mfcc', phraser_key=segment.key)
            metadata = store.load_metadata(key)
            payload = store.phraser_key_to_acoustic_feature(
                segment.key, 'mfcc')

        self.assertEqual(metadata.output_type, 'acoustic_feature')
        self.assertEqual(metadata.feature_name, 'mfcc')
        self.assertIsNone(metadata.model_name)
        self.assertIsNone(metadata.model_id)
        self.assertIsNone(metadata.layer)
        self.assertIsNone(metadata.collar)
        self.assertEqual(metadata.phraser_source_id, 'cgn-main')
        self.assertEqual(metadata.shard_id,
            'mfcc_acoustic_feature_0001')
        self.assertTrue(metadata.dataset_path.startswith('/items/'))
        np.testing.assert_array_equal(payload, matrix)
        self.assertEqual(store.model_registry.model_metadatas, [])

    def test_store_mfcc_does_not_recompute_existing_feature(self):
        segment = FakeSegment(pk('segment-1'), np.ones((3, 4)))
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            with mock.patch.object(store.phraser_registry,
                'segment_to_source_id', return_value='cgn-main'):
                store_mfcc(segment, store)
                store_mfcc(segment, store)

        self.assertEqual(segment.mfcc_load_count, 1)

    def test_delete_phraser_key_filters_by_feature_name_without_model(self):
        segment = FakeSegment(pk('segment-1'), np.ones((3, 4)))
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            with mock.patch.object(store.phraser_registry,
                'segment_to_source_id', return_value='cgn-main'):
                store_mfcc(segment, store)
            store.delete_phraser_key(segment.key,
                output_type='acoustic_feature', feature_name='mfcc',
                verbose=False)
            payload = store.phraser_key_to_acoustic_feature(
                segment.key, 'mfcc')

        self.assertIsNone(payload)

    def test_compaction_preserves_non_layered_dataset_path(self):
        segment = FakeSegment(pk('segment-1'), np.ones((3, 4)))
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            with mock.patch.object(store.phraser_registry,
                'segment_to_source_id', return_value='cgn-main'):
                store_mfcc(segment, store)
            key = store.make_echoframe_key('acoustic_feature',
                feature_name='mfcc', phraser_key=segment.key)
            metadata = store.load_metadata(key)
            updated = store.storage.compact_shard_to(
                metadata.shard_id, [metadata],
                target_shard_id='mfcc_acoustic_feature_0002',
                delete_source=False)[0]
            payload = store.storage.load(updated)

        self.assertTrue(updated.dataset_path.startswith('/items/'))
        np.testing.assert_array_equal(payload, np.ones((3, 4)))

    def test_batch_stores_only_missing_features(self):
        segments = [
            FakeSegment(pk('segment-1'), np.ones((2, 3))),
            FakeSegment(pk('segment-2'), np.full((2, 3), 2)),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            with mock.patch.object(store.phraser_registry,
                    'segments_to_source_id', return_value='cgn-main'), \
                    mock.patch('echoframe.acoustic_features.progressbar',
                        side_effect=lambda values: values):
                store_mfcc_batch(segments, store, verbose=False)
                store_mfcc_batch(segments, store, verbose=False)
            payloads = store.phraser_keys_to_acoustic_features(
                [segment.key for segment in segments], 'mfcc')

        self.assertEqual([segment.mfcc_load_count for segment in segments],
            [1, 1])
        np.testing.assert_array_equal(payloads[0], np.ones((2, 3)))
        np.testing.assert_array_equal(payloads[1], np.full((2, 3), 2))

    def test_item_requires_matrix_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            key = store.make_echoframe_key('acoustic_feature',
                feature_name='mfcc', phraser_key=pk('segment-1'))
            with self.assertRaisesRegex(ValueError, 'must be 2D'):
                make_acoustic_feature_item(key, 'mfcc', np.ones(4), store,
                    phraser_source_id='cgn-main')

    def test_feature_name_filter_distinguishes_feature_kinds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = make_fake_store(tmpdir)
            records = []
            for feature_name in ('mfcc', 'pitch'):
                key = store.make_echoframe_key('acoustic_feature',
                    feature_name=feature_name, phraser_key=pk('segment-1'))
                metadata = EchoframeMetadata(key,
                    feature_name=feature_name,
                    phraser_source_id='cgn-main')
                records.append(metadata)

        result = filter_metadata(records, output_type='acoustic_feature',
            feature_name='mfcc')
        self.assertEqual([metadata.feature_name for metadata in result],
            ['mfcc'])


class TestAcousticFeaturePublicApi(unittest.TestCase):
    def test_public_helpers_are_exported(self):
        self.assertIs(echoframe.store_mfcc, store_mfcc)
        self.assertIs(echoframe.store_mfcc_batch, store_mfcc_batch)
        self.assertFalse(hasattr(echoframe, '_store_acoustic_feature'))


if __name__ == '__main__':
    unittest.main()
