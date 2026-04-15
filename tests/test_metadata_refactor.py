'''Tests for the metadata subclass refactor groundwork.'''

import unittest

from echoframe.metadata import (
    AttentionMetadata,
    CodebookIndicesMetadata,
    CodebookMatrixMetadata,
    EchoframeMetadata,
    HiddenStateMetadata,
    ModelMetadata,
    metadata_class_for_output_type,
)


class TestMetadataClassDispatch(unittest.TestCase):

    def test_hidden_state_dispatch(self):
        metadata = EchoframeMetadata.from_dict({
            'phraser_key': 'pk1',
            'collar': 0,
            'model_name': 'bert',
            'model_id': None,
            'output_type': 'hidden_state',
            'layer': 1,
            'local_path': None,
            'huggingface_id': None,
            'language': None,
            'storage_status': 'live',
            'shard_id': None,
            'dataset_path': None,
            'shape': None,
            'dtype': None,
            'tags': [],
            'created_at': None,
            'deleted_at': None,
            'accessed_at': None,
        })
        self.assertIsInstance(metadata, HiddenStateMetadata)

    def test_attention_dispatch(self):
        metadata = EchoframeMetadata.from_dict({
            'phraser_key': 'pk1',
            'collar': 0,
            'model_name': 'bert',
            'model_id': None,
            'output_type': 'attention',
            'layer': 1,
            'local_path': None,
            'huggingface_id': None,
            'language': None,
            'storage_status': 'live',
            'shard_id': None,
            'dataset_path': None,
            'shape': None,
            'dtype': None,
            'tags': [],
            'created_at': None,
            'deleted_at': None,
            'accessed_at': None,
        })
        self.assertIsInstance(metadata, AttentionMetadata)

    def test_codebook_indices_dispatch(self):
        metadata = EchoframeMetadata.from_dict({
            'phraser_key': 'pk1',
            'collar': 0,
            'model_name': 'bert',
            'model_id': None,
            'output_type': 'codebook_indices',
            'layer': 0,
            'local_path': None,
            'huggingface_id': None,
            'language': None,
            'storage_status': 'live',
            'shard_id': None,
            'dataset_path': None,
            'shape': None,
            'dtype': None,
            'tags': [],
            'created_at': None,
            'deleted_at': None,
            'accessed_at': None,
        })
        self.assertIsInstance(metadata, CodebookIndicesMetadata)

    def test_codebook_matrix_dispatch(self):
        metadata = EchoframeMetadata.from_dict({
            'phraser_key': 'pk1',
            'collar': 0,
            'model_name': 'bert',
            'model_id': 3,
            'output_type': 'codebook_matrix',
            'layer': 0,
            'local_path': None,
            'huggingface_id': None,
            'language': None,
            'storage_status': 'live',
            'shard_id': None,
            'dataset_path': None,
            'shape': None,
            'dtype': None,
            'tags': [],
            'created_at': None,
            'deleted_at': None,
            'accessed_at': None,
        })
        self.assertIsInstance(metadata, CodebookMatrixMetadata)

    def test_model_metadata_dispatch(self):
        metadata = EchoframeMetadata.from_dict({
            'phraser_key': None,
            'collar': None,
            'model_name': 'bert-base-uncased',
            'model_id': 4,
            'output_type': 'model_metadata',
            'layer': None,
            'local_path': '/tmp/bert',
            'huggingface_id': 'bert-base-uncased',
            'language': 'en',
            'storage_status': 'live',
            'shard_id': None,
            'dataset_path': None,
            'shape': None,
            'dtype': None,
            'tags': [],
            'created_at': None,
            'deleted_at': None,
            'accessed_at': None,
        })
        self.assertIsInstance(metadata, ModelMetadata)


class TestSubclassCopyHelpers(unittest.TestCase):

    def test_with_tags_preserves_hidden_state_subclass(self):
        metadata = HiddenStateMetadata(
            phraser_key='pk1', collar=0, model_name='bert', layer=1)
        updated = metadata.with_tags(['a'])
        self.assertIsInstance(updated, HiddenStateMetadata)

    def test_mark_deleted_preserves_model_metadata_subclass(self):
        metadata = ModelMetadata(model_name='bert-base-uncased', model_id=0)
        deleted = metadata.mark_deleted()
        self.assertIsInstance(deleted, ModelMetadata)
        self.assertEqual(deleted.storage_status, 'deleted')


class TestModelMetadata(unittest.TestCase):

    def test_model_metadata_accepts_model_specific_fields(self):
        metadata = ModelMetadata(
            model_name='bert-base-uncased',
            model_id=7,
            local_path='/models/bert',
            huggingface_id='bert-base-uncased',
            language='en',
        )
        self.assertEqual(metadata.model_id, 7)
        self.assertEqual(metadata.local_path, '/models/bert')
        self.assertEqual(metadata.huggingface_id, 'bert-base-uncased')
        self.assertEqual(metadata.language, 'en')

    def test_model_metadata_has_deterministic_fallback_key(self):
        metadata = ModelMetadata(
            model_name='bert-base-uncased',
            model_id=7,
        )
        self.assertEqual(metadata.echoframe_key, metadata.echoframe_key)
        self.assertEqual(metadata.format_echoframe_key(),
            metadata.echoframe_key.hex())


class TestMetadataClassLookup(unittest.TestCase):

    def test_metadata_class_lookup(self):
        self.assertIs(
            metadata_class_for_output_type('attention'),
            AttentionMetadata,
        )

    def test_unknown_output_type_raises(self):
        with self.assertRaises(ValueError):
            metadata_class_for_output_type('unknown')


if __name__ == '__main__':
    unittest.main()
