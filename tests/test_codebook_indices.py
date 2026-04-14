'''Tests for codebook matrix storage and codebook index reconstruction.'''

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from echoframe import (
    Codebook,
    EchoframeMetadata,
    Store,
    TokenCodebooks,
)
from echoframe.index import LmdbIndex
from echoframe.output_storage import Hdf5ShardStore


class FakeCursor:
    def __init__(self, store):
        self.store = store
        self.keys = []
        self.index = 0

    def set_range(self, prefix):
        self.keys = sorted(key for key in self.store if key >= prefix)
        self.index = 0
        return bool(self.keys)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.keys):
            raise StopIteration
        key = self.keys[self.index]
        self.index += 1
        return key, self.store[key]


class FakeTxn:
    def __init__(self, env, write):
        self.env = env
        self.write = write

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def put(self, key, value, db):
        self.env.dbs[db][key] = value

    def get(self, key, db):
        return self.env.dbs[db].get(key)

    def delete(self, key, db):
        self.env.dbs[db].pop(key, None)

    def cursor(self, db):
        return FakeCursor(self.env.dbs[db])


class FakeEnv:
    def __init__(self):
        self.dbs = {}

    def open_db(self, name):
        self.dbs.setdefault(name, {})
        return name

    def begin(self, write=False):
        return FakeTxn(self, write=write)


class FakeDataset:
    def __init__(self, data):
        self.data = data
        self.shape = self._shape(data)
        self.dtype = type(self._leaf(data)).__name__

    def __getitem__(self, item):
        if item == ():
            return self.data
        raise KeyError(item)

    def _shape(self, data):
        if isinstance(data, list) and data:
            return (len(data),) + self._shape(data[0])
        if isinstance(data, list):
            return (0,)
        return ()

    def _leaf(self, data):
        if isinstance(data, list) and data:
            return self._leaf(data[0])
        return data


class FakeGroup(dict):
    def create_dataset(self, name, data):
        dataset = FakeDataset(data)
        self[name] = dataset
        return dataset


class FakeH5File:
    def __init__(self, files, path, mode):
        self.files = files
        self.path = str(path)
        if 'r' in mode and not path.exists():
            raise FileNotFoundError(path)
        if 'r' not in mode:
            path.touch(exist_ok=True)
        self.groups = self.files.setdefault(self.path, {})

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def require_group(self, path):
        return self.groups.setdefault(path, FakeGroup())

    def __getitem__(self, path):
        group_path, name = path.rsplit('/', 1)
        return self.groups[group_path][name]

    def __contains__(self, path):
        group_path, name = path.rsplit('/', 1)
        return name in self.groups.get(group_path, {})


class FakeH5Module:
    def __init__(self):
        self.files = {}

    def File(self, path, mode):
        return FakeH5File(self.files, path, mode)


class CountingStore:
    def __init__(self, matrices):
        self.matrices = matrices
        self.load_calls = 0

    def load_with_echoframe_key(self, echoframe_key):
        self.load_calls += 1
        return self.matrices[echoframe_key]


class TestMetadataValidation(unittest.TestCase):
    def test_codebook_outputs_require_layer_zero(self):
        valid_indices = EchoframeMetadata(
            phraser_key='phrase-1',
            collar=0,
            model_name='wav2vec2',
            output_type='codebook_indices',
            layer=0,
        )
        valid_matrix = EchoframeMetadata(
            phraser_key='phrase-1',
            collar=0,
            model_name='wav2vec2',
            output_type='codebook_matrix',
            layer=0,
        )

        self.assertEqual(valid_indices.layer, 0)
        self.assertEqual(valid_matrix.layer, 0)
        with self.assertRaisesRegex(
            ValueError, 'codebook output types require layer to be exactly 0'):
            EchoframeMetadata(
                phraser_key='phrase-1',
                collar=0,
                model_name='wav2vec2',
                output_type='codebook_indices',
                layer=1,
            )


class TestStoreCodebookMatrixRoundTrip(unittest.TestCase):
    def _make_store(self):
        tmpdir = tempfile.TemporaryDirectory()
        index = LmdbIndex(Path(tmpdir.name) / 'index', env=FakeEnv(),
            shards_root=Path(tmpdir.name) / 'shards')
        storage = Hdf5ShardStore(Path(tmpdir.name) / 'shards',
            h5_module=FakeH5Module())
        store = Store(tmpdir.name, index=index, storage=storage)
        return tmpdir, store

    def test_put_and_load_codebook_matrix(self):
        tmpdir, store = self._make_store()
        with tmpdir:
            matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
            metadata = store.put(
                phraser_key='phrase-1',
                collar=0,
                model_name='wav2vec2',
                output_type='codebook_matrix',
                layer=0,
                data=matrix,
            )

            self.assertEqual(metadata.output_type, 'codebook_matrix')
            self.assertEqual(metadata.layer, 0)
            np.testing.assert_array_equal(
                store.load(
                    phraser_key='phrase-1',
                    collar=0,
                    model_name='wav2vec2',
                    output_type='codebook_matrix',
                    layer=0,
                ),
                matrix,
            )


class TestCodebookLoading(unittest.TestCase):
    def test_codebook_matrix_is_lazily_loaded_and_cached(self):
        store = CountingStore({
            'matrix-entry': np.array([[1.0, 2.0], [3.0, 4.0]]),
        })
        obj = Codebook(
            echoframe_key='indices-entry',
            data=np.array([[0, 1], [1, 0]]),
            model_architecture='wav2vec2',
            codebook_matrix_echoframe_key='matrix-entry',
        ).bind_store(store)

        first = obj.codebook_matrix
        second = obj.codebook_matrix

        np.testing.assert_array_equal(first, second)
        self.assertEqual(store.load_calls, 1)


class TestCodebookVectorReconstruction(unittest.TestCase):
    def test_wav2vec2_codevectors_are_reconstructed_from_two_indices(self):
        store = CountingStore({
            'matrix-entry': np.array([[1.0, 2.0], [3.0, 4.0]]),
        })
        obj = Codebook(
            echoframe_key='indices-entry',
            data=np.array([[0, 1], [1, 0]]),
            model_architecture='wav2vec2',
            codebook_matrix_echoframe_key='matrix-entry',
        ).bind_store(store)

        vectors = obj.codevectors

        np.testing.assert_array_equal(vectors, np.array([
            [1.0, 2.0, 3.0, 4.0],
            [3.0, 4.0, 1.0, 2.0],
        ]))
        self.assertEqual(store.load_calls, 1)

    def test_spidr_codevectors_preserve_frame_major_structure(self):
        store = CountingStore({
            'matrix-entry': np.array([
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]),
        })
        obj = Codebook(
            echoframe_key='indices-entry',
            data=np.array([
                [1, 0],
                [0, 1],
            ]),
            model_architecture='spidr',
            codebook_matrix_echoframe_key='matrix-entry',
        ).bind_store(store)

        vectors = obj.codevectors

        self.assertEqual(vectors.shape, (2, 2, 2))
        np.testing.assert_array_equal(vectors[0], np.array([
            [3.0, 4.0],
            [5.0, 6.0],
        ]))
        np.testing.assert_array_equal(vectors[1], np.array([
            [1.0, 2.0],
            [7.0, 8.0],
        ]))
        self.assertEqual(store.load_calls, 1)

    def test_spidr_single_frame_indices_preserve_head_axis(self):
        store = CountingStore({
            'matrix-entry': np.array([
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]),
        })
        obj = Codebook(
            echoframe_key='indices-entry',
            data=np.array([1, 0]),
            model_architecture='spidr',
            codebook_matrix_echoframe_key='matrix-entry',
        ).bind_store(store)

        vectors = obj.codevectors

        self.assertEqual(vectors.shape, (1, 2, 2))
        np.testing.assert_array_equal(vectors[0], np.array([
            [3.0, 4.0],
            [5.0, 6.0],
        ]))

    def test_to_numpy_returns_normalized_indices(self):
        obj = Codebook(echoframe_key='indices-entry',
            data=[(0, 1), (1, 0)], model_architecture='wav2vec2',
            codebook_matrix_echoframe_key='matrix-entry')

        result = obj.to_numpy()

        np.testing.assert_array_equal(result, np.array([
            [0, 1],
            [1, 0],
        ]))


class TestTokenCodebooks(unittest.TestCase):
    def test_token_collection_tracks_shared_architecture(self):
        first = Codebook(
            echoframe_key='indices-entry-1',
            data=np.array([[0, 1]]),
            model_architecture='wav2vec2',
            codebook_matrix_echoframe_key='matrix-entry',
        )
        second = Codebook(
            echoframe_key='indices-entry-2',
            data=np.array([[1, 0]]),
            model_architecture='wav2vec2',
            codebook_matrix_echoframe_key='matrix-entry',
        )

        result = TokenCodebooks(tokens=[first, second])

        self.assertEqual(result.token_count, 2)
        self.assertEqual(result.model_architecture, 'wav2vec2')

    def test_to_numpy_stacks_uniform_tokens(self):
        first = Codebook(echoframe_key='indices-entry-1',
            data=np.array([[0, 1]]), model_architecture='wav2vec2',
            codebook_matrix_echoframe_key='matrix-entry')
        second = Codebook(echoframe_key='indices-entry-2',
            data=np.array([[1, 0]]), model_architecture='wav2vec2',
            codebook_matrix_echoframe_key='matrix-entry')

        result = TokenCodebooks(tokens=[first, second]).to_numpy()

        np.testing.assert_array_equal(result, np.array([
            [[0, 1]],
            [[1, 0]],
        ]))

    def test_codevectors_stacks_uniform_tokens(self):
        store = CountingStore({
            'matrix-entry': np.array([[1.0], [2.0]]),
        })
        first = Codebook(echoframe_key='indices-entry-1',
            data=np.array([[0, 1]]), model_architecture='wav2vec2',
            codebook_matrix_echoframe_key='matrix-entry').bind_store(store)
        second = Codebook(echoframe_key='indices-entry-2',
            data=np.array([[1, 0]]), model_architecture='wav2vec2',
            codebook_matrix_echoframe_key='matrix-entry').bind_store(store)

        result = TokenCodebooks(tokens=[first, second]).codevectors

        np.testing.assert_array_equal(result, np.array([
            [[1.0, 2.0]],
            [[2.0, 1.0]],
        ]))


if __name__ == '__main__':
    unittest.main()
