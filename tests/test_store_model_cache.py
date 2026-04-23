'''Focused tests for Store model cache behavior.'''

import tempfile
import unittest
from unittest import mock

from echoframe.store import Store


class TestStoreLoadModel(unittest.TestCase):

    def test_load_model_raises_for_unknown_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp)

            with self.assertRaisesRegex(ValueError,
                'model_name is not registered'):
                store.load_model('unknown-model')

    def test_load_model_reuses_cached_model_for_same_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp)
            store.register_model('wav2vec2',
                huggingface_id='facebook/wav2vec2-base')

            with mock.patch('echoframe.store.model_loader.load_model',
                return_value='model') as load_model, mock.patch(
                'echoframe.store.model_loader.model_is_on_gpu',
                return_value=False) as model_is_on_gpu, mock.patch(
                'echoframe.store.model_loader.move_model_to_gpu') as move_gpu:
                result1 = store.load_model('wav2vec2')
                result2 = store.load_model('wav2vec2')

        self.assertEqual(result1, 'model')
        self.assertEqual(result2, 'model')
        load_model.assert_called_once()
        model_is_on_gpu.assert_called_once_with('model')
        move_gpu.assert_not_called()

    def test_load_model_moves_cached_model_to_gpu_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp)
            store.register_model('wav2vec2',
                huggingface_id='facebook/wav2vec2-base')

            with mock.patch('echoframe.store.model_loader.load_model',
                return_value='cpu-model') as load_model, mock.patch(
                'echoframe.store.model_loader.model_is_on_gpu',
                return_value=False) as model_is_on_gpu, mock.patch(
                'echoframe.store.model_loader.move_model_to_gpu',
                return_value='gpu-model') as move_gpu:
                store.load_model('wav2vec2')
                result = store.load_model('wav2vec2', gpu=True)

        self.assertEqual(result, 'gpu-model')
        self.assertEqual(store._model, 'gpu-model')
        load_model.assert_called_once()
        move_gpu.assert_called_once_with('cpu-model')
        self.assertEqual(model_is_on_gpu.call_count, 1)

    def test_load_model_replaces_cached_model_for_new_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp)
            store.register_model('wav2vec2',
                huggingface_id='facebook/wav2vec2-base')
            store.register_model('hubert',
                huggingface_id='facebook/hubert-base-ls960')

            with mock.patch('echoframe.store.model_loader.load_model',
                side_effect=['m1', 'm2']) as load_model, mock.patch(
                'echoframe.store.model_loader.model_is_on_gpu',
                return_value=True) as model_is_on_gpu, mock.patch(
                'echoframe.store.model_loader.move_model_to_cpu',
                return_value='m1-cpu') as move_cpu:
                first = store.load_model('wav2vec2', gpu=True)
                second = store.load_model('hubert')

        self.assertEqual(first, 'm1')
        self.assertEqual(second, 'm2')
        self.assertEqual(store._model, 'm2')
        self.assertEqual(store._model_name, 'hubert')
        self.assertEqual(load_model.call_count, 2)
        move_cpu.assert_called_once_with('m1')
        model_is_on_gpu.assert_called_once_with('m1')

    def test_remove_cached_model_clears_model_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp)
            store._model = 'model'
            store._model_name = 'wav2vec2'

            with mock.patch('echoframe.store.model_loader.model_is_on_gpu',
                return_value=False) as model_is_on_gpu, mock.patch(
                'echoframe.store.model_loader.move_model_to_cpu') as move_cpu:
                store.remove_cached_model()

        self.assertIsNone(store._model)
        self.assertIsNone(store._model_name)
        model_is_on_gpu.assert_called_once_with('model')
        move_cpu.assert_not_called()


if __name__ == '__main__':
    unittest.main()
