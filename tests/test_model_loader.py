'''Focused tests for model_loader helpers.'''

import unittest
from unittest import mock

from echoframe import model_loader
from echoframe.model_registry import ModelMetadata


class TestLoadModel(unittest.TestCase):

    def test_load_model_prefers_local_path(self):
        metadata = ModelMetadata('wav2vec2', local_path='/tmp/model.pt',
            huggingface_id='facebook/wav2vec2-base')

        with mock.patch.object(model_loader.to_vector_load, 'load_model',
            return_value='model') as load_model:
            result = model_loader.load_model(metadata, gpu=True)

        self.assertEqual(result, 'model')
        load_model.assert_called_once_with('/tmp/model.pt', gpu=True)

    def test_load_model_uses_huggingface_id(self):
        metadata = ModelMetadata('wav2vec2',
            huggingface_id='facebook/wav2vec2-base')

        with mock.patch.object(model_loader.to_vector_load, 'load_model',
            return_value='model') as load_model:
            result = model_loader.load_model(metadata)

        self.assertEqual(result, 'model')
        load_model.assert_called_once_with('facebook/wav2vec2-base',
            gpu=False)

    def test_load_model_raises_without_loadable_source(self):
        metadata = ModelMetadata('wav2vec2')

        with self.assertRaisesRegex(ValueError, 'no loadable source'):
            model_loader.load_model(metadata)


class TestGpuHelpers(unittest.TestCase):

    def test_model_is_on_gpu_delegates(self):
        model = object()

        with mock.patch.object(model_loader.to_vector_load, 'model_is_on_gpu',
            return_value=False) as model_is_on_gpu:
            result = model_loader.model_is_on_gpu(model)

        self.assertFalse(result)
        model_is_on_gpu.assert_called_once_with(model)

    def test_remove_model_from_gpu_moves_to_cpu(self):
        model = object()

        with mock.patch.object(model_loader.to_vector_load,
            'move_model_to_cpu', return_value='cpu-model') as move_to_cpu:
            result = model_loader.remove_model_from_gpu(model)

        self.assertEqual(result, 'cpu-model')
        move_to_cpu.assert_called_once_with(model)


if __name__ == '__main__':
    unittest.main()
