'''Focused tests for model_loader helpers.'''

import unittest
from unittest import mock

from echoframe import model_loader
from echoframe.model_registry import ModelMetadata


class TestLoadModel(unittest.TestCase):

    def test_load_model_prefers_local_path(self):
        metadata = ModelMetadata('wav2vec2', local_path='/tmp/model.pt',
            huggingface_id='facebook/wav2vec2-base')

        with mock.patch('to_vector.load.load_model',
            return_value='model') as load_model:
            result = model_loader.load_model(metadata, gpu=True)

        self.assertEqual(result, 'model')
        load_model.assert_called_once_with('/tmp/model.pt', gpu=True)

    def test_load_model_uses_huggingface_id(self):
        metadata = ModelMetadata('wav2vec2',
            huggingface_id='facebook/wav2vec2-base')

        with mock.patch('to_vector.load.load_model',
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

        with mock.patch('to_vector.load.model_is_on_gpu',
            return_value=False) as model_is_on_gpu:
            result = model_loader.model_is_on_gpu(model)

        self.assertFalse(result)
        model_is_on_gpu.assert_called_once_with(model)

    def test_remove_model_from_gpu_moves_to_cpu_and_empties_cache(self):
        model = object()

        with mock.patch('to_vector.load.move_model_to_cpu',
            return_value='cpu-model') as move_to_cpu, mock.patch(
            'torch.cuda.empty_cache') as empty_cache:
            result = model_loader.remove_model_from_gpu(model)

        self.assertEqual(result, 'cpu-model')
        move_to_cpu.assert_called_once_with(model)
        empty_cache.assert_called_once_with()


class TestLoadCodebookModel(unittest.TestCase):

    def test_wav2vec2_uses_pretraining_loader(self):
        metadata = ModelMetadata('wav2vec2',
            huggingface_id='facebook/wav2vec2-base')
        with mock.patch('to_vector.model_registry.filename_model_type',
            return_value=None), mock.patch('to_vector.load.load_model_pt',
            return_value='pt') as load_pt, mock.patch(
            'to_vector.load.load_model') as load_model:
            result = model_loader.load_codebook_model(metadata)
        self.assertEqual(result, 'pt')
        load_pt.assert_called_once_with('facebook/wav2vec2-base', gpu=False)
        load_model.assert_not_called()

    def test_spidr_from_local_config_uses_normal_loader(self):
        metadata = ModelMetadata('spidr', local_path='/models/spidr')
        with mock.patch('to_vector.model_registry.filename_model_type',
            return_value='spidr'), mock.patch('to_vector.load.load_model',
            return_value='spidr-model') as load_model, mock.patch(
            'to_vector.load.load_model_pt') as load_pt:
            result = model_loader.load_codebook_model(metadata, gpu=True)
        self.assertEqual(result, 'spidr-model')
        load_model.assert_called_once_with('/models/spidr', gpu=True)
        load_pt.assert_not_called()

    def test_registered_architecture_wins_over_config(self):
        metadata = ModelMetadata('spidr', local_path='/models/spidr',
            architecture='SpidR')
        with mock.patch('to_vector.model_registry.filename_model_type'
            ) as model_type, mock.patch('to_vector.load.load_model',
            return_value='spidr-model'):
            model_loader.load_codebook_model(metadata)
        model_type.assert_not_called()


if __name__ == '__main__':
    unittest.main()
