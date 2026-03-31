'''Disk-backed storage for model outputs linked to phraser keys.'''

from .metadata import OUTPUT_TYPES, Metadata
from .output_storage import Hdf5ShardStore
from .store import Store

__all__ = ['Store', 'OUTPUT_TYPES', 'Metadata', '__version__']

__version__ = '0.1.0'
