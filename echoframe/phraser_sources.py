'''Phraser source records for linked phraser databases.'''


class PhraserSource:
    '''Persisted reference to a phraser database source.'''

    def __init__(self, source_id, root, phraser_store=None):
        '''Create one phraser source reference.
        source_id:       stable source identifier stored on metadata
        root:            phraser store root/path
        phraser_store:   optional already-open phraser Store instance
        '''
        self.source_id = source_id
        self.root = str(root)
        self._phraser_store = phraser_store
        self._validate()

    def __repr__(self):
        return f'PhraserSource(source_id={self.source_id!r}, root={self.root!r})'

    def open(self):
        '''Return an open phraser Store for this source.'''
        if self._phraser_store is None:
            import phraser
            self._phraser_store = phraser.open_store(path=self.root)
        return self._phraser_store

    def to_dict(self):
        '''Serialize this source to config.json data.'''
        return {'root': self.root}

    @classmethod
    def from_dict(cls, source_id, data):
        '''Build one source record from serialized config data.'''
        if not isinstance(data, dict):
            raise ValueError('phraser source records must be JSON objects')
        return cls(source_id=source_id, root=data.get('root'))

    def _validate(self):
        _validate_source_id(self.source_id)
        _validate_root(self.root)


def _validate_source_id(source_id):
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError('phraser_source_id must be a non-empty string')


def _validate_root(root):
    if not isinstance(root, str) or not root.strip():
        raise ValueError('phraser source root must be a non-empty string')
