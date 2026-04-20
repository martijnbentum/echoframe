'''Shared test helpers and fakes for echoframe tests.'''

from __future__ import annotations

from pathlib import Path
import tempfile

from echoframe.index import LmdbIndex
from echoframe.metadata import filter_metadata, metadata_class_for_output_type
from echoframe.output_storage import Hdf5ShardStore
from echoframe.store import Store


class FakeCursor:
    def __init__(self, store: dict[bytes, bytes]) -> None:
        self.store = store
        self.keys: list[bytes] = []
        self.index = 0

    def set_range(self, prefix: bytes) -> bool:
        self.keys = sorted(key for key in self.store if key >= prefix)
        self.index = 0
        return bool(self.keys)

    def __iter__(self) -> 'FakeCursor':
        return self

    def __next__(self) -> tuple[bytes, bytes]:
        if self.index >= len(self.keys):
            raise StopIteration
        key = self.keys[self.index]
        self.index += 1
        return key, self.store[key]


class FakeTxn:
    def __init__(self, env: 'FakeEnv', write: bool) -> None:
        self.env = env
        self.write = write

    def __enter__(self) -> 'FakeTxn':
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def put(self, key: bytes, value: bytes, db: bytes) -> None:
        self.env.dbs[db][key] = value

    def get(self, key: bytes, db: bytes) -> bytes | None:
        return self.env.dbs[db].get(key)

    def delete(self, key: bytes, db: bytes) -> None:
        self.env.dbs[db].pop(key, None)

    def cursor(self, db: bytes) -> FakeCursor:
        return FakeCursor(self.env.dbs[db])


class FakeEnv:
    def __init__(self) -> None:
        self.dbs: dict[bytes, dict[bytes, bytes]] = {}

    def open_db(self, name: bytes) -> bytes:
        self.dbs.setdefault(name, {})
        return name

    def begin(self, write: bool=False) -> FakeTxn:
        return FakeTxn(self, write=write)


class FakeDataset:
    def __init__(self, data: object) -> None:
        self.data = data
        self.shape = self._shape(data)
        self.dtype = type(self._leaf(data)).__name__

    def __getitem__(self, item: object) -> object:
        if item == ():
            return self.data
        raise KeyError(item)

    def _shape(self, data: object) -> tuple[int, ...]:
        if isinstance(data, list) and data:
            return (len(data),) + self._shape(data[0])
        if isinstance(data, list):
            return (0,)
        return ()

    def _leaf(self, data: object) -> object:
        if isinstance(data, list) and data:
            return self._leaf(data[0])
        return data


class FakeGroup(dict[str, FakeDataset]):
    def create_dataset(self, name: str, data: object) -> FakeDataset:
        dataset = FakeDataset(data)
        self[name] = dataset
        return dataset


class FakeH5File:
    def __init__(self, files: dict[str, dict[str, FakeGroup]],
        path: Path, mode: str) -> None:
        self.files = files
        self.path = str(path)
        if 'r' in mode and not path.exists():
            raise FileNotFoundError(path)
        if 'r' not in mode:
            path.touch(exist_ok=True)
        self.groups = self.files.setdefault(self.path, {})

    def __enter__(self) -> 'FakeH5File':
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def require_group(self, path: str) -> FakeGroup:
        return self.groups.setdefault(path, FakeGroup())

    def __contains__(self, path: str) -> bool:
        group_path, name = path.rsplit('/', 1)
        return name in self.groups.get(group_path, {})

    def __getitem__(self, path: str) -> FakeDataset:
        group_path, name = path.rsplit('/', 1)
        return self.groups[group_path][name]

    def __delitem__(self, path: str) -> None:
        group_path, name = path.rsplit('/', 1)
        del self.groups[group_path][name]


class FakeH5Module:
    def __init__(self) -> None:
        self.files: dict[str, dict[str, FakeGroup]] = {}

    def File(self, path: Path, mode: str) -> FakeH5File:
        return FakeH5File(self.files, path, mode)


def make_fake_store(tmpdir: str) -> Store:
    index = LmdbIndex(Path(tmpdir) / 'index', env=FakeEnv(),
        shards_root=Path(tmpdir) / 'shards')
    storage = Hdf5ShardStore(Path(tmpdir) / 'shards',
        h5_module=FakeH5Module())
    return Store(tmpdir, index=index, storage=storage)


def make_real_store() -> tuple[tempfile.TemporaryDirectory[str], Store]:
    tmpdir = tempfile.TemporaryDirectory()
    store = Store(tmpdir.name, max_shard_size_bytes=1024 * 1024)
    return tmpdir, store


def payload_to_list(payload):
    return payload.tolist() if hasattr(payload, 'tolist') else payload


def pk(value) -> bytes:
    if isinstance(value, bytes):
        raw = value
    else:
        raw = str(value).encode('utf-8')
    if len(raw) > 22:
        raise ValueError('test phraser keys must be <= 22 bytes')
    return raw.ljust(22, b'\0')


def hex_key(metadata) -> str:
    return metadata.format_echoframe_key()


def ensure_model(store: Store, model_name: str) -> None:
    if store.load_model_metadata(model_name) is None:
        store.register_model(model_name)


def make_key(store: Store, *, phraser_key, collar, model_name, output_type,
    layer):
    ensure_model(store, model_name)
    phraser_key = pk(phraser_key)
    kwargs = {
        'output_type': output_type,
        'model_name': model_name,
    }
    if output_type in {'hidden_state', 'attention'}:
        kwargs.update({
            'phraser_key': phraser_key,
            'layer': layer,
            'collar': collar,
        })
    elif output_type == 'codebook_indices':
        kwargs.update({
            'phraser_key': phraser_key,
            'collar': collar,
        })
    elif output_type == 'codebook_matrix':
        pass
    else:
        message = 'unsupported output_type in test helper: '
        message += f'{output_type}'
        raise ValueError(message)
    return store.make_echoframe_key(**kwargs)


def put(store: Store, *, phraser_key, collar, model_name, output_type,
    layer, data, tags=None):
    phraser_key = pk(phraser_key)
    echoframe_key = make_key(store, phraser_key=phraser_key,
        collar=collar, model_name=model_name, output_type=output_type,
        layer=layer)
    metadata_cls = metadata_class_for_output_type(output_type)
    metadata = metadata_cls(phraser_key=phraser_key, collar=collar,
        model_name=model_name, layer=layer, tags=tags,
        echoframe_key=echoframe_key)
    return store.save(echoframe_key, metadata, data)


def put_item(store: Store, *, phraser_key, collar, model_name, output_type,
    layer, data, tags=None):
    phraser_key = pk(phraser_key)
    echoframe_key = make_key(store, phraser_key=phraser_key,
        collar=collar, model_name=model_name, output_type=output_type,
        layer=layer)
    metadata_cls = metadata_class_for_output_type(output_type)
    metadata = metadata_cls(phraser_key=phraser_key, collar=collar,
        model_name=model_name, layer=layer, tags=tags,
        echoframe_key=echoframe_key)
    return {'echoframe_key': echoframe_key, 'metadata': metadata, 'data': data}


def put_many(store: Store, items):
    prepared = []
    for item in items:
        if 'metadata' in item:
            prepared.append(item)
            continue
        prepared.append(put_item(store, **item))
    return store.save_many(prepared)


def find(store: Store, phraser_key, include_deleted=False, **filters):
    records = store.find_phraser(pk(phraser_key),
        include_deleted=include_deleted)
    return filter_metadata(records, **filters)


def find_one(store: Store, *, phraser_key, collar, model_name, output_type,
    layer, match='exact'):
    matches = find(store, phraser_key, model_name=model_name,
        output_type=output_type, layer=layer, collar=collar, match=match)
    if not matches:
        return None
    return matches[0]


def exists(store, phraser_key, collar, model_name, output_type, layer,
    match='exact'):
    return find_one(store, phraser_key=phraser_key, collar=collar,
        model_name=model_name, output_type=output_type, layer=layer,
        match=match) is not None


def load_query(store: Store, *, phraser_key, collar, model_name, output_type,
    layer, match='exact'):
    metadata = find_one(store, phraser_key=phraser_key, collar=collar,
        model_name=model_name, output_type=output_type, layer=layer,
        match=match)
    if metadata is None:
        raise ValueError('no stored output matched the requested criteria')
    return store.load(metadata.echoframe_key)


def find_many(store: Store, queries):
    return [find_one(store, **query) for query in queries]


def load_many_queries(store: Store, queries, strict=False):
    payloads = []
    for query in queries:
        metadata = find_one(store, **query)
        if metadata is None:
            if strict:
                raise ValueError(
                    'no stored output matched one of the requested queries')
            payloads.append(None)
        else:
            payloads.append(store.load(metadata.echoframe_key))
    return payloads


def delete(store: Store, *, phraser_key, collar, model_name, output_type,
    layer, match='exact'):
    store.delete_phraser_key(phraser_key=pk(phraser_key), collar=collar,
        model_name=model_name, output_type=output_type, layer=layer,
        match=match)
    records = store.find_phraser(pk(phraser_key), include_deleted=True)
    matches = filter_metadata(records, model_name=model_name,
        output_type=output_type, layer=layer, collar=collar, match=match)
    if not matches:
        return None
    return matches[0]


def load_object_frames(store: Store, *, phraser_key, model_name, layer,
    collar=500, output_type='hidden_state', match='exact'):
    if collar is None:
        try:
            payloads, metadata_list = store.phraser_key_to_outputs(
                phraser_key=pk(phraser_key), model_name=model_name,
                layer=layer, collar=collar, output_type=output_type,
                match=match)
        except ValueError:
            return {}
        return {metadata.collar: payload for metadata, payload in zip(
            metadata_list, payloads)}
    payload, metadata = store.phraser_key_to_output(
        phraser_key=pk(phraser_key), model_name=model_name, layer=layer,
        collar=collar, output_type=output_type, match=match)
    return payload


def iter_object_frames(store: Store, *, phraser_key, model_name, layer,
    collar=None, output_type='hidden_state', match='exact'):
    try:
        payloads, metadata_list = store.phraser_key_to_outputs(
            phraser_key=pk(phraser_key), model_name=model_name, layer=layer,
            collar=collar, output_type=output_type, match=match)
    except ValueError:
        if collar is None:
            return iter(())
        raise
    return zip(metadata_list, payloads)


def find_or_compute(store: Store, *, phraser_key, collar, model_name,
    output_type, layer, compute, match='exact', tags=None,
    add_tags_on_hit=False):
    ensure_model(store, model_name)
    return store.find_or_compute(phraser_key=pk(phraser_key), collar=collar,
        model_name=model_name, output_type=output_type, layer=layer,
        compute=compute, match=match, tags=tags,
        add_tags_on_hit=add_tags_on_hit)
