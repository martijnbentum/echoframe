'''Resolve phraser objects against registered echoframe phraser sources.'''

from .phraser_sources import normalise_phraser_source_root


def phraser_segment_to_phraser_source_id(store, segment):
    '''Return the registered phraser source id for a bound phraser segment.'''
    phraser_store = segment.store
    root = normalise_phraser_source_root(phraser_store.path)
    matches = []
    for source in store.list_phraser_sources():
        if normalise_phraser_source_root(source.root) == root:
            matches.append(source)
    if len(matches) == 1: return matches[0].source_id
    if not matches:
        raise ValueError('phraser segment store is not registered')
    raise ValueError('multiple phraser sources match segment store')


def phraser_segments_to_phraser_source_id(store, segments):
    '''Return the single registered phraser source id for bound segments.'''
    segments = list(segments)
    if not segments:
        raise ValueError('segments must not be empty')
    source_ids = []
    for segment in segments:
        source_id = phraser_segment_to_phraser_source_id(store, segment)
        if source_id not in source_ids: source_ids.append(source_id)
    if len(source_ids) == 1: return source_ids[0]
    raise ValueError('batch segments must come from one phraser source')


def phraser_source_id_to_phraser_source(store, phraser_source_id):
    '''Return the registered phraser source for source id.'''
    if phraser_source_id is None:
        raise ValueError('phraser_source_id is required')
    source = store.load_phraser_source(phraser_source_id)
    if source is None:
        raise ValueError(f'unknown phraser_source_id: {phraser_source_id!r}')
    return source
