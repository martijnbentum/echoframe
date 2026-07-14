'''Shared pytest setup for echoframe tests.

Import the real, lightweight ``frame`` package before any test module is
collected. ``tests/test_segment_features.py`` installs a ``SimpleNamespace``
stub for ``frame`` via ``sys.modules.setdefault`` to avoid heavy optional
imports; that stub is not a real package, which breaks submodule imports such
as ``from frame.frames import make_frames_from_numpy_matrix``. Loading the real
package here makes the ``setdefault`` a no-op while leaving the test's
``mock.patch.object`` usage intact.
'''

import frame  # noqa: F401
import pytest

from echoframe import lmdb_helper


@pytest.fixture(autouse=True)
def close_cached_lmdb_envs():
    '''Close LMDB envs leaked by a test so the suite cannot exhaust fds.'''
    yield
    for cached in list(lmdb_helper._ENV_CACHE.values()):
        cached['env'].close()
    lmdb_helper._ENV_CACHE.clear()
