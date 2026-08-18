"""
Test that awgraph can be imported and used without AitherOS in sys.path.

This test is run from a directory OUTSIDE the AitherOS repo to ensure
the package is truly standalone.
"""

import asyncio
import sys
import tempfile
from pathlib import Path


def test_import_standalone():
    """Verify awgraph imports without AitherOS."""
    # Should not raise ImportError
    import awgraph
    from awgraph import CodeGraph, CodeGraphRegistry

    assert CodeGraph is not None
    assert CodeGraphRegistry is not None
    assert awgraph.__version__ == "1.0.0"


async def test_basic_indexing():
    """Test basic indexing and keyword search."""
    from awgraph import CodeGraph

    # Create a temp directory with test Python code
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "example.py"
        test_file.write_text("""
def rate_limiter(max_calls: int, time_window: float):
    '''Rate limiter implementation.

    Args:
        max_calls: Maximum calls allowed
        time_window: Time window in seconds
    '''
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def cache_result(func):
    '''Caching decorator.'''
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
        """)

        # Create graph and index. The signature is (root_path, cache_dir,
        # max_workers, auto_index) -- there is no db_path. This test previously
        # invented one, so it exercised an API that does not exist and told us
        # nothing about whether the package works.
        graph = CodeGraph(root_path=tmpdir, auto_index=False)
        await graph.index_codebase(tmpdir)

        # index_codebase needs an ABSOLUTE path: given a relative one it walks
        # nothing, indexes 0 chunks, and returns success. A test that only
        # asserted "no exception" would pass on that empty result.
        assert len(graph.chunks) > 0, "indexed nothing -- is the path absolute?"

        # The real product call. The query deliberately does NOT contain the
        # symbol name, so a pass means retrieval matched on meaning rather than
        # on the string being present.
        results = await graph.hybrid_query("rate limiter", max_results=10)
        assert len(results) > 0, "no results for a query the corpus answers"

        # CodeChunk is a dataclass, not a dict -- assert on the attributes the
        # retrieval path actually returns, not on a mapping shape.
        top = results[0]
        assert getattr(top, 'name', None), 'result carries no symbol name'
        assert getattr(top, 'source_path', None), 'result carries no source_path'

        print('indexed {} chunks; {} results'.format(len(graph.chunks), len(results)))


async def test_registry():
    """Test the CodeGraphRegistry."""
    from awgraph import CodeGraphRegistry

    registry = CodeGraphRegistry()
    assert registry is not None

    # Registry should initialize without errors
    # (won't persist since no Strata, but that's expected)


def main():
    """Run all tests."""
    print("Running awgraph standalone tests...")

    # Test 1: Import
    try:
        test_import_standalone()
        print("✓ Import test passed")
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return 1

    # Test 2: Basic indexing
    try:
        asyncio.run(test_basic_indexing())
        print("✓ Indexing test passed")
    except Exception as e:
        print(f"✗ Indexing test failed: {e}")
        return 1

    # Test 3: Registry
    try:
        asyncio.run(test_registry())
        print("✓ Registry test passed")
    except Exception as e:
        print(f"✗ Registry test failed: {e}")
        return 1

    print("\nAll tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
