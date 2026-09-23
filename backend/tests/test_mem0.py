"""Round trip against the real hosted Mem0: what add() stores, search() must return.

Needs MEM0_API_KEY in backend/.env and internet. Mem0 processes adds asynchronously,
so we poll search() for a while before failing.
"""
import asyncio
import uuid

import pytest

import config
import memory_service

pytestmark = pytest.mark.skipif(not config.MEM0_API_KEY, reason="MEM0_API_KEY not set")


@pytest.mark.asyncio
async def test_search_returns_what_add_stored():
    memory_service._client = None  # fresh client bound to this test's event loop
    session = f"test-{uuid.uuid4().hex[:8]}"
    claim = "Vidya Setu scheme ka budget 5,000 crore hai"
    await memory_service.add_correction(session, claim, "source says 4,500 crore, not 5,000 crore")

    found = []
    for _ in range(20):
        found = await memory_service.search(session, "Vidya Setu scheme budget")
        if found:
            break
        await asyncio.sleep(3)
    assert any("4,500 crore" in m and "Vidya Setu" in m for m in found), found

    # another session must not see it
    assert await memory_service.search(f"other-{session}", "Vidya Setu scheme budget") == []
    await memory_service.client().delete_all(user_id=session)
