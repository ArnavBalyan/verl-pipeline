import os
from ray.util import get_actor

def _gid(): return int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0])

class _MemPermit:
    def __init__(self, need_bytes: int):
        self.need = need_bytes
        self.sup  = get_actor(f"sup_{_gid()}")

    async def __aenter__(self):
        await self.sup.acquire.remote(self.need)

    async def __aexit__(self, *_):
        self.sup.release.remote(self.need)
