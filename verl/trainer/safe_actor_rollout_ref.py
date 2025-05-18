from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.trainer.mem_permit import _MemPermit

BYTES = 1024**3

class SafeActorRolloutRefWorker(ActorRolloutRefWorker):
    DEFAULT = 24 * BYTES

    async def _need(self, batch): return self.DEFAULT   # make smarter later

    async def rollout(self, *a, **kw):
        async with _MemPermit(await self._need(a[0])):  return await super().rollout(*a, **kw)

    async def generate_sequences(self, *a, **kw):
        async with _MemPermit(await self._need(a[0])):  return await super().generate_sequences(*a, **kw)

    async def generate_sequences_async(self, *a, **kw):
        async with _MemPermit(await self._need(a[0] or kw.get("prompts"))):
            return await super().generate_sequences_async(*a, **kw)

    async def compute_log_prob(self, *a, **kw):
        async with _MemPermit(await self._need(a[0])):  return await super().compute_log_prob(*a, **kw)

    async def compute_ref_log_prob(self, *a, **kw):
        async with _MemPermit(await self._need(a[0])):  return await super().compute_ref_log_prob(*a, **kw)
