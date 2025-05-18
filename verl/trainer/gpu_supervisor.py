import asyncio, ray, pynvml

@ray.remote
class GpuSupervisor:
    POLL = 0.25

    def __init__(self, gpu_index: int):
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self.free_bytes = int(pynvml.nvmlDeviceGetMemoryInfo(self.h).free)
        self.wait_q = []           # [(need, event)]
        asyncio.get_event_loop().create_task(self._refresh())

    async def _refresh(self):
        while True:
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.h)
            util = pynvml.nvmlDeviceGetUtilizationRates(self.h)
            self.free_bytes = int(mem.free)
            self.util = util.gpu
            await asyncio.sleep(self.POLL)

    async def acquire(self, need: int):
        if self.free_bytes >= need and not self.wait_q:
            self.free_bytes -= need;  return
        ev = asyncio.Event();  self.wait_q.append((need, ev))
        while True:
            await ev.wait()
            if self.free_bytes >= need and self.wait_q[0][0] == need:
                self.wait_q.pop(0);  self.free_bytes -= need;  return
            ev.clear()

    def release(self, give_back: int):
        self.free_bytes += give_back
        for _, ev in self.wait_q: ev.set()

    def metrics(self):            # optional
        return {"free_bytes": self.free_bytes, "util_pct": getattr(self, "util", 0)}
