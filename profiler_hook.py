# profiler_hook.py
import os
import torch
from mmengine.hooks import Hook
from mmengine.runner import Runner

class TorchProfilerHook(Hook):
    """在指定迭代范围内启动 torch.profiler 的 Hook."""

    def __init__(self,
                 trace_dir='profiler_traces',
                 wait=1, warmup=1, active=3, repeat=2,
                 activities=('cpu', 'cuda'),
                 record_shapes=True,
                 profile_memory=True,
                 on_rank0_only=True):
        self.trace_dir = trace_dir
        self.wait, self.warmup, self.active, self.repeat = wait, warmup, active, repeat
        self.activities = [getattr(torch.profiler.ProfilerActivity, a.upper())
                           for a in activities]
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.on_rank0_only = on_rank0_only
        self.prof = None

    # ---------- 生命周期 ----------
    def before_test(self, runner: Runner):
        if self.on_rank0_only and runner.rank != 0:
            return          # 只在主进程启用
        os.makedirs(os.path.join(runner.work_dir, self.trace_dir), exist_ok=True)

        schedule = torch.profiler.schedule(
            wait=self.wait, warmup=self.warmup,
            active=self.active, repeat=self.repeat)

        self.prof = torch.profiler.profile(
            activities=self.activities,
            schedule=schedule,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            on_trace_ready=lambda p: p.export_chrome_trace(
                os.path.join(runner.work_dir,
                             self.trace_dir,
                             f'iter_{runner.iter}.json'))
        )
        self.prof.__enter__()

    def after_test_iter(self, runner: Runner,
                         batch_idx: int,
                         data_batch=None,
                         outputs=None):
        if self.prof:
            self.prof.step()     # 触发调度器

    def after_test(self, runner: Runner):
        if self.prof:
            self.prof.__exit__(None, None, None)
            self.prof = None
