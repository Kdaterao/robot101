"""Lightweight sampled hardware usage for an annotation run."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
from time import monotonic

import psutil

from robot101.data.utilities.annotation_tree import atomic_json


class HardwareUsage:
    """CPU/RAM include the process tree; NVIDIA metrics are device-wide."""

    def __init__(self, interval=1.0):
        self.interval = interval
        self.started = monotonic()
        self.stop_event = threading.Event()
        self.processes = {}
        self.cpu_peak = 0.0
        self.ram_peak = 0.0
        self.host_cpu_peak = 0.0
        self.samples = 0
        self.gpus = {}
        self.notes = set()
        self.finished = False
        self.smi = shutil.which("nvidia-smi")
        if not self.smi:
            self.notes.add("nvidia-smi unavailable; GPU utilization and device VRAM unavailable")
        self.root = psutil.Process(os.getpid())
        self._sample()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _sample(self):
        cpu, ram = 0.0, 0.0
        try:
            processes = [self.root] + self.root.children(recursive=True)
        except psutil.Error:
            processes = [self.root]
        live = set()
        for process in processes:
            try:
                key = (process.pid, process.create_time())
                live.add(key)
                if key not in self.processes:
                    self.processes[key] = process
                    process.cpu_percent(None)  # Prime interval counters.
                else:
                    cpu += self.processes[key].cpu_percent(None)
                ram += process.memory_info().rss / 2**20
            except psutil.Error:
                continue
        self.processes = {k: p for k, p in self.processes.items() if k in live}
        self.cpu_peak = max(self.cpu_peak, cpu)
        self.ram_peak = max(self.ram_peak, ram)
        self.host_cpu_peak = max(self.host_cpu_peak, psutil.cpu_percent(None))
        self.samples += 1
        if not self.smi:
            return
        try:
            result = subprocess.run(
                [self.smi, "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True,
                timeout=2, check=True)
            for fields in csv.reader(result.stdout.splitlines()):
                index, name, utilization, used, total = [f.strip() for f in fields]
                gpu = self.gpus.setdefault(index, dict(index=int(index), name=name,
                    peak_utilization_percent=None, peak_vram_mib=None, total_vram_mib=None))
                for field, value in (("peak_utilization_percent", utilization), ("peak_vram_mib", used)):
                    try:
                        number = float(value)
                        gpu[field] = max(gpu[field] or 0.0, number)
                    except ValueError:
                        self.notes.add(f"GPU {index}: {field} unavailable")
                try:
                    gpu["total_vram_mib"] = float(total)
                except ValueError:
                    pass
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            self.notes.add(f"GPU sampling unavailable: {type(exc).__name__}")

    def _loop(self):
        while not self.stop_event.wait(self.interval):
            self._sample()

    def finish(self, output_dir):
        if self.finished:
            return
        self.stop_event.set()
        self.thread.join()
        self._sample()
        self.finished = True
        record = dict(sample_interval_seconds=self.interval, samples=self.samples,
            wall_seconds=monotonic() - self.started, logical_cpu_count=psutil.cpu_count(),
            peak_process_tree_cpu_percent=self.cpu_peak,
            peak_host_cpu_percent=self.host_cpu_peak, peak_process_tree_rss_mib=self.ram_peak,
            gpus=list(self.gpus.values()), notes=sorted(self.notes),
            scope="CPU/RSS: process tree; GPU utilization/VRAM: whole device, including other jobs",
            limitations="Sampled peaks may miss short spikes; summed RSS can count shared pages more than once")
        output_dir = Path(output_dir)
        atomic_json(output_dir / "hardware_usage.json", record)
        summary_path = output_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            summary["hardware_usage"] = record
            atomic_json(summary_path, summary)
        print(f"Hardware peaks: process-tree CPU {self.cpu_peak:.1f}% "
              f"({self.cpu_peak / 100:.2f} cores), host CPU {self.host_cpu_peak:.1f}%, "
              f"process-tree RAM {self.ram_peak / 1024:.2f} GiB", flush=True)
        for gpu in self.gpus.values():
            utilization = gpu["peak_utilization_percent"]
            vram = gpu["peak_vram_mib"]
            busy = "N/A" if utilization is None else f"{utilization:.0f}%"
            memory = "N/A" if vram is None else f"{vram / 1024:.2f} GiB"
            print(f'  GPU {gpu["index"]} ({gpu["name"]}, whole device): '
                  f"peak utilization {busy}, peak VRAM {memory}", flush=True)
        print(f"Hardware report: {output_dir / 'hardware_usage.json'}", flush=True)
