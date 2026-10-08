#!/usr/bin/env python3
"""Measure a command or an existing PID tree on the GPU VM."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
import signal
from pathlib import Path
import shutil
import subprocess
import threading
import time

import psutil


def number(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def parse_pmon(line, header):
    """Keep unsupported process metrics as None, rather than substituting GPU totals."""
    line = line.strip()
    parts = line.split()
    if line.startswith('#'):
        fields = line.lstrip('#').split()
        return (None, fields) if 'pid' in fields and 'gpu' in fields else (None, header)
    if not header or len(parts) < len(header):
        return None, header
    row = dict(zip(header, parts))
    if not row['pid'].isdigit() or not row['gpu'].isdigit():
        return None, header
    return dict(gpu_index=int(row['gpu']), pid=int(row['pid']),
                sm_percent=number(row.get('sm')), memory_activity_percent=number(row.get('mem')),
                encoder_percent=number(row.get('enc')), decoder_percent=number(row.get('dec')),
                vram_mib=number(row.get('fb'))), header


class GPUMonitor:
    def __init__(self):
        self.process = None
        self.latest = {}
        self.lock = threading.Lock()
        self.errors = []
        if not shutil.which('nvidia-smi'):
            self.errors.append('nvidia-smi unavailable; GPU measurements unavailable')
            return
        self.process = subprocess.Popen(
            ['nvidia-smi', 'pmon', '-s', 'um', '-d', '1'],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.thread = threading.Thread(target=self.read, daemon=True)
        self.thread.start()

    def read(self):
        header = None
        for line in self.process.stdout:
            row, header = parse_pmon(line, header)
            if row:
                with self.lock:
                    self.latest[(row['gpu_index'], row['pid'])] = (time.monotonic(), row)
            elif line.strip() and not line.startswith('#') and header is None:
                self.errors.append(line.strip())

    def snapshot(self):
        now = time.monotonic()
        with self.lock:
            process_rows = [dict(row) for stamp, row in self.latest.values() if now-stamp < 2.5]
        if not shutil.which('nvidia-smi'):
            return [], process_rows
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total',
                '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5, check=True)
            devices = []
            for fields in csv.reader(result.stdout.splitlines()):
                if len(fields) != 6:
                    continue
                idx, uuid, busy, mem_busy, used, total = [f.strip() for f in fields]
                devices.append(dict(index=int(idx), uuid=uuid, utilization_percent=number(busy),
                                    memory_activity_percent=number(mem_busy), used_mib=number(used), total_mib=number(total)))
            # VRAM by compute PID is available on more configurations than pmon SM.
            result = subprocess.run([
                'nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
                '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5, check=True)
            uuid_indices = {d['uuid']: d['index'] for d in devices}
            by_pid = {(r['gpu_index'], r['pid']): r for r in process_rows}
            for fields in csv.reader(result.stdout.splitlines()):
                if len(fields) != 3:
                    continue
                pid, uuid, memory = [f.strip() for f in fields]
                if not pid.isdigit() or uuid not in uuid_indices:
                    continue
                key = (uuid_indices[uuid], int(pid))
                row = by_pid.setdefault(key, dict(gpu_index=key[0], pid=key[1], sm_percent=None,
                                                   memory_activity_percent=None, encoder_percent=None, decoder_percent=None))
                row['vram_mib'] = number(memory)
            return devices, list(by_pid.values())
        except (subprocess.SubprocessError, OSError, ValueError) as exc:
            message = f'GPU query failed: {exc}'
            if message not in self.errors:
                self.errors.append(message)
            return [], process_rows

    def close(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.thread.join(timeout=1)
            self.process.stdout.close()


def add_metric(stats, value):
    if value is not None:
        stats['samples'] += 1
        stats['sum'] += value
        stats['peak'] = value if stats['peak'] is None else max(stats['peak'], value)


def new_metric():
    return dict(samples=0, sum=0., peak=None)


def finish_metrics(value):
    if isinstance(value, dict):
        if set(value) == {'samples', 'sum', 'peak'}:
            return dict(samples=value['samples'], mean=value['sum']/value['samples'] if value['samples'] else None,
                        peak=value['peak'])
        return {key: finish_metrics(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finish_metrics(item) for item in value]
    return value


def summarize(samples):
    # Keep only running totals, so long VM runs do not accumulate samples in RAM.
    processes, devices = {}, {}
    for sample in samples:
        for row in sample['processes']:
            key = (row['pid'], row['created_at'])
            stats = processes.setdefault(key, dict(pid=row['pid'], name=row['name'],
                        cpu_percent_one_core=new_metric(), rss_mib=new_metric(), gpu={}))
            stats['name'] = row['name']
            add_metric(stats['cpu_percent_one_core'], row['cpu_percent_one_core'])
            add_metric(stats['rss_mib'], row['rss_mib'])
            for gpu in row['gpu']:
                gstats = stats['gpu'].setdefault(gpu['gpu_index'], dict(index=gpu['gpu_index'],
                                            sm_percent=new_metric(), vram_mib=new_metric()))
                add_metric(gstats['sm_percent'], gpu.get('sm_percent'))
                add_metric(gstats['vram_mib'], gpu.get('vram_mib'))
        for row in sample['gpus']:
            stats = devices.setdefault(row['index'], dict(index=row['index'],
                            utilization_percent=new_metric(), used_mib=new_metric(), total_mib=row['total_mib']))
            add_metric(stats['utilization_percent'], row['utilization_percent'])
            add_metric(stats['used_mib'], row['used_mib'])
    for stats in processes.values():
        stats['gpu'] = list(stats['gpu'].values())
    return finish_metrics(dict(processes=list(processes.values()), gpus=list(devices.values())))


def print_summary(report):
    def fmt(value):
        return f'{value:.1f}' if value is not None else 'N/A'
    print('PID       Process          CPU mean%  RAM peak MiB  GPU  SM mean%  VRAM peak MiB')
    for process in report['processes']:
        for gpu in process['gpu'] or [None]:
            print(f"{process['pid']:<9} {process['name'][:16]:<16} "
                  f"{fmt(process['cpu_percent_one_core']['mean']):>9} "
                  f"{fmt(process['rss_mib']['peak']):>13} "
                  f"{str(gpu['index']) if gpu else '-':>4} "
                  f"{fmt(gpu['sm_percent']['mean']) if gpu else 'N/A':>9} "
                  f"{fmt(gpu['vram_mib']['peak']) if gpu else 'N/A':>14}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int, help='Attach to this PID and its children')
    parser.add_argument('--interval', type=float, default=2.0)
    parser.add_argument('--duration', type=float, help='Stop measuring after this many seconds; attached jobs keep running')
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('command', nargs=argparse.REMAINDER, help='Command following --')
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if bool(args.pid) == bool(command):
        parser.error('Choose either --pid PID or -- COMMAND')
    if args.interval < 1 or (args.duration is not None and args.duration <= 0):
        parser.error('interval must be >= 1 second and duration must be positive')
    if command and args.duration is not None:
        parser.error('--duration is for --pid; a launched command is monitored until completion')
    out = args.out_dir or Path('outputs/profiles') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    out.mkdir(parents=True, exist_ok=False)
    child = subprocess.Popen(command, start_new_session=os.name == "posix") if command else None
    pid = child.pid if child else args.pid
    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        raise SystemExit(f'PID {pid} already exited; use -- to profile a command from startup')
    monitor = GPUMonitor()
    tracked = {}
    start = time.monotonic()
    print(f'Profiling PID {pid}; measurements: {out}', flush=True)
    print('CPU 100% = one logical core. GPU SM is per PID; whole-GPU utilization is separate.', flush=True)
    stop_reason = 'processes_finished'
    cpu_notes = []
    try:
        with (out/'samples.jsonl').open('w') as logfile, (out/'processes.csv').open('w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['elapsed_s', 'pid', 'name', 'cpu_percent_one_core',
                                    'rss_mib', 'gpu_index', 'gpu_sm_percent', 'vram_mib'])
            writer.writeheader()
            while True:
                if child:
                    child.poll()  # Reap exited children so a zombie root cannot keep the monitor alive.
                try:
                    discovered = [root, *root.children(recursive=True)]
                except psutil.NoSuchProcess:
                    discovered = []
                except (psutil.AccessDenied, PermissionError):
                    discovered = [root]
                    if not cpu_notes:
                        cpu_notes.append('Child-process enumeration was blocked; only known PIDs were sampled.')
                fresh = set()
                for process in discovered:
                    try:
                        identity = (process.pid, process.create_time())
                        if identity not in tracked:
                            process.cpu_percent(None)
                            tracked[identity] = process
                            fresh.add(identity)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                devices, gpu_rows = monitor.snapshot()
                rows = []
                for identity, process in list(tracked.items()):
                    try:
                        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
                            del tracked[identity]
                            continue
                        row = dict(pid=process.pid, created_at=identity[1], name=process.name(),
                                   cpu_percent_one_core=None if identity in fresh else process.cpu_percent(None),
                                   rss_mib=process.memory_info().rss / 1024**2,
                                   gpu=[g for g in gpu_rows if g['pid'] == process.pid])
                        rows.append(row)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        tracked.pop(identity, None)
                if not rows:
                    break
                sample = dict(elapsed_s=round(time.monotonic()-start, 3),
                              host_cpu_percent=psutil.cpu_percent(None),
                              host_ram_percent=psutil.virtual_memory().percent, processes=rows, gpus=devices)
                logfile.write(json.dumps(sample)+'\n')
                logfile.flush()
                for row in rows:
                    for gpu in row['gpu'] or [{}]:
                        writer.writerow(dict(elapsed_s=sample['elapsed_s'], pid=row['pid'], name=row['name'],
                                             cpu_percent_one_core=row['cpu_percent_one_core'], rss_mib=row['rss_mib'],
                                             gpu_index=gpu.get('gpu_index'), gpu_sm_percent=gpu.get('sm_percent'),
                                             vram_mib=gpu.get('vram_mib')))
                csvfile.flush()
                if args.duration is not None and time.monotonic()-start >= args.duration:
                    stop_reason = 'duration_reached'
                    break
                time.sleep(args.interval)
    except KeyboardInterrupt:
        stop_reason = 'interrupted'
        if child and child.poll() is None:
            if os.name == 'posix':
                os.killpg(child.pid, signal.SIGTERM)
            else:
                child.terminate()
    finally:
        monitor.close()
        with (out/'samples.jsonl').open() as logfile:
            report = summarize(json.loads(line) for line in logfile if line.strip())
        report.update(elapsed_s=round(time.monotonic()-start, 3), stop_reason=stop_reason,
                      logical_cpu_count=psutil.cpu_count(), gpu_monitor_notes=list(dict.fromkeys(monitor.errors)),
                      cpu_monitor_notes=cpu_notes,
                      notes=['CPU percent uses one logical core as 100%.',
                             'RSS includes shared pages; summed process RSS can double-count memory.',
                             'GPU process SM null means unavailable, not 0%. Whole-GPU values include other jobs.',
                             'Peaks are sampled; brief spikes and short-lived subprocesses can be missed.'])
        (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
        print_summary(report)
        print(f'Profile saved: {out}/summary.json', flush=True)
    if child:
        if child.poll() is None:
            child.wait()
        raise SystemExit(child.returncode)


if __name__ == '__main__':
    main()
