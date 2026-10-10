#!/usr/bin/env python3
"""Inspect preprocessing's gripper segmentation without loading models or decoding videos."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import html
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robot101.data.utilities.episode_io import load_episode_metadata
from robot101.data.utilities.episode_helpers import GRIPPER_INDEX, _load_lerobot
from robot101.data.utilities.preprocess_config import build_parser
from robot101.data.utilities.stages import Stage, events_from_gripper_thresholds, stages_from_events


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src-repo-id", default=build_parser().get_default("src_repo_id"))
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--src-root", type=Path, help="Optional local LeRobot source directory")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/gripper_stages"))
    add_segmentation_arguments(p)
    return p


def add_segmentation_arguments(p):
    """Use the same detector options in the chart and video preview."""
    defaults = build_parser()
    p.add_argument("--gripper-source", choices=["state", "action"], default=defaults.get_default("gripper_source"))
    p.add_argument("--gripper-event-mode", choices=["movement", "bands"], default=defaults.get_default("gripper_event_mode"))
    p.add_argument("--first-primitive", choices=["grasp", "release"], default=defaults.get_default("first_primitive"))
    for name, kind in [
        ("max-stages", int), ("min-stage-frames", int),
        ("gripper-closed-frac", float), ("gripper-open-frac", float),
        ("gripper-min-dwell-frames", int), ("gripper-vel-stall", float),
        ("gripper-vel-min", float), ("gripper-smooth-window", int),
        ("gripper-min-change-frac", float), ("gripper-min-change-abs", float),
    ]:
        p.add_argument(f"--{name}", type=kind, default=defaults.get_default(name.replace("-", "_")))


def split_gripper_signal(grip, fps, args):
    if args.min_stage_frames < 1 or args.max_stages < 1:
        raise ValueError("Stage limits must be positive")
    events = events_from_gripper_thresholds(
        grip, fps=fps, closed_frac=args.gripper_closed_frac, open_frac=args.gripper_open_frac,
        min_dwell_frames=args.gripper_min_dwell_frames, vel_stall=args.gripper_vel_stall,
        vel_min=args.gripper_vel_min, smooth_window=args.gripper_smooth_window,
        min_change_frac=args.gripper_min_change_frac if args.gripper_event_mode == "movement" else None,
        min_change_abs=args.gripper_min_change_abs)
    stages = stages_from_events(events, len(grip), args.min_stage_frames, args.first_primitive)
    if len(stages) > args.max_stages:
        stages = stages[:args.max_stages]
        stages[-1] = Stage(stages[-1].start, len(grip) - 1, "none")
    return events, stages


PAGE = r'''<!doctype html><meta charset="utf-8">
<title>Gripper stages</title>
<style>body{font:15px system-ui;margin:30px;color:#172334;max-width:1200px}
canvas{width:100%;height:400px;border:1px solid #ccd5df}button{margin:4px;padding:8px;cursor:pointer}
table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}
code{background:#eef2f6;padding:3px}#readout{min-height:24px;font-family:monospace}</style>
<h1>__TITLE__</h1><p>Selected signal: <b id="source"></b>. Blue: state; orange: action.
Colored regions: stages. Vertical lines: detected grasp/release events.
Dashed lines: open/closed thresholds when using the band detector.</p>
<p id="rule">Detection requires a substantial directional movement followed by settling.</p>
<p>Stage ranges are inclusive; adjacent stages share their boundary frame, matching preprocessing.</p>
<div id="buttons"></div><canvas id="plot"></canvas><p id="readout">Move the pointer over the chart to inspect a frame.</p>
<h2>Stages</h2><table><thead><tr><th>Stage</th><th>Frames (inclusive)</th><th>Time (s)</th><th>Frames</th><th>Primitive at end</th></tr></thead><tbody id="stages"></tbody></table>
<h2>Detected events</h2><p id="note"></p><table><thead><tr><th>Frame</th><th>Time (s)</th><th>Event</th><th>Raw value</th><th>Smoothed value</th><th>Used as stage boundary</th></tr></thead><tbody id="events"></tbody></table>
<script>
const d=__DATA__, c=document.getElementById('plot'), ctx=c.getContext('2d');
let lo=0,hi=d.state.length-1;
const colors=['#e1edf9','#fbe9d8','#e3f2e4','#efe4f8'];
const t=i=>d.timestamps[i].toFixed(3);
document.getElementById('source').textContent=d.gripper_source;
document.getElementById('rule').textContent=d.settings.gripper_event_mode==='movement'
 ? `Movement detector: at least ${(d.settings.gripper_min_change_frac*100).toFixed(0)}% of the smoothed episode range (and absolute minimum ${d.settings.gripper_min_change_abs}), followed by settling.`
 : 'Band detector: settled open/closed gripper with recent motion.';
const button=(text,start,end)=>{let b=document.createElement('button');b.textContent=text;b.onclick=()=>{lo=start;hi=end;draw()};document.getElementById('buttons').append(b)};
button('Whole episode',0,hi);
d.stages.forEach((s,i)=>{button('Stage '+i,s.start,s.end);let r=document.createElement('tr');
 r.innerHTML=`<td>${i}</td><td>${s.start}–${s.end}</td><td>${t(s.start)}–${t(s.end)}</td><td>${s.end-s.start+1}</td><td>${s.primitive}</td>`;document.getElementById('stages').append(r)});
d.events.forEach(e=>{let r=document.createElement('tr');r.innerHTML=`<td>${e.frame}</td><td>${t(e.frame)}</td><td>${e.type}</td><td>${e.gripper.toFixed(4)}</td><td>${e.g_smooth.toFixed(4)}</td><td>${d.stages.slice(0,-1).some(s=>s.end===e.frame)?'Yes':'No (stage length/end filtering)'}</td>`;document.getElementById('events').append(r)});
document.getElementById('note').textContent=d.events.length?'All detector events are listed, including those filtered out when constructing stages.':'No grasp/release events detected; the episode remains one stage.';
function draw(){c.width=Math.max(600,c.clientWidth)*devicePixelRatio;c.height=400*devicePixelRatio;ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
 const W=c.width/devicePixelRatio,H=400,L=65,R=W-25,T=30,B=350;
 const values=d.state.slice(lo,hi+1).concat(d.action.slice(lo,hi+1));
 if(d.events.length)values.push(d.events[0].closed_thresh,d.events[0].open_thresh);
 let mn=Math.min(...values),mx=Math.max(...values),pad=(mx-mn||1)*.08;mn-=pad;mx+=pad;
 const x=i=>L+(i-lo)/Math.max(1,hi-lo)*(R-L),y=v=>B-(v-mn)/(mx-mn)*(B-T);
 ctx.clearRect(0,0,W,H);
 d.stages.forEach((s,i)=>{let a=Math.max(lo,s.start),b=Math.min(hi,s.end);if(a>b)return;ctx.fillStyle=colors[i%colors.length];ctx.fillRect(x(a),T,Math.max(1,x(b)-x(a)),B-T);ctx.fillStyle='#334155';ctx.fillText('Stage '+i,x(a)+5,T+14)});
 ctx.font='12px system-ui';for(let j=0;j<=4;j++){let v=mn+(mx-mn)*j/4;ctx.strokeStyle='#d0d6df';ctx.beginPath();ctx.moveTo(L,y(v));ctx.lineTo(R,y(v));ctx.stroke();ctx.fillStyle='#334155';ctx.fillText(v.toFixed(2),4,y(v)+4);let f=Math.round(lo+(hi-lo)*j/4);ctx.fillText(`${f} / ${t(f)}s`,x(f)-20,B+22)}
 if(d.events.length&&d.settings.gripper_event_mode==='bands'){ctx.setLineDash([5,5]);for(let k of ['closed_thresh','open_thresh']){ctx.strokeStyle='#64748b';ctx.beginPath();ctx.moveTo(L,y(d.events[0][k]));ctx.lineTo(R,y(d.events[0][k]));ctx.stroke()}ctx.setLineDash([])}
 for(let [key,color] of [['state','#2563eb'],['action','#d97706']]){ctx.strokeStyle=color;ctx.lineWidth=key===d.gripper_source?2.5:1;ctx.beginPath();for(let i=lo;i<=hi;i++){if(i===lo)ctx.moveTo(x(i),y(d[key][i]));else ctx.lineTo(x(i),y(d[key][i]))}ctx.stroke()}
 ctx.lineWidth=1;d.events.forEach(e=>{if(e.frame<lo||e.frame>hi)return;ctx.strokeStyle=e.type==='grasp'?'#b91c1c':'#15803d';ctx.beginPath();ctx.moveTo(x(e.frame),T);ctx.lineTo(x(e.frame),B);ctx.stroke();ctx.fillStyle=ctx.strokeStyle;ctx.fillText(e.type,x(e.frame)+3,T+30)});
}
c.onmousemove=e=>{let rect=c.getBoundingClientRect(),i=Math.max(lo,Math.min(hi,Math.round(lo+(e.clientX-rect.left-65)/(rect.width-90)*Math.max(1,hi-lo))));document.getElementById('readout').textContent=`Frame ${i} | ${t(i)}s | state ${d.state[i].toFixed(4)} | action ${d.action[i].toFixed(4)} | stages ${d.stages.map((s,j)=>i>=s.start&&i<=s.end?j:null).filter(j=>j!==null).join(', ')}`};
window.onresize=draw;draw();
</script>'''


def main():
    args = parser().parse_args()
    if args.episode < 0 or args.max_stages < 1 or args.min_stage_frames < 1:
        raise SystemExit("Episode must be nonnegative; stage limits must be positive")
    Dataset, Metadata, _, cache = _load_lerobot()
    root = args.src_root or cache / args.src_repo_id
    meta = Metadata(args.src_repo_id, root=root)
    if args.episode >= meta.total_episodes:
        raise SystemExit(f"Episode {args.episode} outside dataset ({meta.total_episodes} episodes)")
    source = Dataset(args.src_repo_id, episodes=[args.episode], root=root, video_backend="pyav")
    cameras = [name.removeprefix("observation.images.") for name, f in meta.features.items()
               if name.startswith("observation.images.") and f.get("dtype") in {"video", "image"}]
    data = load_episode_metadata(source, cameras)
    for key in ("states", "actions"):
        if data[key].shape[1] <= GRIPPER_INDEX:
            raise SystemExit(f"{key} has no gripper at index {GRIPPER_INDEX}")
        if not np.isfinite(data[key][:, GRIPPER_INDEX]).all():
            raise SystemExit(f"{key} gripper contains nonfinite values")
    fps = float(meta.fps)
    grip = data["gripper"] if args.gripper_source == "state" else data["action_gripper"]
    events, stages = split_gripper_signal(grip, fps, args)
    report = dict(repo_id=args.src_repo_id, episode=args.episode, fps=fps,
                  gripper_source=args.gripper_source, gripper_index=GRIPPER_INDEX,
                  timestamps=data["timestamps"], state=data["gripper"].tolist(),
                  action=data["action_gripper"].tolist(), events=events,
                  stages=[asdict(s) for s in stages],
                  settings={k: v for k, v in vars(args).items() if k.startswith("gripper_") or k in
                            {"max_stages", "min_stage_frames", "first_primitive"}})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"ep{args.episode:06d}"
    stem.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    title = html.escape(f"{args.src_repo_id} — episode {args.episode}")
    stem.with_suffix(".html").write_text(PAGE.replace("__TITLE__", title).replace(
        "__DATA__", json.dumps(report).replace("<", "\\u003c")))
    with stem.with_suffix(".csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "time_s", "state_gripper", "action_gripper", "stage_ids", "event"])
        by_frame = {e["frame"]: e["type"] for e in events}
        for i in range(data["n"]):
            writer.writerow([i, data["timestamps"][i], data["gripper"][i], data["action_gripper"][i],
                             ";".join(str(j) for j, s in enumerate(stages) if s.start <= i <= s.end),
                             by_frame.get(i, "")])
    print(f"Episode {args.episode}: {data['n']} frames, {len(events)} events, {len(stages)} stages")
    for i, stage in enumerate(stages):
        print(f"  stage {i}: frames {stage.start}–{stage.end} (inclusive), primitive={stage.primitive}")
    print(f"Open {stem.with_suffix('.html').resolve()} in a browser; JSON and CSV saved alongside it.")


if __name__ == "__main__":
    main()
