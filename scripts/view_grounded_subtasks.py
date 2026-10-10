#!/usr/bin/env python3
"""Preview gripper splits on random source episodes, or play saved preprocessed subtasks."""
from __future__ import annotations

import argparse
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import random
import sys
from urllib.parse import urlparse
import webbrowser

DEFAULT_REPO = "felsager/community_dataset_v3_ee_smolVLA"


def segmentation_helpers():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import inspect_gripper_stages
    return inspect_gripper_stages


def episode_metadata(parts, needed, read):
    """Locate selected IDs in ordered LeRobot metadata shards by binary search."""
    import pandas as pd
    cache = {}
    selected = []
    for episode in sorted(needed):
        lo, hi = 0, len(parts) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if mid not in cache:
                table = read(parts[mid])
                if table.empty:
                    raise ValueError(f"Empty episode metadata shard: {parts[mid]}")
                cache[mid] = (int(table.episode_index.min()), int(table.episode_index.max()),
                              table[table.episode_index.isin(needed)])
            first, last, rows = cache[mid]
            if episode < first:
                hi = mid - 1
            elif episode > last:
                lo = mid + 1
            else:
                matches = rows[rows.episode_index == episode]
                if len(matches) != 1:
                    raise ValueError(f"Missing or duplicate episode metadata for {episode}")
                selected.append(matches)
                break
        else:
            raise ValueError(f"Episode {episode} not found in ordered LeRobot metadata")
    return pd.concat(selected, ignore_index=True)

HTML = r'''<!doctype html><meta charset="utf-8"><title>Grounded subtask viewer</title>
<style>body{font:16px system-ui;background:#111827;color:#e5e7eb;margin:24px}button,select{font:inherit;padding:8px;margin:4px;background:#273449;color:inherit;border:1px solid #64748b;border-radius:5px}button{cursor:pointer}.cameras{display:flex;flex-wrap:wrap;gap:12px}.camera{flex:1;min-width:280px}.videowrap{position:relative}video{display:block;width:100%;background:black}.heatmap{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}pre{white-space:pre-wrap}#status{color:#fbbf24}input[type=range]{width:100%}</style>
<h1>Gripper subtask viewer</h1><p id="repo"></p>
<label>Episode <select id="episode"></select></label><label>Subtask <select id="subtask"></select></label>
<button id="previous">Previous subtask</button><button id="play">Play / Pause</button><button id="next">Next subtask</button>
<label><input id="auto" type="checkbox" checked>Automatically advance through subtasks and episodes</label>
<p id="task"></p><p id="status">Loading metadata…</p><div class="cameras" id="cameras"></div>
<input id="seek" type="range" min="0" max="1" step="1"><p id="position"></p><pre id="diagnostics"></pre>
<p>Videos show clean footage; point heatmaps are drawn from the episode tracking sidecar. If a video cannot play, try a browser with AV1 support.</p>
<script>
const $=id=>document.getElementById(id);let catalog,ei=0,si=0,media=[],master=null,serial=0,transition=false,pending=false;const trackCache={};
function current(){return catalog.episodes[ei].subtasks[si]}
function pause(){media.forEach(m=>m.video.pause())}
async function play(){if(!master)return;try{await Promise.all(media.map(m=>m.video.play()))}catch(e){pause();$('status').textContent='Playback failed: '+e.message}}
function seek(frame){const e=catalog.episodes[ei];media.forEach(m=>{m.video.currentTime=m.offset+frame/e.fps})}
async function loadTracks(episode){if(trackCache[episode])return trackCache[episode];const response=await fetch(`/tracks/${episode}`);if(!response.ok)throw Error(`Could not load tracks for episode ${episode}`);trackCache[episode]=await response.json();return trackCache[episode]}
function pointsAt(camera,frame){const data=trackCache[catalog.episodes[ei].episode];if(!data)return [];const stage=data.stages.find(x=>frame>=x.start_frame&&frame<=x.end_frame);if(!stage)return [];const local=frame-stage.start_frame,third=stage.third_person[camera],sets=camera==='wrist'?[stage.pov,stage.previous_pov]:[third,third?.transition],points=[];for(const set of sets){if(!set||!set.tracks||!set.visibility)continue;for(let i=0;i<set.tracks.length;i++){if(set.visibility[i]?.[local]){const p=set.tracks[i]?.[local];if(p&&Number.isFinite(p[0])&&Number.isFinite(p[1]))points.push(p)}}}return points}
function drawHeatmap(item,frame){const {canvas,ctx,name}=item;if(!canvas.width||!canvas.height)return;const data=trackCache[catalog.episodes[ei].episode],w=canvas.width,h=canvas.height;ctx.clearRect(0,0,w,h);if(!data?.clean_video){item.lastFrame=frame;return}const points=pointsAt(name,frame);if(!points.length){item.lastFrame=frame;return}const sigma=(data.sigma||40)/4,rad=Math.max(3,Math.round(3*sigma)),s2=2*sigma*sigma,heat=new Float32Array(w*h);let max=0;for(const p of points){const cx=Math.round(p[0]/4),cy=Math.round(p[1]/4);for(let y=Math.max(0,cy-rad);y<=Math.min(h-1,cy+rad);y++)for(let x=Math.max(0,cx-rad);x<=Math.min(w-1,cx+rad);x++){const dx=x-cx,dy=y-cy,i=y*w+x;heat[i]+=Math.exp(-(dx*dx+dy*dy)/s2);if(heat[i]>max)max=heat[i]}}if(!max){item.lastFrame=frame;return}const image=ctx.createImageData(w,h),alpha=Math.round(255*(data.alpha??0.55));for(let i=0;i<heat.length;i++){const v=heat[i]/max,j=i*4;image.data[j]=Math.round(255*Math.max(0,Math.min(1,1.5-Math.abs(4*v-3))));image.data[j+1]=Math.round(255*Math.max(0,Math.min(1,1.5-Math.abs(4*v-2))));image.data[j+2]=Math.round(255*Math.max(0,Math.min(1,1.5-Math.abs(4*v-1))));image.data[j+3]=alpha}ctx.putImageData(image,0,0);item.lastFrame=frame}
function options(){const e=catalog.episodes[ei];$('episode').value=ei;$('subtask').replaceChildren();e.subtasks.forEach((s,j)=>$('subtask').add(new Option(`Subtask ${s.subtask ?? j}: ${s.primitive} [${s.start_frame}–${s.end_frame}]`,j)));$('subtask').value=si}
async function load(resume=false){pause();const request=++serial;transition=true;const e=catalog.episodes[ei],s=current();options();$('status').textContent='Loading camera videos…';$('task').textContent=e.task;
$('diagnostics').textContent=JSON.stringify({source_episode:e.source_episode,destination_episode:e.episode,boundary_mode:s.boundary_mode||'saved',saved_subtask_count:e.saved_subtask_count,displayed_subtask_count:e.subtasks.length,subtask:s.subtask,primitive:s.primitive,gripper_event:s.gripper_event,pov:s.pov,third_person:s.third_person},null,2);
$('seek').min=s.start_frame;$('seek').max=s.end_frame;$('seek').value=s.start_frame;$('cameras').replaceChildren();media=[];master=null;
try{const [loaded]=await Promise.all([Promise.all(e.cameras.map(cam=>new Promise((resolve,reject)=>{const div=document.createElement('div');div.className='camera';const label=document.createElement('h3');label.textContent=cam.name;const wrap=document.createElement('div');wrap.className='videowrap';const video=document.createElement('video');video.muted=true;video.playsInline=true;video.preload='auto';video.src=`/video/${e.episode}/${encodeURIComponent(cam.name)}`;const canvas=document.createElement('canvas');canvas.className='heatmap';const ctx=canvas.getContext('2d');wrap.append(video,canvas);div.append(label,wrap);$('cameras').append(div);video.onloadedmetadata=()=>{canvas.width=Math.max(1,Math.ceil(video.videoWidth/4));canvas.height=Math.max(1,Math.ceil(video.videoHeight/4));resolve({video,canvas,ctx,offset:cam.offset,name:cam.name,lastFrame:-1})};video.onerror=()=>reject(Error(`${cam.name}: could not load/decode video`))}))),loadTracks(e.episode)]);
if(request!==serial){loaded.forEach(m=>m.video.pause());return}media=loaded;master=(media.find(m=>m.name==='wrist')||media[0])?.video;if(!master)throw Error('No camera videos');master.onended=()=>{if(!transition&&$('auto').checked)move(1,true)};seek(s.start_frame);$('status').textContent=catalog.warning||'';transition=false;if(resume)await play();
}catch(e){if(request===serial){transition=false;$('status').textContent=String(e)}}}
function move(delta,resume){if(pending)return;let nextE=ei,nextS=si+delta;if(nextS>=catalog.episodes[nextE].subtasks.length){nextE++;nextS=0}else if(nextS<0){nextE--;if(nextE>=0)nextS=catalog.episodes[nextE].subtasks.length-1}if(nextE<0||nextE>=catalog.episodes.length){pause();$('status').textContent='End of dataset';return}ei=nextE;si=nextS;load(resume)}
$('episode').onchange=()=>{ei=Number($('episode').value);si=0;load(false)};$('subtask').onchange=()=>{si=Number($('subtask').value);load(false)};
$('play').onclick=()=>{if(master?.paused)play();else pause()};$('next').onclick=()=>move(1,master&&!master.paused);$('previous').onclick=()=>move(-1,master&&!master.paused);$('seek').oninput=()=>seek(Number($('seek').value));
function tick(){if(master&&!transition){const e=catalog.episodes[ei],s=current(),base=media.find(m=>m.video===master);const frame=Math.round((master.currentTime-base.offset)*e.fps);$('seek').value=Math.max(s.start_frame,Math.min(s.end_frame,frame));$('position').textContent=`Episode ${e.episode} · subtask ${s.subtask ?? si} · frame ${frame} / ${s.end_frame}`;media.forEach(m=>{if(m.lastFrame!==frame)drawHeatmap(m,frame)});
if(!master.paused){if(master.currentTime>=base.offset+(s.end_frame+1)/e.fps-.01){pause();if($('auto').checked)move(1,true)}else{media.forEach(m=>{if(m.video!==master&&Math.abs((m.video.currentTime-m.offset)-(master.currentTime-base.offset))>.15)m.video.currentTime=m.offset+master.currentTime-base.offset})}}}requestAnimationFrame(tick)}
fetch('/catalog').then(async r=>{if(!r.ok)throw Error(await r.text());return r.json()}).then(d=>{catalog=d;$('repo').textContent=d.repo;d.episodes.forEach((e,j)=>$('episode').add(new Option(`Episode ${e.episode} (source ${e.source_episode})`,j)));load();tick()}).catch(e=>$('status').textContent=String(e));
</script>'''


class Viewer:
    def __init__(self, repo, revision, root=None, max_episodes=3, episode_offset=0, recompute_settings=None,
                 random_episodes=None, seed=None):
        import pandas as pd
        from huggingface_hub import HfApi, hf_hub_download
        if max_episodes < 1 or episode_offset < 0:
            raise ValueError("Episode limit must be positive; episode offset must be nonnegative")
        if random_episodes is not None and random_episodes < 1:
            raise ValueError("Random episode count must be positive")
        self.repo, self.revision = repo, revision
        self.local = root is not None
        if self.local:
            self.root = root.resolve()
            filenames = [p.relative_to(self.root).as_posix() for p in (self.root / "point_tracks").glob("ep*.json")]
            filenames += [p.relative_to(self.root).as_posix() for p in (self.root / "meta/episodes").rglob("*.parquet")]
        else:
            api = HfApi()
            self.revision = api.repo_info(repo, repo_type="dataset", revision=revision).sha
            # Listing filenames does not download the dataset. Fetch only the
            # selected reports and metadata shards needed to locate their videos.
            filenames = api.list_repo_files(repo, repo_type="dataset", revision=self.revision)

        def file(name):
            return self.root / name if self.local else Path(hf_hub_download(
                repo, name, repo_type="dataset", revision=self.revision))

        info = json.loads(file("meta/info.json").read_text())
        parts = sorted(name for name in filenames if name.startswith("meta/episodes/") and name.endswith(".parquet"))
        if not parts:
            raise ValueError("Expected LeRobot v3 meta/episodes/*.parquet metadata")
        report_files = sorted(name for name in filenames if re.fullmatch(r"point_tracks/ep\d+\.json", name))
        raw_source = not report_files
        reports = []
        if raw_source:
            total = int(info["total_episodes"])
            available = range(episode_offset, total)
            ids = random.Random(seed).sample(available, min(random_episodes or max_episodes, len(available)))
            reports = [(ep, dict(episode=ep, destination_episode=ep, subtasks=[])) for ep in ids]
            if recompute_settings is None:
                recompute_settings = segmentation_helpers().parser().parse_args([])
        else:
            positions = list(range(episode_offset, len(report_files)))
            if random_episodes is not None:
                positions = random.Random(seed).sample(positions, min(random_episodes, len(positions)))
            else:
                positions = positions[:max_episodes]
            for position in positions:
                report = json.loads(file(report_files[position]).read_text())
                if report.get("subtasks"):
                    reports.append((position, report))
        if not reports:
            raise ValueError("No subtask reports in the selected episode range")
        legacy = any("destination_episode" not in r for _, r in reports)
        if legacy and len(report_files) != int(info.get("total_episodes", len(report_files))):
            raise ValueError("Legacy reports cannot be mapped safely: report count differs from destination episodes")
        needed = {int(r.get("destination_episode", position)) for position, r in reports}
        episodes = episode_metadata(parts, needed, lambda name: pd.read_parquet(file(name)))
        self.warning = ("Legacy reports: source episodes are mapped to destination episodes in sorted order. "
                        "This assumes preprocessing used ascending episode order." if legacy else "")
        self.episodes, self.videos, self.cached_videos = [], {}, {}
        self.track_data = {}
        self.recomputed = recompute_settings is not None
        data_tables = {}
        for position, report in reports:
            ep = int(report.get("destination_episode", position))
            rows = episodes[episodes["episode_index"] == ep]
            if len(rows) != 1:
                raise ValueError(f"Missing or duplicate metadata for destination episode {ep}")
            row = rows.iloc[0]
            if raw_source:
                tasks = row.get("tasks", [])
                report["task"] = tasks if isinstance(tasks, str) else " | ".join(str(t) for t in tasks)
            heatmap = report.get("heatmap", {})
            track_stages = []
            for record in report.get("subtasks", []):
                pov = record.get("pov", {})
                previous = pov.get("previous_stage") or {}
                track_stages.append({
                    "start_frame": record["start_frame"],
                    "end_frame": record["end_frame"],
                    "pov": {"tracks": pov.get("tracks", []), "visibility": pov.get("visibility", [])},
                    "previous_pov": {"tracks": previous.get("tracks", []), "visibility": previous.get("visibility", [])},
                    "third_person": {cam: {"tracks": value.get("tracks", []),
                                           "visibility": value.get("visibility", []),
                                           "transition": value.get("transition_from_previous", {})}
                                     for cam, value in record.get("third_person", {}).items()},
                })
            self.track_data[ep] = {"clean_video": "heatmap" in report,
                                   "sigma": heatmap.get("sigma", 40.0),
                                   "alpha": heatmap.get("alpha", 0.55), "stages": track_stages}
            cameras = []
            for name in ("wrist", "top", "side"):
                key = f"observation.images.{name}"
                if info["features"].get(key, {}).get("dtype") != "video":
                    continue
                path = info["video_path"].format(video_key=key,
                    chunk_index=int(row[f"videos/{key}/chunk_index"]),
                    file_index=int(row[f"videos/{key}/file_index"]))
                self.videos[(ep, name)] = path
                cameras.append(dict(name=name, offset=float(row[f"videos/{key}/from_timestamp"])))
            stages = []
            for record in report["subtasks"]:
                summary = {k: record[k] for k in ("subtask", "start_frame", "end_frame", "primitive")}
                summary["pov"] = {k: v for k, v in record.get("pov", {}).items()
                                  if k in {"source", "status", "reason"}}
                summary["third_person"] = {cam: {k: v for k, v in value.items() if k in
                    {"source", "status", "reason", "entity", "fallback_mode", "snapshot_frame", "snapshot_source", "tracking_fps"}}
                    for cam, value in record.get("third_person", {}).items()}
                stages.append(summary)
            if self.recomputed:
                import numpy as np
                helper = segmentation_helpers()
                column = "observation.state" if recompute_settings.gripper_source == "state" else "action"
                data_name = info["data_path"].format(chunk_index=int(row["data/chunk_index"]),
                                                     file_index=int(row["data/file_index"]))
                if data_name not in data_tables:
                    data_tables[data_name] = pd.read_parquet(file(data_name), columns=["episode_index", "frame_index", column],
                                                             filters=[("episode_index", "in", sorted(needed))])
                episode_data = data_tables[data_name]
                episode_data = episode_data[episode_data["episode_index"] == ep].sort_values("frame_index")
                if not len(episode_data) or not np.array_equal(episode_data["frame_index"].to_numpy(), np.arange(len(episode_data))):
                    raise ValueError(f"Episode {ep}: missing or noncontiguous frame data")
                signal = np.stack(episode_data[column].to_numpy())
                if signal.ndim != 2 or signal.shape[1] <= helper.GRIPPER_INDEX:
                    raise ValueError(f"Episode {ep}: no gripper at index {helper.GRIPPER_INDEX}")
                signal = signal[:, helper.GRIPPER_INDEX]
                if not np.isfinite(signal).all():
                    raise ValueError(f"Episode {ep}: nonfinite gripper values")
                events, preview_stages = helper.split_gripper_signal(signal, float(info["fps"]), recompute_settings)
                stages = [dict(subtask=i, start_frame=s.start, end_frame=s.end, primitive=s.primitive,
                    pov={"source": "raw_video" if raw_source else "saved_tracks"}, third_person={}, boundary_mode="recomputed_preview",
                    gripper_event=next((e for e in events if e["frame"] == s.end), None))
                    for i, s in enumerate(preview_stages)]
            self.episodes.append(dict(episode=ep, source_episode=report["episode"],
                task=report.get("task", ""), fps=float(info["fps"]), cameras=cameras, subtasks=stages,
                saved_subtask_count=len(report["subtasks"])))
        self.episodes.sort(key=lambda e: e["episode"])
        if self.recomputed:
            self.warning += (" Raw source videos; subtask boundaries are recomputed from gripper values. No dataset changes are written."
                             if raw_source else " Preview boundaries were recomputed from gripper values. Point overlays still follow saved tracking stages; this preview does not regenerate tracks or modify the dataset.")

    def video(self, episode, camera):
        if (episode, camera) in self.cached_videos:
            return self.cached_videos[(episode, camera)]
        filename = self.videos[(episode, camera)]
        if self.local:
            path = (self.root / filename).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError("Video outside dataset root")
            return path
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(self.repo, filename, repo_type="dataset", revision=self.revision))
        self.cached_videos[(episode, camera)] = path
        return path


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, viewer, **kwargs):
        self.viewer = viewer
        super().__init__(*args, **kwargs)

    def content(self, data, kind):
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        route = urlparse(self.path).path
        try:
            if route == "/":
                self.content(HTML.encode(), "text/html; charset=utf-8")
            elif route == "/catalog":
                self.content(json.dumps(dict(repo=self.viewer.repo, warning=self.viewer.warning,
                                             episodes=self.viewer.episodes)).encode(), "application/json")
            elif match := re.fullmatch(r"/tracks/(\d+)", route):
                data = self.viewer.track_data.get(int(match[1]), {"stages": []})
                self.content(json.dumps(data).encode(), "application/json")
            elif match := re.fullmatch(r"/video/(\d+)/(wrist|top|side)", route):
                self.stream(self.viewer.video(int(match[1]), match[2]))
            else:
                self.send_error(404)
        except (KeyError, FileNotFoundError, ValueError) as exc:
            self.send_error(404, str(exc))
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            self.log_error("%s", exc)
            self.send_error(500, "Unable to load video; see terminal")

    def stream(self, path):
        size = path.stat().st_size
        start, end = 0, size - 1
        value = self.headers.get("Range")
        if value:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", value)
            if not match or not any(match.groups()):
                self.send_error(416)
                return
            if match[1]:
                start = int(match[1]); end = min(end, int(match[2])) if match[2] else end
            else:
                start = max(0, size - int(match[2]))
            if start > end or start >= size:
                self.send_response(416); self.send_header("Content-Range", f"bytes */{size}"); self.end_headers()
                return
        self.send_response(206 if value else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if value:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as file:
            file.seek(start)
            remaining = end - start + 1
            while remaining:
                data = file.read(min(1024 * 1024, remaining))
                if not data:
                    break
                self.wfile.write(data); remaining -= len(data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", default=DEFAULT_REPO)
    p.add_argument("--revision", default="main")
    p.add_argument("--root", type=Path, help="Optional local preprocessed dataset root")
    p.add_argument("--max-episodes", type=int, default=3, help="Episode count (default: 3 random source episodes, or 3 saved reports)")
    p.add_argument("--episode-offset", type=int, default=0, help="Minimum source episode ID, or number of saved reports to skip")
    p.add_argument("--random-episodes", type=int, help="Sample X distinct episodes; overrides --max-episodes")
    p.add_argument("--seed", type=int, help="Repeat the same random episode selection")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--open-browser", action="store_true")
    p.add_argument("--recompute-stages", action="store_true", help="Preview new gripper splits without rewriting videos or the dataset")
    selection_parser = argparse.ArgumentParser(add_help=False)
    selection_parser.add_argument("--repo-id", default=DEFAULT_REPO)
    selection_parser.add_argument("--recompute-stages", action="store_true")
    preliminary, _ = selection_parser.parse_known_args()
    automatic_preview = preliminary.repo_id == DEFAULT_REPO
    if preliminary.recompute_stages or automatic_preview:
        segmentation_helpers().add_segmentation_arguments(p)
    args = p.parse_args()
    print(f"Loading subtask metadata from {args.repo_id}…", flush=True)
    viewer = Viewer(args.repo_id, args.revision, args.root, args.max_episodes, args.episode_offset,
                    args if args.recompute_stages or automatic_preview else None,
                    args.random_episodes, args.seed)
    print("Selected episodes: " + ", ".join(str(e["episode"]) for e in viewer.episodes), flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, viewer=viewer))
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"{len(viewer.episodes)} episodes. Open {url}; Ctrl+C to stop. Videos download on demand.", flush=True)
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
