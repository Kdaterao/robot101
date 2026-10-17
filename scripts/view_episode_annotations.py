#!/usr/bin/env python3
"""Inspect Hugging Face or local hierarchical segmentation and captions."""
from __future__ import annotations

import argparse
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
from urllib.parse import urlparse
import webbrowser


HTML = r'''<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Episode annotation viewer</title>
<style>
body{font:15px system-ui;background:#111827;color:#e5e7eb;margin:24px}
button,select{font:inherit;background:#253449;color:inherit;padding:8px;border:1px solid #64748b;border-radius:6px}
button{cursor:pointer}h1{font-size:24px}h2{font-size:19px}.muted{color:#aebbd0}
.panel{background:#1d293b;padding:16px;border-radius:10px;margin:14px 0}
#timeline{display:flex;gap:3px;min-height:54px;overflow:hidden}#timeline button{min-width:0;overflow:hidden;padding:6px 2px;background:#23465f}
#timeline .active{background:#047857}#layout{display:grid;grid-template-columns:minmax(280px,1fr) minmax(360px,2fr);gap:16px}
#nodes{max-height:65vh;overflow:auto}#nodes button{display:block;width:100%;text-align:left;margin-bottom:5px;font-size:13px}
#nodes .active{border-color:#34d399;background:#24533f}#caption{white-space:pre-wrap;line-height:1.5}
#frames{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}figure{margin:0}img{width:100%;border-radius:5px}figcaption{font-size:12px;color:#b7c5da}
.warning{color:#fbbf24}#status{min-height:22px}#task{white-space:pre-wrap}
@media(max-width:850px){#layout{grid-template-columns:1fr}}
</style>
<h1>Episode segmentation and captions</h1>
<p class="muted">Local inspection frames, not video playback. Boundaries are [start, end) source frames.</p>
<label>Episode <select id="episode"></select></label>
<label>Nodes <select id="mode"><option value="selected">Selected subtasks</option><option value="all">Full caption hierarchy</option></select></label>
<label>Text <select id="textmode"><option value="final">Rewritten annotation when available</option><option value="original">Original model caption</option></select></label>
<button id="refresh">Refresh results</button>
<label><input type="checkbox" id="auto" checked> Refresh every 10 seconds</label>
<p id="status" class="warning"></p><p id="task"></p>
<div class="panel"><h2>Controller subtask cut</h2><div id="timeline"></div><p id="method" class="muted"></p></div>
<div id="layout"><div class="panel"><h2>Hierarchy nodes</h2><div id="nodes"></div></div>
<div class="panel"><h2 id="title">Select a node</h2><p id="bounds" class="muted"></p><p id="state" class="warning"></p><p id="action"></p><div id="caption"></div>
<details id="originalbox"><summary>Original model caption</summary><p id="original"></p></details>
<details id="previousbox"><summary>Previous-subtask context supplied to the model</summary><pre id="previous" style="white-space:pre-wrap"></pre></details>
<details id="reviewbox"><summary>Review diagnostics and critique</summary><pre id="review" style="white-space:pre-wrap"></pre></details>
<h2>Sampled frames</h2><div id="frames"></div></div></div>
<script>
const $=id=>document.getElementById(id);let catalog=[],episodeId=null,nodeId=null,current;
function caption(n){const raw=n.caption||{},state=n.aggregation,a=state?.annotation;
 if($('textmode').value==='final'&&a)return {...raw,text:a.detailed_caption,status:state.status,truncated:false};return raw}
function selectNode(id){nodeId=id;render()}
function render(){
 current=catalog.find(e=>e.episode_index===episodeId);if(!current)return;
 const nodes=current.nodes||[],selected=new Set(current.selected_node_ids||[]),byId=new Map(nodes.map(n=>[n.id,n]));
 const cut=nodes.filter(n=>selected.has(n.id)).sort((a,b)=>a.start_frame-b.start_frame);
 $('task').textContent=`Source: ${current.source_repo_id||'?'} · camera: ${current.camera||'?'} · episode ${episodeId}\nTask metadata: ${current.task||'(none)'}`;
 $('method').textContent=`${nodes.length} retained nodes · ${cut.length} selected subtasks · ${current.fps} FPS · ${current.n_frames||'?'} source frames`;
 $('timeline').replaceChildren();
 for(const n of cut){const b=document.createElement('button');b.textContent=n.id;b.title=`${n.id}: ${(n.start_frame/current.fps).toFixed(2)}–${(n.end_frame/current.fps).toFixed(2)} s`;b.style.flex=String(n.end_frame-n.start_frame);b.className=n.id===nodeId?'active':'';b.onclick=()=>selectNode(n.id);$('timeline').append(b)}
 const shown=($('mode').value==='selected'?cut:nodes.slice().sort((a,b)=>a.start_frame-b.start_frame||b.end_frame-a.end_frame));
 if(!shown.some(n=>n.id===nodeId))nodeId=shown[0]?.id;
 $('nodes').replaceChildren();
 for(const n of shown){const b=document.createElement('button'),c=caption(n);let depth=0,p=n.parent_id||n.parent,seen=new Set([n.id]);while(p&&byId.has(p)&&!seen.has(p)){seen.add(p);depth++;const ancestor=byId.get(p);p=ancestor.parent_id||ancestor.parent}
 b.style.paddingLeft=`${8+Math.min(depth,12)*10}px`;b.className=n.id===nodeId?'active':'';b.textContent=`${n.id} · ${(n.start_frame/current.fps).toFixed(2)}–${(n.end_frame/current.fps).toFixed(2)}s · ${n.caption_kind||'node'} · ${c.status||'pending'}`;b.onclick=()=>selectNode(n.id);$('nodes').append(b)}
 const n=byId.get(nodeId);$('previousbox').hidden=!n?.caption?.previous_subtask;$('previous').textContent=n?.caption?.previous_subtask?JSON.stringify(n.caption.previous_subtask,null,2):'';$('frames').replaceChildren();if(!n){$('title').textContent='No retained nodes';$('bounds').textContent='';$('state').textContent='Episode needs review';$('caption').textContent=JSON.stringify(current.review_reasons||[]);return}
 const c=caption(n);$('title').textContent=`${n.id}${selected.has(n.id)?' · selected subtask':' · hierarchy context'}`;
 $('bounds').textContent=`Frames [${n.start_frame}, ${n.end_frame}) · ${(n.start_frame/current.fps).toFixed(2)}–${(n.end_frame/current.fps).toFixed(2)} seconds · parent ${n.parent_id||n.parent||'none'} · children ${(n.children||[]).join(', ')||'none'}`;
 $('state').textContent=`Caption: ${c.status||'pending'}${c.truncated?' · truncated, review required':''}${c.error?' · '+c.error:''}`;
 $('caption').textContent=c.text||c.detailed_caption||'No completed caption yet. Refresh while the annotation process runs.';
 const aggregation=n.aggregation,a=aggregation?.annotation,showFinal=$('textmode').value==='final';
 if(showFinal&&aggregation)$('state').textContent=`Aggregation: ${aggregation.status}${a?.robotics?.review_status?' · '+a.robotics.review_status:''}${aggregation.error?' · '+aggregation.error:''}${!a?' · showing original caption':''}`;
 $('action').textContent=showFinal&&a?`Action: ${a.brief_action}\n${a.detailed_action}`:'';
 $('action').style.whiteSpace='pre-wrap';$('originalbox').hidden=!a;$('original').textContent=n.caption?.text||'Unavailable';
 $('reviewbox').hidden=!a;$('review').textContent=a?JSON.stringify({robotics:a.robotics,critique:a.critique},null,2):'';
 const ids=[...new Set(c.sample_frames||n.preview_frames||[])];
 const views=n.caption?.views?.length?n.caption.views:[{camera:current.camera,image_subdir:'',sample_frames:ids}];
 for(const view of views)for(const id of [...new Set(view.sample_frames||ids)]){const sub=view.image_subdir||'';if(!/^[A-Za-z0-9_-]*$/.test(sub))continue;const figure=document.createElement('figure'),img=document.createElement('img'),label=document.createElement('figcaption');img.loading='lazy';img.src=`/images/ep${String(episodeId).padStart(6,'0')}/${sub?encodeURIComponent(sub)+'/':''}f${String(id).padStart(6,'0')}.jpg`;img.alt=`${view.camera} frame ${id}`;img.onerror=()=>{img.remove();label.textContent+= ' · image unavailable'};label.textContent=`${view.camera} · frame ${id} · ${(id/current.fps).toFixed(2)} s`;figure.append(img,label);$('frames').append(figure)}
 if(n.caption?.helper_status)$('state').textContent+=` · helper: ${n.caption.helper_status}`;
 if(!ids.length)$('frames').textContent='No saved inspection frames for this node yet.';
}
async function load(){try{const response=await fetch('/catalog',{cache:'no-store'});if(!response.ok)throw Error(await response.text());const data=await response.json();catalog=data.episodes;if(!catalog.some(e=>e.episode_index===episodeId))episodeId=catalog[0]?.episode_index;$('episode').replaceChildren();for(const e of catalog)$('episode').add(new Option(`Episode ${e.episode_index}`,e.episode_index));$('episode').value=String(episodeId);$('status').textContent=data.errors.join(' · ')||(!catalog.length?'No segmentation artifacts found yet.':'');render()}catch(e){$('status').textContent=String(e)}}
 $('episode').onchange=()=>{episodeId=Number($('episode').value);nodeId=null;render()};$('mode').onchange=()=>render();$('textmode').onchange=()=>render();$('refresh').onclick=load;setInterval(()=>{if($('auto').checked)load()},10000);load();
</script>'''


def load_catalog(root, limit, selected):
    paths = {}
    # A finished episode artifact overrides the raw segmentation record.
    for directory in ("segments", "episodes"):
        for path in (root / directory).glob("ep*.json"):
            if re.fullmatch(r"ep\d+\.json", path.name):
                paths[int(path.stem[2:])] = path
    ids = sorted(paths)
    if selected is not None:
        ids = [i for i in ids if i in selected]
    episodes, errors = [], []
    for episode in ids[:limit]:
        try:
            record = json.loads(paths[episode].read_text())
            for node in record.get("nodes", []):
                node_id = node["id"]
                if not re.fullmatch(r"n\d+", node_id):
                    continue
                path = root / "node_captions" / f"ep{episode:06d}" / f"{node_id}.json"
                contextual = root / "context_captions" / f"ep{episode:06d}" / f"{node_id}.json"
                if contextual.exists():
                    path = contextual
                if path.exists():
                    try:
                        node["caption"] = json.loads(path.read_text())
                    except (OSError, ValueError) as exc:
                        errors.append(f"{path.name}: {exc}")
                if not node.get("caption", {}).get("sample_frames"):
                    start, end = node["start_frame"], node["end_frame"]
                    middle = (start + end - 1) // 2
                    candidates = [start, middle, end - 1]
                    node["preview_frames"] = [i for i in dict.fromkeys(candidates)
                        if (root / "images" / f"ep{episode:06d}" / f"f{i:06d}.jpg").is_file()]
            episodes.append(record)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"Episode {episode}: {exc}")
    return dict(episodes=episodes, errors=errors)


def hub_image_loader(repo_id, revision):
    from huggingface_hub import HfApi, hf_hub_download
    files = set(HfApi().list_repo_files(repo_id, repo_type="dataset", revision=revision))

    def image_path(relative):
        name = f"captioning/{relative}"
        if name not in files:
            return None
        return Path(hf_hub_download(repo_id, name, repo_type="dataset", revision=revision))
    return image_path


def manifest_images(root):
    path = root / "caption_source.json"
    if not path.exists():
        return None
    source = json.loads(path.read_text())
    return hub_image_loader(source["requested"]["repo_id"], source["resolved_revision"])


def load_hub(repo_id, revision, limit, selected, stage="captioning"):
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    revision = api.repo_info(repo_id, repo_type="dataset", revision=revision).sha
    files = set(api.list_repo_files(repo_id, repo_type="dataset", revision=revision))
    episodes = {}
    for name in files:
        match = re.fullmatch(rf"{stage}/episodes/ep(\d+)\.json", name)
        if match:
            episodes[int(match[1])] = name
    ids = sorted(episodes)
    if selected is not None:
        missing = selected - episodes.keys()
        if missing:
            raise ValueError(f"Episodes not found in the dataset: {sorted(missing)}")
        ids = [i for i in ids if i in selected]
    ids = ids[:limit]
    if not ids:
        raise ValueError(f"No {stage}/episodes/ep*.json artifacts found in this dataset")
    root = None
    for episode in ids:
        print(f"Loading episode {episode} from {repo_id}…", flush=True)
        path = Path(hf_hub_download(repo_id, episodes[episode], repo_type="dataset", revision=revision))
        root = path.parent.parent

    manifest = f"{stage}/caption_source.json"
    fallback = None
    if stage == "aggregation" and manifest in files:
        hf_hub_download(repo_id, manifest, repo_type="dataset", revision=revision)
        fallback = manifest_images(root)

    def image_path(relative):
        name = f"captioning/{relative}"
        if name not in files:
            return fallback(relative) if fallback else None
        return Path(hf_hub_download(repo_id, name, repo_type="dataset", revision=revision))

    # Pin to an immutable snapshot and restrict catalog to this selection,
    # even if other episodes already exist in the shared HF cache.
    return root, set(ids), image_path, revision


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, root, limit, selected, hub_image=None, **kwargs):
        self.root, self.limit, self.selected = root, limit, selected
        self.hub_image = hub_image
        super().__init__(*args, **kwargs)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            payload, content_type = HTML.encode(), "text/html; charset=utf-8"
        elif path == "/catalog":
            payload = json.dumps(load_catalog(self.root, self.limit, self.selected)).encode()
            content_type = "application/json"
        elif re.fullmatch(r"/images/ep\d+/(?:[A-Za-z0-9_-]+/)?f\d+\.jpg", path):
            relative = path.lstrip("/")
            if self.hub_image:
                episode = int(relative.split("/")[1][2:])
                if self.selected is not None and episode not in self.selected:
                    self.send_error(404)
                    return
                try:
                    target = self.hub_image(relative)
                except Exception:
                    self.send_error(502, "Unable to download inspection image")
                    return
            else:
                target = (self.root / relative).resolve()
                if not target.is_relative_to(self.root):
                    target = None
            if target is None or not target.is_file():
                self.send_error(404)
                return
            payload, content_type = target.read_bytes(), "image/jpeg"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="kdaterao/so101_action_annotations")
    parser.add_argument("--revision", default="main", help="Hub revision, pinned to a snapshot at startup")
    parser.add_argument("--stage", choices=["captioning", "aggregation"], default="captioning",
                        help="Hub artifact folder to inspect")
    parser.add_argument("--root", type=Path, help="Use local artifacts instead of the Hub")
    parser.add_argument("--episodes", help="Optional comma-separated episode IDs, e.g. 0,1,2")
    parser.add_argument("--max-episodes", type=int, default=3)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()
    if args.max_episodes < 1:
        parser.error("--max-episodes must be positive")
    try:
        selected = None if args.episodes is None else {int(i.strip()) for i in args.episodes.split(",")}
    except ValueError:
        parser.error("--episodes expects comma-separated integer IDs")
    hub_image = None
    if args.root is not None:
        root = args.root.resolve()
        if not root.is_dir():
            parser.error(f"Annotation directory does not exist: {root}")
        try:
            hub_image = manifest_images(root)
        except Exception as exc:
            print(f"Inspection images unavailable from source Hub repo: {exc}", flush=True)
    else:
        try:
            root, selected, hub_image, revision = load_hub(args.repo_id, args.revision, args.max_episodes, selected, args.stage)
        except Exception as exc:
            parser.error(f"Unable to load annotation dataset: {exc}. For private datasets, run hf auth login.")
        print(f"Snapshot: {args.repo_id}@{revision}; images download only when displayed.", flush=True)
    handler = partial(Handler, root=root, limit=args.max_episodes, selected=selected, hub_image=hub_image)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    address = f"http://{args.host}:{server.server_port}"
    print(f"Annotation viewer: {address}\nReading {root}; at most {args.max_episodes} episodes. Ctrl+C to stop.", flush=True)
    if args.open_browser:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
