#!/usr/bin/env python3
"""Browse the uploaded SO-101 point dataset in a local web browser (read only)."""
from __future__ import annotations

import argparse
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import math
from urllib.parse import urlparse
import webbrowser

HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SO-101 point dataset viewer</title><style>
body{font:16px system-ui;margin:0;background:#111827;color:#e5e7eb}main{max-width:1200px;margin:auto;padding:24px}
h1{font-size:24px;margin:0 0 8px}p{color:#aab5c6}nav{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:18px 0}
select,input,button{font:inherit;padding:8px;border-radius:6px;border:1px solid #536174;background:#1f2937;color:inherit}
button{cursor:pointer}button:disabled{opacity:.4;cursor:default}input[type=number]{width:80px}
#canvas{background:#030712;border-radius:12px;min-height:200px;display:flex;justify-content:center}
svg{width:100%;max-height:70vh}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#1f2937;padding:16px;border-radius:8px}
#status{color:#fbbf24}label{display:flex;gap:6px;align-items:center}a{color:#93c5fd}
</style></head><body><main>
<h1>SO-101 point dataset</h1><p id="repo"></p>
<nav><label>Split <select id="split"></select></label><label>Camera <select id="camera"></select></label>
<label>Episode <select id="episode"></select></label><label><input type="checkbox" id="overlay" checked>Show point</label></nav>
<nav><button id="prev">← Previous</button><button id="next">Next →</button><button id="random">Random</button>
<label>Sample <input id="jump" type="number" min="1"></label><span id="count"></span></nav>
<p id="status" role="status">Loading dataset…</p><div id="canvas"></div><pre id="details"></pre>
<p>Use ← / → to browse. Points are human labels, not model predictions. This viewer does not edit the dataset.</p>
</main><script>
const $=id=>document.getElementById(id);let rows=[],filtered=[],index=0,serial=0;
function choices(id,values,all){const el=$(id);el.replaceChildren();if(all)el.add(new Option('All',''));for(const v of values)el.add(new Option(v,v));}
function filter(){filtered=rows.filter(r=>(!$('split').value||r.split===$('split').value)&&(!$('camera').value||r.camera===$('camera').value)&&(!$('episode').value||String(r.episode)===$('episode').value));index=0;show();}
async function show(){const request=++serial;const row=filtered[index];$('prev').disabled=!row||index===0;$('next').disabled=!row||index===filtered.length-1;$('random').disabled=!row;$('jump').disabled=!row;
$('jump').value=row?index+1:'';$('jump').max=filtered.length;$('count').textContent=`of ${filtered.length} samples`;$('canvas').replaceChildren();$('details').textContent='';
if(!row){$('status').textContent='No samples match these filters.';return;}$('status').textContent='Loading image…';
try{const response=await fetch(`/sample/${row.id}`);if(!response.ok)throw Error(await response.text());const sample=await response.json();if(request!==serial)return;
const ns='http://www.w3.org/2000/svg';const svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox',`0 0 ${sample.width} ${sample.height}`);svg.setAttribute('role','img');svg.setAttribute('aria-label',`Image and point label: ${row.label}`);
const image=document.createElementNS(ns,'image');image.setAttribute('width',sample.width);image.setAttribute('height',sample.height);image.setAttribute('href',`/image/${row.id}`);image.addEventListener('error',()=>{$('status').textContent='Image failed to load.';});svg.append(image);
if(sample.point_xy&&$('overlay').checked){const [x,y]=sample.point_xy;const radius=Math.max(sample.width,sample.height)*.012;
const marker=document.createElementNS(ns,'circle');for(const [k,v]of Object.entries({cx:x,cy:y,r:radius,fill:'none',stroke:'#ff375f','stroke-width':Math.max(2,radius/4)}))marker.setAttribute(k,v);svg.append(marker);
const cross=document.createElementNS(ns,'path');cross.setAttribute('d',`M ${x-radius*1.6} ${y} H ${x+radius*1.6} M ${x} ${y-radius*1.6} V ${y+radius*1.6}`);cross.setAttribute('stroke','#ff375f');cross.setAttribute('stroke-width',Math.max(1,radius/6));svg.append(cross);}
$('canvas').append(svg);$('details').textContent=JSON.stringify({...row,...sample},null,2);$('status').textContent=sample.warning||'';
}catch(e){if(request===serial)$('status').textContent=String(e);}}
$('prev').onclick=()=>{if(index>0){index--;show();}};$('next').onclick=()=>{if(index+1<filtered.length){index++;show();}};
$('random').onclick=()=>{index=Math.floor(Math.random()*filtered.length);show();};$('jump').onchange=()=>{const n=Number($('jump').value);if(Number.isInteger(n)&&n>=1&&n<=filtered.length){index=n-1;show();}};
for(const id of ['split','camera','episode'])$(id).onchange=filter;$('overlay').onchange=show;
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName))return;if(e.key==='ArrowLeft')$('prev').click();if(e.key==='ArrowRight')$('next').click();});
fetch('/catalog').then(async r=>{if(!r.ok)throw Error(await r.text());return r.json();}).then(data=>{rows=data.rows;$('repo').textContent=data.repo;
choices('split',[...new Set(rows.map(r=>r.split))],true);choices('camera',[...new Set(rows.map(r=>r.camera))].sort(),true);choices('episode',[...new Set(rows.map(r=>r.episode))].sort((a,b)=>a-b),true);filter();}).catch(e=>{$('status').textContent=String(e);});
</script></body></html>'''


class DatasetViewer:
    def __init__(self, repo: str, config: str | None, revision: str):
        from datasets import Image, load_dataset
        self.repo = repo
        self.datasets = load_dataset(repo, name=config, revision=revision)
        self.rows = []
        self.locations = []
        for split, dataset in self.datasets.items():
            required = {'image', 'point_xy_100'}
            if not required.issubset(dataset.column_names):
                raise ValueError(f'{split}: missing columns {sorted(required-set(dataset.column_names))}')
            # Do not decode every image while building filters. Embedded image bytes
            # remain in the Arrow cache; only the selected image is decoded.
            self.datasets[split] = dataset.cast_column('image', Image(decode=False))
            metadata = dataset.remove_columns('image')
            for offset, row in enumerate(metadata):
                self.rows.append(dict(id=len(self.rows), split=split,
                                      episode=row.get('episode'), camera=row.get('camera', 'unknown'),
                                      label=row.get('label', ''), source_image=row.get('source_image', '')))
                self.locations.append((split, offset))
        if not self.rows:
            raise ValueError('The dataset has no samples.')

    def sample(self, sample_id: int):
        from PIL import Image
        if not 0 <= sample_id < len(self.locations):
            raise IndexError('Sample not found')
        split, offset = self.locations[sample_id]
        row = self.datasets[split][offset]
        raw = row['image']
        # HF uploads embed image bytes; a training VM's old absolute path is
        # deliberately ignored when bytes are present.
        source = BytesIO(raw['bytes']) if raw.get('bytes') is not None else raw['path']
        with Image.open(source) as loaded:
            image = loaded.convert('RGB')
        width, height = image.size
        point = None
        warning = None
        try:
            coords = row['point_xy_100']
            if len(coords) != 2:
                raise ValueError('expected two coordinates')
            x, y = [float(value) for value in coords]
            if not all(math.isfinite(v) and 0 <= v < 100 for v in (x, y)):
                raise ValueError('coordinates must be finite and in [0, 100)')
            # Exact inverse of prepare_molmo_so101.py, not width-1/height-1.
            point = [x * width / 100, y * height / 100]
        except (TypeError, ValueError, KeyError) as exc:
            warning = f'Invalid point annotation: {exc}'
        return image, dict(width=width, height=height, point_xy=point,
                           point_xy_100=row.get('point_xy_100') if point else None, warning=warning)


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, viewer: DatasetViewer, **kwargs):
        self.viewer = viewer
        super().__init__(*args, **kwargs)

    def send_content(self, content: bytes, content_type: str, status=200):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path == '/':
                self.send_content(HTML.encode(), 'text/html; charset=utf-8')
            elif path == '/catalog':
                self.send_content(json.dumps(dict(repo=self.viewer.repo, rows=self.viewer.rows)).encode(), 'application/json')
            elif path.startswith(('/sample/', '/image/')):
                sample_id = int(path.rsplit('/', 1)[1])
                image, metadata = self.viewer.sample(sample_id)
                if path.startswith('/sample/'):
                    self.send_content(json.dumps(metadata, allow_nan=False).encode(), 'application/json')
                else:
                    output = BytesIO()
                    image.save(output, format='PNG')
                    self.send_content(output.getvalue(), 'image/png')
            else:
                self.send_content(b'Not found', 'text/plain', 404)
        except (ValueError, IndexError):
            self.send_content(b'Invalid sample or image data', 'text/plain', 404)
        except Exception as exc:
            self.log_error('Could not load sample: %s', exc)
            self.send_content(b'Could not load sample; see the viewer terminal.', 'text/plain', 500)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-id', default='kdaterao/so101_molmo2_gripper_preprocessed')
    parser.add_argument('--config', default=None, help='Optional Hugging Face dataset configuration')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--open-browser', action='store_true')
    args = parser.parse_args()
    print(f'Loading {args.repo_id} (uses HF_TOKEN or your saved Hugging Face login)…', flush=True)
    try:
        viewer = DatasetViewer(args.repo_id, args.config, args.revision)
    except Exception as exc:
        raise SystemExit(f'Unable to load dataset: {exc}\nFor private datasets, run hf auth login first.') from exc
    server = ThreadingHTTPServer((args.host, args.port), partial(Handler, viewer=viewer))
    url = f'http://{args.host}:{server.server_port}'
    print(f'{len(viewer.rows)} samples ready. Open {url}\nCtrl+C to stop.', flush=True)
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
