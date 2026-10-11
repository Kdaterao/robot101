#!/usr/bin/env python3
"""Collect exact-frame SAM gripper prompt corrections from mask_review items."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-dir', type=Path, required=True,
                        help='point_tracks/mask_review from the preprocessor')
    parser.add_argument('--output', type=Path, help='Defaults to point_tracks/mask_corrections.json')
    parser.add_argument('--overwrite', action='store_true', help='Revisit previously saved correction keys')
    parser.add_argument('--max-width', type=int, default=1200)
    args = parser.parse_args()
    if args.max_width < 1:
        parser.error('--max-width must be positive')
    import tkinter as tk
    from PIL import Image, ImageTk
    output = args.output or args.review_dir.parent / 'mask_corrections.json'
    corrections = json.loads(output.read_text()) if output.exists() else {}
    root = tk.Tk()
    root.withdraw()
    stopped = [False]
    for path in sorted(args.review_dir.glob('ep*_stage*.json')):
        review = json.loads(path.read_text())
        key = review['correction_key']
        if key in corrections and not args.overwrite:
            continue
        image_path = args.review_dir / review.get('review_image', '')
        if not image_path.is_file():
            print(f'Skip {key}: no review image')
            continue
        image = Image.open(image_path).convert('RGB')
        w, h = image.size
        ratio = min(1.0, args.max_width/w)
        dw, dh = max(1, round(w*ratio)), max(1, round(h*ratio))
        window = tk.Toplevel(root)
        window.title(f'Gripper mask {key}, frame {review["frame_index"]}')
        tk.Label(window, text='Left click: gripper body/finger. Right click: exclude. Enter: save. U: undo. N: skip. Q: quit.').pack()
        tk.Label(window, text=review.get('reason', '')).pack()
        canvas = tk.Canvas(window, width=dw, height=dh, highlightthickness=0)
        canvas.pack()
        photo = ImageTk.PhotoImage(image.resize((dw, dh)))
        canvas.create_image(0, 0, anchor='nw', image=photo)
        points, labels, marks = [], [], []

        def click(event, positive):
            if not (0 <= event.x < dw and 0 <= event.y < dh):
                return
            points.append([min(w-1, event.x*w/dw), min(h-1, event.y*h/dh)])
            labels.append(positive)
            marks.append(canvas.create_oval(event.x-4, event.y-4, event.x+4, event.y+4,
                                            fill='lime' if positive else 'red'))

        def undo(event=None):
            if marks:
                canvas.delete(marks.pop())
                points.pop(); labels.pop()

        def save(event=None):
            if 1 not in labels:
                return
            corrections[key] = {'frame_index': review['frame_index'], 'points_xy': points,
                                'labels': labels}
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(corrections, indent=2)+'\n')
            temporary.replace(output)
            window.destroy()

        def quit_viewer(event=None):
            stopped[0] = True
            window.destroy()

        canvas.bind('<Button-1>', lambda e: click(e, 1))
        canvas.bind('<Button-3>', lambda e: click(e, 0))
        window.bind('<Return>', save)
        window.bind('u', undo)
        window.bind('n', lambda e: window.destroy())
        window.bind('q', quit_viewer)
        window.protocol('WM_DELETE_WINDOW', quit_viewer)
        window.focus_force()
        root.wait_window(window)
        if stopped[0]:
            break
    root.destroy()
    print(f'Corrections saved to {output}')


if __name__ == '__main__':
    main()
