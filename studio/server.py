"""Local HTTP server for the pattern studio.

Serves the editor UI and a small JSON API.  The browser holds the working
copy of the project and sends it with each request; Save persists it to the
project file, Export writes a pattern folder into the repo.
"""

import json
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from render import _write_svg
from . import model, evaluator, codegen

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"


def _jsonable(obj):
    """Recursively convert numpy arrays/scalars for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _eval_payload(project, values):
    """Run validation + evaluation and shape a canvas-friendly payload."""
    issues = model.validate(project)
    result = evaluator.evaluate(project, values)
    pieces = {}
    for piece_id, pc in result["pieces"].items():
        pieces[piece_id] = {
            "skipped": pc["skipped"],
            "points": {n: _jsonable(p) for n, p in pc["points"].items()},
            "point_order": pc["point_order"],
            "errors": pc["errors"],
            "segments": [
                {"index": c["index"], "type": c["type"],
                 "samples": _jsonable(c["samples"])}
                for c in pc["canvas_segments"]
            ],
            "construction_lines": _jsonable(pc["construction_lines"]),
            "unclipped_construction_lines":
                _jsonable(pc["unclipped_construction_lines"]),
            "dart_lines": _jsonable(pc["dart_lines"]),
            "annotations": [
                {"text": t, "pos": _jsonable(p)} for t, p in pc["annotations"]
            ],
        }
    return {
        "issues": issues,
        "scalars": _jsonable(result["scalars"]),
        "errors": result["errors"],
        "pieces": pieces,
    }


def _render_payload(project, values, seam_allowance, white_fill):
    """Render evaluated pieces through the real engine → SVG strings."""
    result = evaluator.evaluate(project, values)
    svgs, errors = {}, []
    for piece in project.get("pieces", []):
        pc = result["pieces"][piece["id"]]
        if pc["skipped"]:
            continue
        if pc["errors"]:
            errors.append({"piece": piece["id"],
                           "messages": [e["message"] for e in pc["errors"]]})
            continue
        if not pc["outline"]:
            errors.append({"piece": piece["id"],
                           "messages": ["piece has no outline segments"]})
            continue
        try:
            args = evaluator.render_args(piece, pc,
                                         seam_allowance=seam_allowance,
                                         white_fill=white_fill)
            svg, w, h = _write_svg(None, args.pop("outline"), **args)
            svgs[piece["id"]] = {"svg": svg, "w": w, "h": h}
        except Exception as e:
            errors.append({"piece": piece["id"], "messages": [str(e)]})
    return {"svgs": svgs, "errors": errors,
            "top_errors": result["errors"]}


class StudioHandler(BaseHTTPRequestHandler):
    server_version = "PatternStudio/1.0"
    project_path = None   # set by run()

    # ── plumbing ──────────────────────────────────────────────────────────

    def _send(self, code, body, content_type="application/json"):
        data = body if isinstance(body, bytes) else \
            json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    def log_message(self, fmt, *args):
        pass   # keep the terminal quiet

    # ── routes ────────────────────────────────────────────────────────────

    def do_GET(self):
        try:
            if self.path in ("/", "/index.html"):
                html = (STATIC_DIR / "index.html").read_bytes()
                self._send(200, html, "text/html; charset=utf-8")
            elif self.path == "/api/project":
                path = Path(self.project_path)
                if path.exists():
                    project = json.loads(path.read_text())
                else:
                    stem = path.stem.replace(".studio", "")
                    project = model.new_project(stem if stem.isidentifier()
                                                else "new_pattern")
                self._send(200, {"path": str(path), "project": project,
                                 "exists": path.exists()})
            else:
                self._send(404, {"error": f"not found: {self.path}"})
        except Exception as e:
            traceback.print_exc()
            self._send(500, {"error": str(e)})

    def do_POST(self):
        try:
            body = self._body()
            if self.path == "/api/project":
                path = Path(self.project_path)
                path.write_text(
                    json.dumps(body["project"], indent=2, ensure_ascii=False)
                    + "\n", encoding="utf-8")
                self._send(200, {"saved": str(path)})
            elif self.path == "/api/eval":
                self._send(200, _eval_payload(body["project"],
                                              body.get("values", {})))
            elif self.path == "/api/render":
                self._send(200, _render_payload(
                    body["project"], body.get("values", {}),
                    float(body.get("seam_allowance", 0.75)),
                    bool(body.get("white_fill", False))))
            elif self.path == "/api/export":
                issues = model.validate(body["project"])
                if issues:
                    self._send(400, {"error": "project has validation issues",
                                     "issues": issues})
                    return
                result = codegen.export(body["project"], REPO_ROOT,
                                        body.get("confirm_overwrite", False))
                self._send(409 if "needs_confirm" in result else 200, result)
            else:
                self._send(404, {"error": f"not found: {self.path}"})
        except Exception as e:
            traceback.print_exc()
            self._send(500, {"error": str(e)})


def run(project_path, port=8765, open_browser=True):
    StudioHandler.project_path = str(project_path)
    server = ThreadingHTTPServer(("127.0.0.1", port), StudioHandler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Pattern Studio · {project_path}")
    print(f"  → {url}   (ctrl-c to stop)")
    if open_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
