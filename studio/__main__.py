"""Launch the pattern studio:  python -m studio [project.studio.json]"""

import argparse

from .server import run

parser = argparse.ArgumentParser(
    prog="python -m studio",
    description="Local GUI for authoring patterns.")
parser.add_argument("project", nargs="?", default="new_pattern.studio.json",
                    help="project file to open or create "
                         "(default: new_pattern.studio.json)")
parser.add_argument("--port", type=int, default=8765)
parser.add_argument("--no-browser", action="store_true",
                    help="don't open the browser automatically")
args = parser.parse_args()

run(args.project, port=args.port, open_browser=not args.no_browser)
