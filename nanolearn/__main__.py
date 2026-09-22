"""Command line entry point.

Three ways in, in increasing order of nerdiness::

    python start.py             # start the app and open the browser
    python -m nanolearn         # the same thing
    python -m nanolearn doctor  # print what this computer can do, and stop
"""

from __future__ import annotations

import argparse
import sys

from . import APP_NAME, __version__, engine, store


def _doctor() -> int:
    print()
    print("  %s %s — what this computer can do" % (APP_NAME, __version__))
    print("  " + "-" * 56)
    print("  Python %s" % sys.version.split()[0])
    print("  Your files live in: %s" % store.data_dir())
    print()
    print("  Learning engines")
    print("    built-in (plain Python)   always available")
    if engine.sklearn_available():
        print("    full (scikit-learn %s)  available" % engine.sklearn_version())
    else:
        print("    full (scikit-learn)       not installed — run: python -m pip install scikit-learn")
    print()
    print("  Practice data that ships with the app")
    for key, value in store.SAMPLES.items():
        path = store.sample_path(key)
        print("    %-8s %s  (%s)" % (key, value["title"], path))
    print()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nanolearn",
        description="%s — %s" % (APP_NAME, "drop a spreadsheet, get an answer machine."),
        epilog="Run it with no arguments and a browser window opens.",
    )
    parser.add_argument("command", nargs="?", default="run", choices=["run", "doctor", "version"])
    parser.add_argument("--port", type=int, default=8761, help="which port to use (default 8761)")
    parser.add_argument("--host", default="127.0.0.1", help="address to listen on (default: this computer only)")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    parser.add_argument("--version", action="store_true", help="print the version and stop")
    args = parser.parse_args(argv)

    if args.version or args.command == "version":
        print("%s %s" % (APP_NAME, __version__))
        return 0
    if args.command == "doctor":
        return _doctor()

    from .server import create_app

    app = create_app()
    try:
        app.serve(host=args.host, port=args.port, open_browser=not args.no_browser)
    except OSError as exc:
        print("\n  Could not start on %s:%d (%s).\n  Try a different port: --port 8771\n"
              % (args.host, args.port, exc))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
