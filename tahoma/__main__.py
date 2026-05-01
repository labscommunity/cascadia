"""Allow `python -m tahoma worker ...`."""

from tahoma.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
