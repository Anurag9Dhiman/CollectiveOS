"""
Local filesystem connector — list directories, read and write files on this Mac.

All paths are sandboxed to the user's home directory. A small set of sensitive
subdirectories (.ssh, .gnupg, .aws, etc.) are blocked even within home.
Binary files are detected and refused on read.

Read tools : list_directory, read_local_file  — call freely.
Write tool  : write_local_file                — confirm with user first.
"""

import mimetypes
import os
import stat

_HOME = os.path.expanduser("~")
_MAX_READ_BYTES = 50_000  # 50 KB — keeps token cost reasonable

# Sensitive dirs blocked even inside home
_BLOCKED = {
    ".ssh", ".gnupg", ".aws", ".kube",
    "Library/Keychains", "Library/Cookies",
    "Library/Saved Application State",
}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _resolve(path: str) -> tuple[str, str | None]:
    """
    Expand and normalise *path*, restrict to home directory.
    Returns (abs_path, error_string_or_None).
    """
    expanded = os.path.expanduser(path.strip() or "~")

    # Treat bare relative paths as relative to home
    if not os.path.isabs(expanded):
        expanded = os.path.join(_HOME, expanded)

    real = os.path.realpath(expanded)

    if not real.startswith(_HOME):
        return "", f"Access denied: path must be inside your home directory ({_HOME})."

    # Check blocked subdirs
    rel = os.path.relpath(real, _HOME)
    for blocked in _BLOCKED:
        if rel == blocked or rel.startswith(blocked + os.sep):
            return "", f"Access denied: {blocked} is a protected directory."

    return real, None


def _size_str(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


# ---------------------------------------------------------------------------
# Read tools
# ---------------------------------------------------------------------------

def list_directory(path: str = "~", show_hidden: bool = False) -> str:
    """
    List the contents of a directory on this Mac.

    - path:        Directory path. Defaults to the home directory (~).
                   Accepts ~ and relative paths (resolved from home).
    - show_hidden: If true, include dotfiles and hidden entries. Defaults to false.

    Returns each entry on its own line with type (dir/file), size, and
    last-modified date.
    """
    abs_path, err = _resolve(path)
    if err:
        return err

    if not os.path.exists(abs_path):
        return f"Path does not exist: {abs_path}"
    if not os.path.isdir(abs_path):
        return f"Not a directory: {abs_path}"

    try:
        entries = os.scandir(abs_path)
    except PermissionError:
        return f"Permission denied reading: {abs_path}"

    lines = [f"Contents of {abs_path}:\n"]
    dirs, files = [], []

    for entry in sorted(entries, key=lambda e: e.name.lower()):
        if not show_hidden and entry.name.startswith("."):
            continue
        try:
            st = entry.stat(follow_symlinks=False)
        except OSError:
            continue

        import datetime
        mtime = datetime.datetime.fromtimestamp(st.st_mtime).strftime("%b %-d %Y")

        if entry.is_dir(follow_symlinks=False):
            dirs.append(f"  📁  {entry.name}/   ({mtime})")
        else:
            size = _size_str(st.st_size)
            files.append(f"  📄  {entry.name}   {size}   ({mtime})")

    lines += dirs + files

    if len(lines) == 1:
        lines.append("  (empty directory)")

    return "\n".join(lines)


def read_local_file(path: str) -> str:
    """
    Read the text content of a file on this Mac.

    - path: File path. Accepts ~ and relative paths (resolved from home).

    Refuses binary files. Caps output at 50 KB to avoid excessive token usage.
    """
    abs_path, err = _resolve(path)
    if err:
        return err

    if not os.path.exists(abs_path):
        return f"File not found: {abs_path}"
    if os.path.isdir(abs_path):
        return f"That is a directory, not a file. Use list_directory instead."

    # Check file size before reading
    size = os.path.getsize(abs_path)
    if size > 10 * 1024 * 1024:  # 10 MB hard limit
        return f"File too large to read ({_size_str(size)}). Max is 10 MB."

    # Detect likely binary files via mime type
    mime, _ = mimetypes.guess_type(abs_path)
    if mime and not (mime.startswith("text/") or mime in {
        "application/json", "application/xml", "application/javascript",
        "application/x-sh", "application/x-yaml",
    }):
        return (
            f"File appears to be binary ({mime}). "
            "Only text files can be read."
        )

    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
            content = fh.read(_MAX_READ_BYTES)
            truncated = fh.read(1)  # check if more remains
    except PermissionError:
        return f"Permission denied reading: {abs_path}"
    except Exception as e:
        return f"Error reading file: {e}"

    header = f"--- {abs_path} ({_size_str(size)}) ---\n"
    if truncated:
        return header + content + f"\n\n[... truncated at 50 KB — file is {_size_str(size)} total]"
    return header + content


# ---------------------------------------------------------------------------
# Write tool
# ---------------------------------------------------------------------------

def write_local_file(path: str, content: str) -> str:
    """
    Write (create or overwrite) a file on this Mac with the given text content.

    Only call this after the user has confirmed the file path and content.
    Creates any missing parent directories automatically.

    - path:    File path to write. Must be inside the home directory.
    - content: Full text content to write to the file.
    """
    abs_path, err = _resolve(path)
    if err:
        return err

    if os.path.isdir(abs_path):
        return f"Cannot write: {abs_path} is a directory."

    parent = os.path.dirname(abs_path)
    try:
        os.makedirs(parent, exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as fh:
            fh.write(content)
    except PermissionError:
        return f"Permission denied writing to: {abs_path}"
    except Exception as e:
        return f"Error writing file: {e}"

    size = _size_str(len(content.encode("utf-8")))
    existed = "Overwrote" if os.path.exists(abs_path) else "Created"
    return f"{existed} {abs_path} ({size})."
