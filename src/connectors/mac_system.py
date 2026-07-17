"""
Mac system connector — local system info and automation.

No API keys or external services. All calls use macOS built-in tools:
  system_profiler, pmset, df, vm_stat, uptime, osascript, open.

Only works on macOS. Functions return an error string on other platforms
so the agent can tell the user rather than crashing.
"""

import platform
import subprocess
import shlex


def _run(cmd: str, timeout: int = 10) -> str:
    """Run a shell command and return its stdout, or an error string."""
    try:
        result = subprocess.run(
            shlex.split(cmd),
            capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return f"Command timed out: {cmd}"
    except Exception as e:
        return f"Error: {e}"


def _require_macos() -> str | None:
    if platform.system() != "Darwin":
        return "This tool only works on macOS."
    return None


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def get_system_info() -> str:
    """
    Return a snapshot of the Mac's current system status:
    battery level and charging state, disk usage, memory pressure,
    CPU model, macOS version, and uptime.
    """
    err = _require_macos()
    if err:
        return err

    lines = []

    # macOS version
    sw = _run("sw_vers -productVersion")
    if sw:
        lines.append(f"macOS: {sw}")

    # Battery
    pmset = _run("pmset -g batt")
    for line in pmset.splitlines():
        if "%" in line:
            lines.append(f"Battery: {line.strip()}")
            break

    # Uptime
    uptime_raw = _run("uptime")
    if uptime_raw:
        # "up 2 days, 3:42" — extract the "up …" portion
        if "up" in uptime_raw:
            up_part = uptime_raw.split("up")[1].split(",  load")[0].strip()
            lines.append(f"Uptime: {up_part}")

    # Disk usage for /
    df_out = _run("df -h /")
    for line in df_out.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 5:
            lines.append(f"Disk (/): {parts[2]} used of {parts[1]}  ({parts[4]} full)")
        break

    # Memory pressure (vm_stat gives pages; convert to rough MB)
    vm = _run("vm_stat")
    free_pages = 0
    for vline in vm.splitlines():
        if "Pages free" in vline:
            try:
                free_pages = int(vline.split(":")[1].strip().rstrip("."))
            except ValueError:
                pass
            break
    if free_pages:
        free_mb = (free_pages * 4096) // (1024 * 1024)
        lines.append(f"Free memory: ~{free_mb} MB")

    # CPU model (short form)
    cpu = _run("sysctl -n machdep.cpu.brand_string")
    if cpu:
        lines.append(f"CPU: {cpu}")

    return "\n".join(lines) if lines else "Could not read system info."


def get_wifi_info() -> str:
    """Return the current Wi-Fi network name and local IP address."""
    err = _require_macos()
    if err:
        return err

    # SSID via airport
    ssid = _run(
        "/System/Library/PrivateFrameworks/Apple80211.framework"
        "/Versions/Current/Resources/airport -I"
    )
    network = "unknown"
    for line in ssid.splitlines():
        if " SSID:" in line:
            network = line.split("SSID:")[1].strip()
            break

    # Local IP via ipconfig
    ip = _run("ipconfig getifaddr en0") or _run("ipconfig getifaddr en1") or "unavailable"

    return f"Wi-Fi: {network}\nLocal IP: {ip}"


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def show_notification(title: str, body: str) -> str:
    """
    Show a macOS notification banner.

    title: notification title
    body:  notification message text
    """
    err = _require_macos()
    if err:
        return err

    # Sanitise inputs to prevent osascript injection
    safe_title = title.replace('"', "'")
    safe_body  = body.replace('"', "'")
    script = f'display notification "{safe_body}" with title "{safe_title}"'
    _run(f"osascript -e {shlex.quote(script)}")
    return f'Notification sent: "{title}"'


def open_application(name: str) -> str:
    """
    Open a macOS application by name, e.g. 'Safari', 'Spotify', 'VS Code'.
    Uses the system `open -a` command.
    """
    err = _require_macos()
    if err:
        return err

    result = subprocess.run(
        ["open", "-a", name],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return f"Could not open '{name}': {result.stderr.strip()}"
    return f"Opened {name}."


def set_system_volume(level: int) -> str:
    """
    Set the macOS system output volume (0–100).
    This is the system audio level, independent of Spotify's own volume.
    """
    err = _require_macos()
    if err:
        return err

    level = max(0, min(100, level))
    script = f"set volume output volume {level}"
    _run(f"osascript -e {shlex.quote(script)}")
    return f"System volume set to {level}%."
