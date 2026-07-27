"""
Apple-native connector — Contacts, Reminders, Notes, and Clipboard on Mac.

All tools use osascript (AppleScript) except Clipboard, which uses the
built-in pbpaste / pbcopy commands.

Permissions (macOS will prompt once on first use):
  Contacts  — System Settings → Privacy → Contacts
  Reminders — System Settings → Privacy → Reminders
  Notes     — Automation permission for the calling process

Read  tools : contacts_search, reminders_list, notes_list, notes_read, clipboard_read
Write tools : reminders_add, reminders_complete, notes_append, notes_create, clipboard_write
"""

import datetime
import platform
import re
import subprocess


def _require_macos() -> str | None:
    if platform.system() != "Darwin":
        return "Apple-native tools only work on macOS."
    return None


def _run(script: str, timeout: int = 15) -> tuple[str, str]:
    r = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True, text=True, timeout=timeout,
    )
    return r.stdout.strip(), r.stderr.strip()


def _strip_html(html: str) -> str:
    """Remove HTML tags and decode basic entities."""
    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    text = text.replace("&lt;", "<").replace("&gt;", ">")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ---------------------------------------------------------------------------
# Contacts
# ---------------------------------------------------------------------------

def contacts_search(name: str) -> str:
    """
    Search Apple Contacts for people matching a name (full or partial).
    Returns their phone numbers and email addresses.
    macOS will prompt for Contacts access on first use.
    """
    err = _require_macos()
    if err:
        return err

    safe = name.replace('"', "")
    script = (
        'tell application "Contacts"\n'
        f'  set matches to every person whose name contains "{safe}"\n'
        '  if (count of matches) is 0 then return "No contacts found."\n'
        '  set out to ""\n'
        '  repeat with p in matches\n'
        '    set out to out & name of p & "\\n"\n'
        '    repeat with ph in phones of p\n'
        '      set out to out & "  Phone: " & value of ph & "\\n"\n'
        '    end repeat\n'
        '    repeat with em in emails of p\n'
        '      set out to out & "  Email: " & value of em & "\\n"\n'
        '    end repeat\n'
        '    set out to out & "\\n"\n'
        '  end repeat\n'
        '  return out\n'
        'end tell'
    )
    out, err_msg = _run(script)
    if err_msg and not out:
        return f"Contacts error: {err_msg}"
    return out or "No contacts found."


# ---------------------------------------------------------------------------
# Reminders
# ---------------------------------------------------------------------------

def reminders_list(list_name: str = "", due_today: bool = False) -> str:
    """
    List incomplete reminders from the Reminders app.

    - list_name: Name of a specific Reminders list (e.g. 'Groceries').
                 Leave blank to show all lists.
    - due_today: If true, only return reminders due today or overdue.
    """
    err = _require_macos()
    if err:
        return err

    if list_name:
        safe = list_name.replace('"', "")
        source = f'reminders of list "{safe}"'
    else:
        source = "reminders"

    script = (
        'tell application "Reminders"\n'
        f'  set rems to {source}\n'
        '  set out to ""\n'
        '  repeat with r in rems\n'
        '    if completed of r is false then\n'
        '      set out to out & name of r\n'
        '      try\n'
        '        set d to due date of r\n'
        '        set out to out & " (due: " & (d as string) & ")"\n'
        '      end try\n'
        '      set out to out & "\\n"\n'
        '    end if\n'
        '  end repeat\n'
        '  if out is "" then return "No incomplete reminders found."\n'
        '  return out\n'
        'end tell'
    )
    out, err_msg = _run(script)
    if err_msg and not out:
        return f"Reminders error: {err_msg}"
    return out or "No incomplete reminders found."


def reminders_add(title: str, due_date: str = "", list_name: str = "") -> str:
    """
    Add a new reminder to the Reminders app.
    Always confirm with the user before calling.

    - title:     Reminder text.
    - due_date:  Optional due date in YYYY-MM-DD format, e.g. '2026-07-28'.
    - list_name: Reminders list to add to. Defaults to the default list.
    """
    err = _require_macos()
    if err:
        return err

    safe_title = title.replace('"', "'")
    safe_list  = list_name.replace('"', "") if list_name else ""

    props = f'{{name: "{safe_title}"}}'

    date_line = ""
    if due_date:
        try:
            dt = datetime.datetime.strptime(due_date, "%Y-%m-%d")
            as_date = dt.strftime("%B %-d, %Y")
            date_line = f'set due date of newR to date "{as_date}"\n'
        except ValueError:
            pass

    if safe_list:
        target = f'list "{safe_list}"'
    else:
        target = "default list"

    script = (
        'tell application "Reminders"\n'
        f'  set newR to make new reminder at end of {target} with properties {props}\n'
        + date_line +
        '  return "Reminder added: " & name of newR\n'
        'end tell'
    )
    out, err_msg = _run(script)
    if err_msg and not out:
        return f"Reminders error: {err_msg}"
    return out or f"Reminder '{title}' added."


def reminders_complete(title: str, list_name: str = "") -> str:
    """
    Mark a reminder as completed by its title.
    Always confirm with the user before calling.

    - title:     Title of the reminder to complete (exact or partial match).
    - list_name: Narrow the search to a specific list (optional).
    """
    err = _require_macos()
    if err:
        return err

    safe_title = title.replace('"', "'")
    safe_list  = list_name.replace('"', "") if list_name else ""

    if safe_list:
        source = f'reminders of list "{safe_list}"'
    else:
        source = "reminders"

    script = (
        'tell application "Reminders"\n'
        f'  set matches to (every reminder of ({source}) whose name contains "{safe_title}" and completed is false)\n'
        '  if (count of matches) = 0 then return "No matching incomplete reminder found."\n'
        '  set completed of first item of matches to true\n'
        '  return "Completed: " & name of first item of matches\n'
        'end tell'
    )
    out, err_msg = _run(script)
    if err_msg and not out:
        return f"Reminders error: {err_msg}"
    return out or "Reminder completed."


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------

def notes_list(folder: str = "") -> str:
    """
    List note titles in the Notes app.
    - folder: Narrow to a specific folder name (optional).
    """
    err = _require_macos()
    if err:
        return err

    if folder:
        safe = folder.replace('"', "")
        source = f'notes of folder "{safe}"'
    else:
        source = "notes"

    script = (
        'tell application "Notes"\n'
        f'  set ns to {source}\n'
        '  set out to ""\n'
        '  repeat with n in ns\n'
        '    set out to out & name of n & "\\n"\n'
        '  end repeat\n'
        '  if out is "" then return "No notes found."\n'
        '  return out\n'
        'end tell'
    )
    out, err_msg = _run(script)
    if err_msg and not out:
        return f"Notes error: {err_msg}"
    return out or "No notes found."


def notes_read(title: str) -> str:
    """
    Read the content of a note by title (partial match).
    Returns plain text with HTML stripped.
    """
    err = _require_macos()
    if err:
        return err

    safe = title.replace('"', "'")
    script = (
        'tell application "Notes"\n'
        f'  set matches to (every note whose name contains "{safe}")\n'
        '  if (count of matches) = 0 then return "No note found matching that title."\n'
        '  set n to first item of matches\n'
        '  return body of n\n'
        'end tell'
    )
    out, err_msg = _run(script)
    if err_msg and not out:
        return f"Notes error: {err_msg}"
    if not out or out == "No note found matching that title.":
        return out or "No note found."
    return f"Note: {title}\n\n{_strip_html(out)}"


def notes_create(title: str, body: str, folder: str = "") -> str:
    """
    Create a new note in the Notes app.
    Always confirm title and content with the user before calling.

    - title:  Note title.
    - body:   Note body text.
    - folder: Folder to create the note in (optional, defaults to root).
    """
    err = _require_macos()
    if err:
        return err

    safe_title  = title.replace('"', "'")
    safe_body   = body.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "<br>")
    safe_folder = folder.replace('"', "") if folder else ""

    props = f'{{name: "{safe_title}", body: "{safe_body}"}}'

    if safe_folder:
        target = f'folder "{safe_folder}"'
    else:
        target = "default account"

    script = (
        'tell application "Notes"\n'
        f'  make new note at {target} with properties {props}\n'
        f'  return "Note created: {safe_title}"\n'
        'end tell'
    )
    out, err_msg = _run(script)
    if err_msg and not out:
        return f"Notes error: {err_msg}"
    return out or f"Note '{title}' created."


def notes_append(title: str, text: str) -> str:
    """
    Append text to an existing note (matched by title).
    Always confirm which note and what text with the user before calling.

    - title: Title of the note to append to (partial match).
    - text:  Text to add at the end of the note.
    """
    err = _require_macos()
    if err:
        return err

    safe_title = title.replace('"', "'")
    safe_text  = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "<br>")

    script = (
        'tell application "Notes"\n'
        f'  set matches to (every note whose name contains "{safe_title}")\n'
        '  if (count of matches) = 0 then return "No note found matching that title."\n'
        '  set n to first item of matches\n'
        f'  set body of n to (body of n) & "<br>{safe_text}"\n'
        '  return "Appended to: " & name of n\n'
        'end tell'
    )
    out, err_msg = _run(script)
    if err_msg and not out:
        return f"Notes error: {err_msg}"
    return out or f"Appended to '{title}'."


# ---------------------------------------------------------------------------
# Clipboard
# ---------------------------------------------------------------------------

def clipboard_read() -> str:
    """
    Read the current contents of the Mac system clipboard.
    Returns plain text only — ignores images or other binary content.
    """
    err = _require_macos()
    if err:
        return err

    result = subprocess.run(
        ["pbpaste"], capture_output=True, text=True, timeout=5,
    )
    content = result.stdout
    if not content:
        return "Clipboard is empty (or contains non-text content)."

    if len(content) > 4000:
        return content[:4000] + f"\n\n[... truncated — clipboard has {len(content)} chars total]"
    return f"Clipboard contents:\n{content}"


def clipboard_write(text: str) -> str:
    """
    Write text to the Mac system clipboard, replacing its current contents.
    Always confirm the text with the user before calling.

    - text: The text to copy to the clipboard.
    """
    err = _require_macos()
    if err:
        return err

    subprocess.run(
        ["pbcopy"], input=text, text=True, timeout=5,
    )
    preview = text[:80] + ("…" if len(text) > 80 else "")
    return f"Copied to clipboard ({len(text)} chars): {preview!r}"
