#!/usr/bin/env python3
"""Query terminal cell size via escape sequences and print shell exports.

Run BEFORE Textual starts — once Textual owns stdin, escape probes fail.
Usage in bash:  eval "$(python scripts/query_cell_size.py)"
"""

import os, sys, struct, select, termios, tty, fcntl


def _from_tiocgwinsz():
    """Try TIOCGWINSZ first (works locally, fails over SSH with xpix=0)."""
    try:
        buf = fcntl.ioctl(sys.__stdout__, termios.TIOCGWINSZ, b'\x00' * 8)
        rows, cols, xpix, ypix = struct.unpack('HHHH', buf)
        if rows > 0 and cols > 0 and xpix > 0 and ypix > 0:
            return round(xpix / cols), round(ypix / rows)
    except Exception:
        pass
    return None


def _from_escape_sequence():
    r"""Query via \e[14t — returns window size in pixels: ESC[4;height;widtht."""
    if not (sys.__stdin__ and sys.__stdin__.isatty()):
        return None

    fd = sys.__stdin__.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        os.write(sys.__stdout__.fileno(), b'\x1b[14t')

        resp = b''
        while select.select([fd], [], [], 0.5)[0]:
            ch = os.read(fd, 1)
            resp += ch
            if ch == b't':
                break

        text = resp.decode('ascii', errors='ignore')
        if '4;' in text:
            body = text.split('[', 1)[-1].rstrip('t')
            parts = body.split(';')
            if len(parts) >= 3:
                win_h, win_w = int(parts[1]), int(parts[2])
                buf = fcntl.ioctl(sys.__stdout__, termios.TIOCGWINSZ, b'\x00' * 8)
                rows, cols = struct.unpack('HH', buf[:4])
                if rows > 0 and cols > 0:
                    return round(win_w / cols), round(win_h / rows)
    except Exception:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return None


result = _from_tiocgwinsz() or _from_escape_sequence()
if result:
    cw, ch = result
    print(f'export TEXTUAL_CELL_WIDTH={cw}')
    print(f'export TEXTUAL_CELL_HEIGHT={ch}')
