"""Sound Effects Module.

Provides asynchronous, non-blocking audio feedback using built-in Windows winsound.
Gives tactile acoustic confirmation for button presses, calculations, and drawing actions.
"""

import threading

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


def _safe_beep(frequency: int, duration_ms: int) -> None:
    """Emits a tone via winsound.Beep safely."""
    if not HAS_WINSOUND:
        return
    try:
        winsound.Beep(frequency, duration_ms)
    except Exception:
        pass


def play_button_click() -> None:
    """Plays a crisp, high-pitched mechanical click for button presses."""
    threading.Thread(target=_safe_beep, args=(1800, 22), daemon=True).start()


def play_draw_start() -> None:
    """Plays a subtle, pleasant chirp when air drawing begins."""
    threading.Thread(target=_safe_beep, args=(1250, 25), daemon=True).start()


def play_calculate_chime() -> None:
    """Plays an ascending dual-tone chime when calculation succeeds."""
    def _chime():
        _safe_beep(1350, 25)
        _safe_beep(1850, 35)
    threading.Thread(target=_chime, daemon=True).start()


def play_clear_tone() -> None:
    """Plays a low sweep tone when clearing canvas or calculator."""
    threading.Thread(target=_safe_beep, args=(600, 35), daemon=True).start()


def play_history_recall() -> None:
    """Plays a quick tone when recalling a previous calculation from history tape."""
    def _recall():
        _safe_beep(1600, 20)
        _safe_beep(2100, 25)
    threading.Thread(target=_recall, daemon=True).start()
