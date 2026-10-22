"""Foreground-only Windows hold-to-jog input; no keyboard hooks or key log.

Only six movement keys are sampled, only while the window bound by ``arm`` is
the current foreground window and belongs to this Python process. Discrete
7/6/8/9 commands remain the MuJoCo viewer callback's responsibility.

The Win32 API is injectable so tests never read the user's keyboard. Importing
this module does not load Win32 or sample any input. No background thread is
created. The main loop owns this object and calls ``poll``.
"""
import os


# Win32 virtual-key values, deliberately not GLFW's key values.
MOVEMENT_KEYS = (
    (0x26, 'X+', (1, 0, 0)),   # VK_UP
    (0x28, 'X-', (-1, 0, 0)),  # VK_DOWN
    (0x25, 'Y+', (0, 1, 0)),   # VK_LEFT
    (0x27, 'Y-', (0, -1, 0)),  # VK_RIGHT
    (0x21, 'Z+', (0, 0, 1)),   # VK_PRIOR / PageUp
    (0x22, 'Z-', (0, 0, -1)),  # VK_NEXT / PageDown
)


class _Win32:
    """The three read-only User32 calls needed for foreground-gated polling."""
    def __init__(self):
        if os.name != 'nt':
            raise RuntimeError('Continuous jog input requires Windows')
        import ctypes
        from ctypes import wintypes
        self.ctypes = ctypes
        self.wintypes = wintypes
        self.user32 = ctypes.WinDLL('user32', use_last_error=True)
        self.user32.GetForegroundWindow.argtypes = []
        self.user32.GetForegroundWindow.restype = wintypes.HWND
        self.user32.GetWindowThreadProcessId.argtypes = [
            wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        self.user32.GetWindowThreadProcessId.restype = wintypes.DWORD
        self.user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
        self.user32.GetAsyncKeyState.restype = wintypes.SHORT

    def foreground_window(self):
        return int(self.user32.GetForegroundWindow() or 0)

    def window_pid(self, hwnd):
        process = self.wintypes.DWORD(0)
        thread = self.user32.GetWindowThreadProcessId(
            self.wintypes.HWND(hwnd), self.ctypes.byref(process))
        return int(process.value) if thread else 0

    def key_state(self, virtual_key):
        return int(self.user32.GetAsyncKeyState(virtual_key))


class WindowsJogInput:
    """Single-owner poller bound to the scene window by a user start command.

``arm()`` binds the current foreground window only if its PID is this process.
``poll()`` returns a dict with ``active``, ``reason``, ``command`` and ``vector``.
An absent command always means zero requested jog velocity. After startup or
focus loss, all six direction keys must first be released in the scene window.
Simultaneous keys are rejected; no priority and no accumulated queue exist.

The caller must re-poll after any expensive planning call before committing a
new segment: input observations are snapshots, not authority to move later.
"""
    def __init__(self, api=None, pid=None):
        self.api = _Win32() if api is None else api
        self.pid = os.getpid() if pid is None else int(pid)
        self.hwnd = None
        self._need_release = True

    @staticmethod
    def _result(active, reason, command=None, vector=(0, 0, 0)):
        return dict(active=bool(active), reason=reason, command=command,
                    vector=tuple(vector))

    @property
    def armed(self):
        return self.hwnd is not None

    def disarm(self):
        self.hwnd = None
        self._need_release = True
        return self._result(False, 'unarmed')

    def arm(self):
        self.disarm()
        hwnd = self.api.foreground_window()
        if not hwnd or self.api.window_pid(hwnd) != self.pid:
            return self._result(False, 'start_window_not_foreground_process')
        self.hwnd = hwnd
        # No direction polling in arm: the next poll must observe neutral.
        return self._result(False, 'release_directions_to_enable')

    def _focused(self):
        hwnd = self.api.foreground_window()
        return (self.hwnd is not None and hwnd == self.hwnd and
                self.api.window_pid(hwnd) == self.pid)

    def poll(self):
        if not self.armed:
            return self._result(False, 'unarmed')
        if not self._focused():
            self._need_release = True
            return self._result(False, 'scene_not_focused')
        held = []
        for key, command, vector in MOVEMENT_KEYS:
            # Ignore the unreliable "pressed since previous call" low bit.
            if self.api.key_state(key) & 0x8000:
                held.append((command, vector))
        # Focus can change between the preceding check and the six reads.
        if not self._focused():
            self._need_release = True
            return self._result(False, 'scene_focus_changed_during_poll')
        if self._need_release:
            if held:
                return self._result(False, 'release_directions_to_enable')
            self._need_release = False
            return self._result(True, 'neutral')
        if len(held) > 1:
            return self._result(True, 'multiple_directions_no_motion')
        if not held:
            return self._result(True, 'neutral')
        command, vector = held[0]
        return self._result(True, 'held', command, vector)
