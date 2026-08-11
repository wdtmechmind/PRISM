from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


class MechHandV3ProtocolError(RuntimeError):
    """Raised when daemon responses do not match expected protocol frames."""


class MechHandV3CommandError(RuntimeError):
    """Raised when daemon returns ERR:* for a command."""


@dataclass
class ParsedResponse:
    command: str
    result: str

    @property
    def ok(self) -> bool:
        return not self.result.startswith("ERR:")

    @property
    def error_reason(self) -> Optional[str]:
        if not self.result.startswith("ERR:"):
            return None
        return self.result[4:]


def _as_int5(values: Sequence[int]) -> List[int]:
    if len(values) != 5:
        raise ValueError("expected 5 integer values, got %d" % len(values))
    return [int(v) for v in values]


def degrees_to_tenths(value_deg: float, clamp: bool = True) -> int:
    raw = int(round(float(value_deg) * 10.0))
    if not clamp:
        return raw
    return max(-900, min(900, raw))


class MechHandV3DaemonClient:
    """TCP client for MechHand V3 daemon frame protocol.

    Protocol:
    - request : @<CMD><VALUE>&
    - response: @_<CMD><RESULT>&
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8080, timeout_s: float = 2.0):
        self.host = str(host)
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self.sock: Optional[socket.socket] = None
        self._rx_buffer = b""

    def connect(self) -> None:
        if self.sock is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout_s)
        sock.connect((self.host, self.port))
        self.sock = sock
        self._rx_buffer = b""

    def close(self) -> None:
        if self.sock is None:
            return
        try:
            self.sock.close()
        finally:
            self.sock = None
            self._rx_buffer = b""

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def _read_one_frame(self) -> str:
        if self.sock is None:
            raise RuntimeError("client is not connected")

        while True:
            end = self._rx_buffer.find(b"&")
            if end >= 0:
                frame_bytes = self._rx_buffer[: end + 1]
                self._rx_buffer = self._rx_buffer[end + 1 :]
                text = frame_bytes.decode("utf-8", errors="replace").strip()
                if text:
                    return text

            chunk = self.sock.recv(1024)
            if not chunk:
                raise ConnectionError("daemon closed connection")
            self._rx_buffer += chunk

    @staticmethod
    def parse_response(frame: str, expected_cmd: Optional[str] = None) -> ParsedResponse:
        text = str(frame).strip()
        if not text.startswith("@_") or not text.endswith("&"):
            raise MechHandV3ProtocolError("malformed response frame: %r" % text)

        body = text[2:-1]
        left = body.find("<")
        right = body.rfind(">")
        if left <= 0 or right < left:
            raise MechHandV3ProtocolError("invalid response body: %r" % body)

        cmd = body[:left]
        result = body[left + 1 : right]
        if expected_cmd is not None and cmd != expected_cmd:
            raise MechHandV3ProtocolError(
                "response command mismatch: expected %s got %s in %r" % (expected_cmd, cmd, text)
            )
        return ParsedResponse(command=cmd, result=result)

    def send(self, cmd: str, value: str = "") -> ParsedResponse:
        if self.sock is None:
            raise RuntimeError("client is not connected")

        clean_cmd = str(cmd).strip().upper()
        if not clean_cmd.isalpha():
            raise ValueError("invalid command: %r" % cmd)

        frame = "@%s<%s>&" % (clean_cmd, str(value))
        self.sock.sendall(frame.encode("utf-8"))
        parsed = self.parse_response(self._read_one_frame(), expected_cmd=clean_cmd)
        if not parsed.ok:
            raise MechHandV3CommandError("%s failed: %s" % (clean_cmd, parsed.result))
        return parsed

    def query_connected(self) -> bool:
        parsed = self.send("CC", "")
        return parsed.result == "1"

    def query_device_id(self) -> int:
        parsed = self.send("ID", "")
        try:
            return int(parsed.result)
        except ValueError as exc:
            raise MechHandV3ProtocolError("ID response is not integer: %r" % parsed.result) from exc

    def list_scripts(self) -> List[str]:
        parsed = self.send("GS", "")
        if not parsed.result:
            return []
        return [token.strip() for token in parsed.result.split(",") if token.strip()]

    def home_calibrate(self) -> None:
        self.send("HC", "")

    def set_speed(self, values: Sequence[int]) -> None:
        nums = _as_int5(values)
        self.send("SV", ",".join(str(v) for v in nums))

    def set_force(self, values: Sequence[int]) -> None:
        nums = _as_int5(values)
        self.send("SF", ",".join(str(v) for v in nums))

    def set_running_time(self, values: Sequence[int]) -> None:
        nums = _as_int5(values)
        self.send("SRT", ",".join(str(v) for v in nums))

    def run_operation_gesture(self, index: int) -> None:
        self.send("ROG", str(int(index)))

    def run_gesture(self, index: int) -> None:
        self.send("RNG", str(int(index)))

    def upload_script(self, name: str) -> bool:
        parsed = self.send("US", str(name).strip())
        return parsed.result == "1"

    def run_uploaded_script(self) -> None:
        self.send("RS", "")

    def move_rm_tenths(self, j1: int, j2: int, j3: int, j4: int, j5: int) -> None:
        values = [int(j1), int(j2), int(j3), int(j4), int(j5)]
        self.send("RM", ",".join(str(v) for v in values))

    def move_rm_degrees(self, j1: float, j2: float, j3: float, j4: float, j5: float) -> None:
        values = [degrees_to_tenths(v, clamp=True) for v in [j1, j2, j3, j4, j5]]
        self.move_rm_tenths(*values)
