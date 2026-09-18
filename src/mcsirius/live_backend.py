"""Standalone live hardware access for the FLAVIA source section."""

from __future__ import annotations

from dataclasses import dataclass, field
import queue
import re
import socket
import threading
import time
from typing import Any
from urllib.request import Request, urlopen


MQTT_HOST = "192.168.0.20"
MQTT_PORT = 1883

CUP_HOST = "192.168.0.19"

KEITHLEY_HOST = "192.168.0.2"
KEITHLEY_PORT = 100

MAGNET_HOST = "192.168.0.5"
MAGNET_PORT = 8462


SOURCE_COMMAND_TOPICS = {
    "cs/sputter/set_u_v":
        "cs/sputter/cmd/set_u_v",
    "cs/extraction/set_u_v":
        "cs/extraction/cmd/set_u_v",
    "cs/einzellens/set_u_v":
        "cs/einzellens/cmd/set_u_v",
}


@dataclass
class LiveChannel:
    value: Any = None
    timestamp: float = field(
        default_factory=time.time
    )
    quality: str = "unknown"


class LiveModel:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._channels: dict[
            str,
            LiveChannel,
        ] = {}

    def update(
        self,
        name: str,
        value: Any,
        *,
        quality: str = "good",
    ) -> None:
        with self._lock:
            channel = self._channels.get(name)

            if channel is None:
                self._channels[name] = LiveChannel(
                    value=value,
                    quality=quality,
                )
                return

            channel.value = value
            channel.timestamp = time.time()
            channel.quality = quality

    def get(
        self,
        name: str,
    ) -> LiveChannel | None:
        with self._lock:
            return self._channels.get(name)


def _parse_payload(
    payload: str,
) -> Any:
    value = payload.strip()

    if value == "1":
        return True

    if value == "0":
        return False

    try:
        return float(
            value.replace(",", ".")
        )
    except ValueError:
        return value


class SourceMqttWorker(
    threading.Thread
):
    def __init__(
        self,
        model: LiveModel,
        *,
        host: str = MQTT_HOST,
        port: int = MQTT_PORT,
    ) -> None:
        super().__init__(daemon=True)

        self.model = model
        self.host = host
        self.port = int(port)
        self._stop_event = (
            threading.Event()
        )

        self._client = None

        self.model.update(
            "mqtt_connected",
            False,
            quality="bad",
        )

    def _new_client(self):
        import paho.mqtt.client as mqtt

        try:
            client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION1,
                protocol=mqtt.MQTTv311,
            )
        except AttributeError:
            client = mqtt.Client(
                protocol=mqtt.MQTTv311
            )

        client.on_connect = self._on_connect
        client.on_disconnect = (
            self._on_disconnect
        )
        client.on_message = self._on_message

        return client

    def _on_connect(
        self,
        client,
        userdata,
        flags,
        rc,
    ):
        ok = int(rc) == 0

        self.model.update(
            "mqtt_connected",
            ok,
            quality=(
                "good"
                if ok
                else "bad"
            ),
        )

        if ok:
            client.subscribe(
                "cs/#",
                qos=0,
            )

    def _on_disconnect(
        self,
        client,
        userdata,
        rc,
    ):
        self.model.update(
            "mqtt_connected",
            False,
            quality="bad",
        )

    def _on_message(
        self,
        client,
        userdata,
        message,
    ):
        try:
            payload = message.payload.decode(
                "utf-8",
                errors="ignore",
            )
        except Exception:
            payload = ""

        self.model.update(
            message.topic,
            _parse_payload(payload),
        )

    def run(self) -> None:
        try:
            self._client = (
                self._new_client()
            )

            self._client.connect(
                self.host,
                self.port,
                keepalive=30,
            )

            self._client.loop_start()

            while (
                not
                self._stop_event.wait(0.1)
            ):
                pass

        except Exception:
            self.model.update(
                "mqtt_connected",
                False,
                quality="bad",
            )

        finally:
            if self._client is not None:
                try:
                    self._client.loop_stop()
                except Exception:
                    pass

                try:
                    self._client.disconnect()
                except Exception:
                    pass

            self.model.update(
                "mqtt_connected",
                False,
                quality="bad",
            )

    def stop(self) -> None:
        self._stop_event.set()

        if self.is_alive():
            self.join(timeout=3.0)

    def publish_value(
        self,
        topic: str,
        value: float,
    ) -> None:
        connected = self.model.get(
            "mqtt_connected"
        )

        if (
            connected is None
            or not connected.value
            or self._client is None
        ):
            raise RuntimeError(
                "Source MQTT is not connected."
            )

        info = self._client.publish(
            topic,
            payload=f"{float(value):.2f}",
            qos=0,
            retain=False,
        )

        rc = getattr(info, "rc", 0)

        if rc != 0:
            raise RuntimeError(
                "MQTT publish failed "
                f"with code {rc}."
            )


class CupSwitchClient:
    def __init__(
        self,
        model: LiveModel,
        *,
        host: str = CUP_HOST,
        timeout_s: float = 2.5,
    ) -> None:
        self.model = model
        self.host = host
        self.timeout_s = timeout_s

        self.model.update(
            "cup/connected",
            False,
            quality="bad",
        )

    def _get(
        self,
        path: str,
    ) -> str:
        request = Request(
            f"http://{self.host}{path}",
            method="GET",
        )

        with urlopen(
            request,
            timeout=self.timeout_s,
        ) as response:
            return response.read().decode(
                "utf-8",
                errors="replace",
            )

    def poll_status(self) -> None:
        try:
            text = self._get("/")

            selected = re.search(
                r"SelectedCup:\s*(\d+)",
                text,
                re.IGNORECASE,
            )

            hv = re.search(
                r"HV:\s*(ON|OFF)",
                text,
                re.IGNORECASE,
            )

            self.model.update(
                "cup/connected",
                True,
            )

            if selected:
                self.model.update(
                    "cup/selected",
                    int(
                        selected.group(1)
                    ),
                )

            if hv:
                self.model.update(
                    "cup/hv",
                    hv.group(1).upper(),
                )

        except Exception:
            self.model.update(
                "cup/connected",
                False,
                quality="bad",
            )

    def select_cup(
        self,
        number: int,
    ) -> None:
        self._get(
            f"/select?c={int(number)}"
        )

        self.poll_status()


class _LineSocket:
    def __init__(
        self,
    ) -> None:
        self.sock = None
        self.buffer = b""

    def open(
        self,
        host: str,
        port: int,
        connect_timeout: float,
        io_timeout: float,
    ) -> None:
        self.close()

        self.sock = socket.create_connection(
            (host, int(port)),
            timeout=connect_timeout,
        )

        self.sock.settimeout(
            io_timeout
        )

        self.buffer = b""

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass

        self.sock = None
        self.buffer = b""

    def send(
        self,
        command: str,
    ) -> None:
        if self.sock is None:
            raise RuntimeError(
                "Socket is not connected."
            )

        self.sock.sendall(
            (
                command.rstrip()
                + "\n"
            ).encode("ascii")
        )

    def readline(self) -> str:
        if self.sock is None:
            raise RuntimeError(
                "Socket is not connected."
            )

        while b"\n" not in self.buffer:
            chunk = self.sock.recv(4096)

            if not chunk:
                raise ConnectionError(
                    "Connection closed."
                )

            self.buffer += chunk

        line, self.buffer = (
            self.buffer.split(
                b"\n",
                1,
            )
        )

        return (
            line.decode(
                "ascii",
                errors="ignore",
            )
            .strip()
            .strip("\r")
        )

    def query(
        self,
        command: str,
    ) -> str:
        self.send(command)
        return self.readline()


class KeithleyWorker(
    threading.Thread
):
    def __init__(
        self,
        model: LiveModel,
        *,
        host: str = KEITHLEY_HOST,
        port: int = KEITHLEY_PORT,
        configure_on_connect: bool = True,
    ) -> None:
        super().__init__(daemon=True)

        self.model = model
        self.host = host
        self.port = int(port)
        self.configure_on_connect = bool(
            configure_on_connect
        )

        self._stop_event = (
            threading.Event()
        )

        self._scpi = _LineSocket()

        self.model.update(
            "keithley/connected",
            False,
            quality="bad",
        )

    def _configure(self) -> None:
        initial = [
            "*RST",
            ":SYST:ZCH ON",
            ":SYST:ZCOR ON",
            ":FORM:ELEM READ",
            ":SENS:FUNC 'CURR'",
            ":SENS:CURR:RANG:AUTO ON",
            ":SENS:CURR:NPLC 0.1",
            ":SYST:ZCH OFF",
        ]

        for command in initial:
            self._scpi.send(command)
            time.sleep(0.3)

        tune = [
            ":SENS:CURR:RANG:AUTO ON",
            ":SENS:CURR:NPLC 0.05",
            ":SYST:AZER:STAT OFF",
            ":SENS:AVER:COUNT 10",
            ":SENS:AVER:TCON MOV",
            ":SENS:AVER:STAT OFF",
        ]

        for command in tune:
            self._scpi.send(command)

    def _connect(self) -> None:
        self._scpi.open(
            self.host,
            self.port,
            0.8,
            2.0,
        )

        if self.configure_on_connect:
            self._configure()

        self.model.update(
            "keithley/connected",
            True,
        )

    def _disconnect(self) -> None:
        self._scpi.close()

        self.model.update(
            "keithley/connected",
            False,
            quality="bad",
        )

    def run(self) -> None:
        while not self._stop_event.is_set():
            connected = self.model.get(
                "keithley/connected"
            )

            if (
                connected is None
                or not connected.value
            ):
                try:
                    self._connect()
                except Exception:
                    self._disconnect()

                    self._stop_event.wait(
                        0.5
                    )
                    continue

            try:
                response = self._scpi.query(
                    "READ?"
                )

                current_a = float(
                    response
                )

                self.model.update(
                    "keithley/current_A",
                    current_a,
                )

                self._stop_event.wait(
                    1.0 / 15.0
                )

            except Exception:
                self._disconnect()

                self._stop_event.wait(
                    0.2
                )

        self._disconnect()

    def stop(self) -> None:
        self._stop_event.set()

        if self.is_alive():
            self.join(timeout=3.0)


class MagnetWorker(
    threading.Thread
):
    def __init__(
        self,
        model: LiveModel,
        *,
        host: str = MAGNET_HOST,
        port: int = MAGNET_PORT,
    ) -> None:
        super().__init__(daemon=True)

        self.model = model
        self.host = host
        self.port = int(port)

        self._stop_event = (
            threading.Event()
        )

        self._commands = queue.Queue()

        self._socket = _LineSocket()

        self.model.update(
            "magnet_connected",
            False,
            quality="bad",
        )

    def _connect(self) -> None:
        self._socket.open(
            self.host,
            self.port,
            2.0,
            2.0,
        )

        self.model.update(
            "magnet_connected",
            True,
        )

    def _disconnect(self) -> None:
        self._socket.close()

        self.model.update(
            "magnet_connected",
            False,
            quality="bad",
        )

    def _poll(self) -> None:
        current = float(
            self._socket.query(
                "meas:curr?"
            )
        )

        voltage = float(
            self._socket.query(
                "meas:volt?"
            )
        )

        self.model.update(
            "magnet_current_meas",
            current,
        )

        self.model.update(
            "magnet_voltage_meas",
            voltage,
        )

    def _set_current(
        self,
        current_a: float,
    ) -> None:
        current_a = max(
            0.0,
            min(
                120.0,
                float(current_a),
            ),
        )

        self._socket.send(
            f"sour:curr {current_a:.4f}"
        )

        self._socket.send(
            "sour:volt 60.000"
        )

        self.model.update(
            "magnet_current_set",
            current_a,
        )

    def set_current(
        self,
        current_a: float,
    ) -> None:
        self._commands.put(
            float(current_a)
        )

    def run(self) -> None:
        next_poll = 0.0

        while not self._stop_event.is_set():
            connected = self.model.get(
                "magnet_connected"
            )

            if (
                connected is None
                or not connected.value
            ):
                try:
                    self._connect()
                except Exception:
                    self._disconnect()

                    self._stop_event.wait(
                        0.5
                    )
                    continue

            try:
                try:
                    current = (
                        self._commands.get_nowait()
                    )

                    self._set_current(
                        current
                    )

                except queue.Empty:
                    pass

                now = time.monotonic()

                if now >= next_poll:
                    self._poll()
                    next_poll = now + 1.0

                self._stop_event.wait(
                    0.02
                )

            except Exception:
                self._disconnect()

        self._disconnect()

    def stop(self) -> None:
        self._stop_event.set()

        if self.is_alive():
            self.join(timeout=3.0)


class StandaloneLiveBackend:
    """
    Minimal FLAVIA-compatible backend for mcsirius.

    Only source MQTT, Cup switch, Keithley and magnet are
    instantiated.
    """

    def __init__(
        self,
        *,
        configure_keithley: bool = True,
    ) -> None:
        self.model = LiveModel()

        self.mqtt = SourceMqttWorker(
            self.model
        )

        self.cup = CupSwitchClient(
            self.model
        )

        self.keithley = KeithleyWorker(
            self.model,
            configure_on_connect=configure_keithley,
        )

        self.magnet = MagnetWorker(
            self.model
        )

        self._started = False

    def start(self) -> None:
        if self._started:
            return

        self._started = True

        self.mqtt.start()
        self.keithley.start()
        self.magnet.start()

        self.cup.poll_status()

    def stop(self) -> None:
        if not self._started:
            return

        self._started = False

        self.mqtt.stop()
        self.keithley.stop()
        self.magnet.stop()

    def set_channel(
        self,
        channel_name: str,
        value: float,
    ) -> None:
        topic = SOURCE_COMMAND_TOPICS.get(
            channel_name
        )

        if topic is None:
            raise KeyError(
                "Unsupported live source channel: "
                f"{channel_name}"
            )

        self.mqtt.publish_value(
            topic,
            float(value),
        )

    def set_magnet_current(
        self,
        current_a: float,
    ) -> None:
        current_a = float(current_a)

        if not 0.0 <= current_a <= 120.0:
            raise ValueError(
                "Magnet current must lie "
                "between 0 and 120 A."
            )

        self.magnet.set_current(
            current_a
        )

    def wait_until_ready(
        self,
        *,
        timeout_s: float = 12.0,
    ) -> None:
        deadline = (
            time.monotonic()
            + timeout_s
        )

        connections = (
            "mqtt_connected",
            "magnet_connected",
            "keithley/connected",
            "cup/connected",
        )

        readbacks = (
            "cs/sputter/meas_u_v",
            "cs/extraction/meas_u_v",
            "cs/einzellens/meas_u_v",
            "magnet_current_meas",
            "keithley/current_A",
        )

        while True:
            self.cup.poll_status()

            connected = all(
                (
                    channel := self.model.get(name)
                ) is not None
                and bool(channel.value)
                for name in connections
            )

            readable = all(
                (
                    channel := self.model.get(name)
                ) is not None
                and channel.value is not None
                for name in readbacks
            )

            if connected and readable:
                return

            if time.monotonic() >= deadline:
                missing = []

                for name in connections:
                    channel = self.model.get(name)

                    if (
                        channel is None
                        or not channel.value
                    ):
                        missing.append(name)

                for name in readbacks:
                    channel = self.model.get(name)

                    if (
                        channel is None
                        or channel.value is None
                    ):
                        missing.append(name)

                raise RuntimeError(
                    "Live hardware did not become ready: "
                    + ", ".join(missing)
                )

            time.sleep(0.2)


@dataclass(frozen=True)
class LiveProbeReport:
    sputter_kv: float
    extraction_kv: float
    einzel_kv: float
    magnet_current_a: float
    selected_cup: int | None
    keithley_current_a: float


def _numeric(
    model: LiveModel,
    name: str,
) -> float:
    channel = model.get(name)

    if (
        channel is None
        or channel.value is None
    ):
        raise RuntimeError(
            f"Live channel unavailable: {name}"
        )

    return float(channel.value)


def probe_live_hardware(
    *,
    timeout_s: float = 12.0,
) -> LiveProbeReport:
    """
    Connect to the minimum mcsirius hardware set and read its
    current state.

    No source-voltage, magnet-current, Cup-selection or
    Keithley-configuration command is issued by this probe.
    """

    backend = StandaloneLiveBackend(
        configure_keithley=False
    )

    backend.start()

    try:
        backend.wait_until_ready(
            timeout_s=timeout_s
        )

        selected = backend.model.get(
            "cup/selected"
        )

        selected_cup = (
            None
            if selected is None
            or selected.value is None
            else int(
                float(selected.value)
            )
        )

        return LiveProbeReport(
            sputter_kv=(
                _numeric(
                    backend.model,
                    "cs/sputter/meas_u_v",
                )
                / 1000.0
            ),
            extraction_kv=(
                _numeric(
                    backend.model,
                    "cs/extraction/meas_u_v",
                )
                / 1000.0
            ),
            einzel_kv=(
                _numeric(
                    backend.model,
                    "cs/einzellens/meas_u_v",
                )
                / 1000.0
            ),
            magnet_current_a=_numeric(
                backend.model,
                "magnet_current_meas",
            ),
            selected_cup=selected_cup,
            keithley_current_a=_numeric(
                backend.model,
                "keithley/current_A",
            ),
        )

    finally:
        backend.stop()
