#!/usr/bin/env python3
"""Lightweight zero-dependency asyncio WebSocket demo broadcaster for Frostmarsh Wave 12C.

Streams binary framed updates directly to the Frostmarsh WebGPU browser client:
  - Frame 0x01: [u8=1, u16 bx (LE), u16 by (LE), u8 depth, u32 den_len (LE), density_bytes..., albedo_bytes...]
      Streams 16x16xD bricks from web/assets/z16_demo_volume.bin (if present) plus periodic live
      animated crystal/magma delta bricks so you can watch incremental Surface-Nets + GPU sub-brick
      uploads live in the browser!
  - Frame 0x02: [u8=2, utf8_json_bytes...]
      Broadcasts animated multiplayer hero knights patrolling the bridge, riverbanks, and cave
      plus glowing flying projectiles at 20 Hz.

Usage:
  python3 web/live_ws_demo.py --host 127.0.0.1 --port 8765
  Then open http://127.0.0.1:8080/?ws=ws://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
import struct
import time
from pathlib import Path
from typing import Optional

WS_MAGIC_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
BRICK_XY = 16


def encode_ws_binary_frame(payload: bytes) -> bytes:
    """Encode an unmasked server-to-client RFC 6455 binary WebSocket frame (opcode 0x02)."""
    n = len(payload)
    if n < 126:
        header = bytes([0x82, n])
    elif n < 65536:
        header = struct.pack("!BBH", 0x82, 126, n)
    else:
        header = struct.pack("!BBQ", 0x82, 127, n)
    return header + payload


def encode_brick_frame(bx: int, by: int, depth: int, density: bytes, albedo: bytes) -> bytes:
    """Build Frostmarsh Wave 12C Frame 0x01: [u8=1, u16 bx, u16 by, u8 depth, u32 den_len, den, alb]."""
    assert len(density) == len(albedo)
    header = struct.pack("<BHHBI", 0x01, bx, by, depth, len(density))
    return header + density + albedo


def encode_entities_frame(obj: dict) -> bytes:
    """Build Frostmarsh Wave 12C Frame 0x02: [u8=2, utf8_json_bytes...]."""
    raw = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    return b"\x02" + raw


class Fm3dVolume:
    """In-memory FM3D volume slices for streaming 16x16xD bricks."""

    def __init__(self, width: int, height: int, depth: int, density: bytes, albedo: bytes) -> None:
        self.width = width
        self.height = height
        self.depth = depth
        self.density = density
        self.albedo = albedo

    @classmethod
    def load(cls, path: Path) -> Optional["Fm3dVolume"]:
        if not path.is_file():
            return None
        data = path.read_bytes()
        if len(data) < 16 or data[0:4] != b"FM3D":
            return None
        w, h, d = struct.unpack_from("<III", data, 4)
        n = w * h * d * 4
        if len(data) < 16 + n * 2:
            return None
        return cls(w, h, d, data[16 : 16 + n], data[16 + n : 16 + n * 2])

    def extract_brick(self, bx: int, by: int) -> tuple[bytes, bytes]:
        x0 = bx * BRICK_XY
        y0 = by * BRICK_XY
        cw = min(BRICK_XY, self.width - x0)
        ch = min(BRICK_XY, self.height - y0)
        den_out = bytearray(self.depth * BRICK_XY * BRICK_XY * 4)
        alb_out = bytearray(self.depth * BRICK_XY * BRICK_XY * 4)
        row_bytes = cw * 4
        for alt in range(self.depth):
            for dy in range(ch):
                src_off = ((alt * self.height + (y0 + dy)) * self.width + x0) * 4
                dst_off = ((alt * BRICK_XY + dy) * BRICK_XY) * 4
                den_out[dst_off : dst_off + row_bytes] = self.density[src_off : src_off + row_bytes]
                alb_out[dst_off : dst_off + row_bytes] = self.albedo[src_off : src_off + row_bytes]
        return bytes(den_out), bytes(alb_out)


def build_animated_crystal_brick(depth: int, phase: float) -> tuple[bytes, bytes]:
    """Generate a live mutating 16x16xD brick with a pulsing crystal/magma spire on the riverbank."""
    n = depth * BRICK_XY * BRICK_XY * 4
    den = bytearray(n)
    alb = bytearray(n)
    spire_h = 5.5 + 3.2 * (0.5 + 0.5 * math.sin(phase))
    spire_h = min(spire_h, depth - 2.0)

    for alt in range(depth):
        wy = alt + 0.5
        for dy in range(BRICK_XY):
            for dx in range(BRICK_XY):
                off = ((alt * BRICK_XY + dy) * BRICK_XY + dx) * 4
                # Ground floor up to y = 4.1
                solid = 0.0
                if wy < 4.1:
                    solid = min(1.0, max(0.0, (4.3 - wy) / 0.6))
                # Pulsing hexagonal crystal/obsidian spire at brick center (8, 8)
                r = math.hypot(dx - 8.0, dy - 8.0)
                taper = max(0.35, 1.0 - max(0.0, wy - 4.0) / max(spire_h - 3.5, 1.0))
                rad = 2.6 * taper
                if wy <= spire_h and r < rad + 0.6:
                    c_solid = min(1.0, max(0.0, (rad + 0.45 - r) / 0.9))
                    solid = max(solid, c_solid)
                    if c_solid > 0.25 and wy >= 4.0:
                        glow = int(min(255, max(0, (0.55 + 0.45 * math.sin(phase * 1.5 + wy * 0.6)) * 255)))
                        den[off] = int(solid * 255)
                        den[off + 3] = 48
                        alb[off] = 255
                        alb[off + 1] = 148
                        alb[off + 2] = 52
                        alb[off + 3] = glow
                        continue
                if solid > 0.02:
                    den[off] = int(solid * 255)
                    den[off + 3] = 205
                    alb[off] = 68
                    alb[off + 1] = 104
                    alb[off + 2] = 66
                    alb[off + 3] = 0
    return bytes(den), bytes(alb)


def build_live_entities_payload(tick: int, t: float, fps: float) -> dict:
    """Create 5 animated patrolling heroes + 3 arcing glowing spell projectiles."""
    bridge_x = 128.0 + 13.0 * math.sin(t * 0.42)
    bridge_u = max(-1.0, min(1.0, (bridge_x - 130.0) / 30.0))
    bridge_y = 9.05 - 0.55 * bridge_u * bridge_u

    patrol_ang = t * 0.55
    orbit_x = 113.5 + 3.2 * math.cos(patrol_ang)
    orbit_z = 134.0 + 5.5 * math.sin(patrol_ang)

    cave_x = 97.5 + 4.5 * math.sin(t * 0.35)
    cave_z = 152.2 + 1.4 * math.cos(t * 0.35)

    players = [
        {
            "id": 1,
            "name": "Bridge-Vanguard",
            "x": round(bridge_x, 3),
            "y": round(bridge_y + 0.02, 3),
            "z": 128.2,
            "yaw": 0.0 if math.cos(t * 0.42) >= 0 else math.pi,
            "color": [0.16, 0.54, 0.98],
            "scale": 1.0,
        },
        {
            "id": 2,
            "name": "River-Scout",
            "x": round(orbit_x, 3),
            "y": 4.05,
            "z": round(orbit_z, 3),
            "yaw": round(-patrol_ang, 3),
            "color": [0.94, 0.24, 0.24],
            "scale": 1.0,
        },
        {
            "id": 3,
            "name": "Cavern-Warden",
            "x": round(cave_x, 3),
            "y": 4.25,
            "z": round(cave_z, 3),
            "yaw": round(-1.8 + 0.3 * math.sin(t * 0.7), 3),
            "color": [0.18, 0.84, 0.50],
            "scale": 1.0,
        },
        {
            "id": 4,
            "name": "Ember-Mage",
            "x": 146.5,
            "y": 4.15,
            "z": round(126.0 + 4.0 * math.sin(t * 0.5), 3),
            "yaw": round(math.pi * 0.5 + 0.25 * math.cos(t * 0.5), 3),
            "color": [0.98, 0.66, 0.16],
            "scale": 1.05,
        },
        {
            "id": 5,
            "name": "Bridge-Duelist",
            "x": round(132.0 - 10.0 * math.cos(t * 0.38), 3),
            "y": 9.02,
            "z": 126.6,
            "yaw": 0.2,
            "color": [0.76, 0.38, 0.98],
            "scale": 1.0,
        },
    ]

    p_arc1 = (t * 0.85) % 1.0
    p_arc2 = (t * 0.72 + 0.33) % 1.0
    p_arc3 = (t * 0.95 + 0.66) % 1.0
    projectiles = [
        {
            "id": 101,
            "x": round(114.0 + p_arc1 * 32.0, 3),
            "y": round(10.2 + 3.4 * math.sin(p_arc1 * math.pi), 3),
            "z": round(128.0 + 2.5 * math.sin(p_arc1 * math.tau), 3),
            "color": [1.0, 0.46, 0.12],
            "radius": 0.42,
        },
        {
            "id": 102,
            "x": round(144.0 - p_arc2 * 28.0, 3),
            "y": round(10.6 + 2.8 * math.sin(p_arc2 * math.pi), 3),
            "z": round(124.5 + p_arc2 * 8.0, 3),
            "color": [0.22, 0.86, 1.0],
            "radius": 0.38,
        },
        {
            "id": 103,
            "x": round(96.0 + 6.5 * math.cos(p_arc3 * math.tau), 3),
            "y": round(6.2 + 1.1 * math.sin(p_arc3 * math.tau * 2.0), 3),
            "z": round(152.0 + 3.5 * math.sin(p_arc3 * math.tau), 3),
            "color": [0.30, 1.0, 0.62],
            "radius": 0.34,
        },
    ]

    return {
        "tick": tick,
        "world_day": 1,
        "players": players,
        "projectiles": projectiles,
        "telemetry": {
            "server_tps": round(fps, 1),
            "online_players": len(players),
            "status": "Live WS Demo",
        },
    }


async def ws_handshake(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> bool:
    """Perform a minimal RFC 6455 HTTP/1.1 WebSocket upgrade handshake."""
    try:
        raw = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=5.0)
    except Exception:
        return False

    headers: dict[str, str] = {}
    lines = raw.decode("latin1", errors="ignore").split("\r\n")
    for line in lines[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()

    key = headers.get("sec-websocket-key")
    if not key:
        writer.write(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n")
        await writer.drain()
        return False

    accept = base64.b64encode(
        hashlib.sha1((key + WS_MAGIC_GUID).encode("ascii")).digest()
    ).decode("ascii")

    resp = (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept}\r\n"
        "\r\n"
    ).encode("ascii")
    writer.write(resp)
    await writer.drain()
    return True


async def drain_client_frames(reader: asyncio.StreamReader, stop_event: asyncio.Event) -> None:
    """Read and ignore incoming client frames until close/EOF."""
    try:
        while not stop_event.is_set():
            hdr = await reader.readexactly(2)
            opcode = hdr[0] & 0x0F
            masked = (hdr[1] & 0x80) != 0
            length = hdr[1] & 0x7F
            if length == 126:
                length = struct.unpack("!H", await reader.readexactly(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", await reader.readexactly(8))[0]
            if masked:
                await reader.readexactly(4)
            if length > 0:
                await reader.readexactly(length)
            if opcode == 0x08:  # Close frame
                break
    except Exception:
        pass
    finally:
        stop_event.set()


async def handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    volume: Optional[Fm3dVolume],
    depth: int,
    fps: float,
    stream_volume_bricks: bool,
) -> None:
    peer = writer.get_extra_info("peername")
    if not await ws_handshake(reader, writer):
        writer.close()
        return

    print(f"[Frostmarsh Live WS] Client connected: {peer}")
    stop_event = asyncio.Event()
    read_task = asyncio.create_task(drain_client_frames(reader, stop_event))

    try:
        # Optional initial volume brick burst from web/assets/z16_demo_volume.bin
        if stream_volume_bricks and volume is not None:
            bx_count = volume.width // BRICK_XY
            by_count = volume.height // BRICK_XY
            # Stream central hero region (bridge + cave + waterfall) first
            coords = [
                (bx, by)
                for by in range(max(0, by_count // 2 - 3), min(by_count, by_count // 2 + 4))
                for bx in range(max(0, bx_count // 2 - 4), min(bx_count, bx_count // 2 + 4))
            ]
            for idx, (bx, by) in enumerate(coords):
                if stop_event.is_set():
                    break
                den, alb = volume.extract_brick(bx, by)
                frame = encode_brick_frame(bx, by, volume.depth, den, alb)
                writer.write(encode_ws_binary_frame(frame))
                if (idx + 1) % 4 == 0:
                    await writer.drain()
                    await asyncio.sleep(0.015)
            await writer.drain()
            print(f"[Frostmarsh Live WS] Streamed {len(coords)} FM3D bricks (Z={volume.depth}) to {peer}")

        active_depth = volume.depth if (stream_volume_bricks and volume is not None) else depth
        interval = 1.0 / max(fps, 1.0)
        tick = 0
        t0 = time.monotonic()
        last_brick_t = t0

        while not stop_event.is_set():
            now = time.monotonic()
            elapsed = now - t0
            tick += 1

            # 20 Hz Frame 0x02: live multiplayer hero figurines + glowing projectiles
            ent_frame = encode_entities_frame(build_live_entities_payload(tick, elapsed, fps))
            writer.write(encode_ws_binary_frame(ent_frame))

            # Every 1.2s, push a live mutating 0x01 brick near the riverbank (bx=6, by=8 -> x=96..112, z=128..144)
            if now - last_brick_t >= 1.2:
                last_brick_t = now
                den, alb = build_animated_crystal_brick(active_depth, elapsed)
                brk_frame = encode_brick_frame(6, 8, active_depth, den, alb)
                writer.write(encode_ws_binary_frame(brk_frame))

            await writer.drain()
            await asyncio.sleep(interval)
    except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
        pass
    finally:
        stop_event.set()
        read_task.cancel()
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        print(f"[Frostmarsh Live WS] Client disconnected: {peer}")


async def run_server(args: argparse.Namespace) -> None:
    vol_path = Path(args.volume)
    volume = Fm3dVolume.load(vol_path) if args.stream_volume else None
    if args.stream_volume:
        if volume is not None:
            print(
                f"[Frostmarsh Live WS] Loaded {vol_path} ({volume.width}x{volume.height}x{volume.depth})"
            )
        else:
            print(f"[Frostmarsh Live WS] Volume {vol_path} not found; streaming live animated delta bricks.")

    server = await asyncio.start_server(
        lambda r, w: handle_client(r, w, volume, args.depth, args.fps, args.stream_volume),
        host=args.host,
        port=args.port,
    )
    url = f"ws://{args.host}:{args.port}"
    print(f"[Frostmarsh Live WS] Listening on {url} (Z={args.depth}, {args.fps:.0f} Hz entities)")
    print(f"[Frostmarsh Live WS] Open browser at: http://127.0.0.1:8080/?ws={url}")
    async with server:
        await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frostmarsh Wave 12C Live Binary WebSocket Demo Broadcaster (stdlib asyncio)"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")
    parser.add_argument(
        "--volume",
        default=str(Path(__file__).resolve().parent / "assets" / "z16_demo_volume.bin"),
        help="Path to FM3D volume for optional initial brick streaming",
    )
    parser.add_argument(
        "--stream-volume",
        action="store_true",
        help="Stream central FM3D bricks from --volume on client connect",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=16,
        help="Altitude depth (16..32) for animated delta bricks (default: 16)",
    )
    parser.add_argument("--fps", type=float, default=20.0, help="Entity broadcast rate in Hz (default: 20)")
    args = parser.parse_args()
    args.depth = max(16, min(32, args.depth))

    try:
        asyncio.run(run_server(args))
    except KeyboardInterrupt:
        print("\n[Frostmarsh Live WS] Broadcaster stopped.")


if __name__ == "__main__":
    main()
