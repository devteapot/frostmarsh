# Approach 4 × Live Neural World — Alternative QUIC Client Report

## Overview
`frostmarsh-approach4-client` is a standalone, headless-capable Rust `wgpu 30.0.2` + `quinn` QUIC client that renders the **live running Frostmarsh Neural World Server** (`127.0.0.1:4433`, `world_model.pt` + `ecosystem_model.pt` on NVIDIA GB10 CUDA FP16) using all four core **Approach 4** GPU rendering techniques.

## Live Neural World Server Verification (`127.0.0.1:4433`)
| Metric | Live Measured Value |
|---|---|
| **GPU Adapter** | `NVIDIA GB10` (`Vulkan`) |
| **Sim Mode / Neural Backend** | `neural` / `cuda-fp16` (Ecosystem: `true`) |
| **Live Sim Tick / Field Tick** | `168231` / `55026` |
| **GB10 Neural Step Latency** | `0.80 ms` last / `3.63 ms` p99 |
| **Conservation Errors (Mass / Energy)** | `0` / `0` (Projection active: `19.31%`) |
| **QUIC Wire Throughput** | `248.9 KB/s` (`1612` LZ4 `A+` chunks, `133` 30 Hz entity snapshots) |
| **LZ4 `A+` Chunk Decompress** | `0.6 µs` / chunk |
| **Live Entities Rendered** | `7` heroes/bots, `2` elemental projectiles, `0` carved ice rafts (Score West `0` : East `0`) |

## Approach 4 Core Techniques Wired Live
1. **Parametric CSG/SDF Greedy-Meshed Baked-AO 3D Characters & Props** (`ParamVolume::compile_to_mesh()`):
   - Built in `5.96 ms` (`38404` verts / `19198` tris, `387` surfel prototypes).
   - **FrostBots** (`team == 0`) → sculpted cobalt-tabard `HeroKnight`s (steel cuirass, gold crests, kite shield, broadsword); **PyroBots** (`team == 1`) → sculpted crimson `HeroMage`s (curved wizard hat, gold stole, fire orb staff); carved `CellBody` floes → beveled `IceRaft`s; golden `RelicPedestal` + bobbing `GoldenRelic`; dash i-frame `DashHalo`s; `MeleeArc` slashes; and `MarshDeer` herds driven by live `ecosystem_model.pt` critter density bits!
2. **Smooth-Min SDF Hard-Z GPU-Instanced Foliage & Wildfire Surfels** (`ParamVolume::compile_to_surfels()`):
   - Live `A+` `wood` (`412` cells → `395` pine/snow-pine surfel trees) and `plant` (`2502` cells → `1166` marsh brush/reed clusters) dynamically update as wildfires burn (`198` active burning cells → `316` flickering `WildfireFlame` surfel plumes). When neural wildfires consume forest cells into `ash` (`462` cells), 3D surfel trees burn away live!
3. **`O(N)` Separable Top-Weighted Smoothed Terrain + Dynamic Firelight & 1 KB Animated `flash()` Palette**:
   - Incremental `16×16` dirty-chunk mesh & normal smoothing (`24765` verts / `35528` tris across 64 chunks) with compositional `A+` water/ice/rock/oil/soil/ash heightfield, animated flow-directed river water, 8 dynamic GPU point lights clustered around live wildfires + Golden Relic + elemental bolts, and a **1,024-byte (`256×1` RGBA8)** `flash()` palette uploaded per frame.
4. **9-Step Cool-Indigo (`#261e48`) / Warm-Cream (`#fff4d6`) Hue-Shifted Tone Ramp + 2× SS Pixel-Art / Clean-3D Resolve**:
   - Two-stage resolve decouples pixel-art quantization, Bayer 4×4 dither, crevice AO, and cool-indigo outlines (`art_res`) from final `1080p`/`4K` output, with toggleable `--style pixel|clean`.

## Performance Benchmarks (NVIDIA GB10)

### GPU Frame-Time Benchmark (Live Frostmarsh Scene)
| Mode | Output | Art Res | p50 (ms) | p99 (ms) | Mean (ms) | FPS (p50) |
|---|---|---|---:|---:|---:|---:|
| 1080p Pixel-Art (px=2) | 1920x1080 | 960x540 | 0.809 | 1.049 | 0.804 | 1235.4 |
| 1080p Clean 3D (px=1) | 1920x1080 | 1920x1080 | 0.727 | 1.133 | 0.769 | 1374.7 |
| 4K Pixel-Art (px=4) | 3840x2160 | 960x540 | 0.833 | 1.125 | 0.866 | 1200.4 |
| 4K Clean 3D (px=1) | 3840x2160 | 3840x2160 | 1.482 | 1.838 | 1.490 | 674.7 |

### 10 Hz Live `A+` Dirty-Chunk Incremental Rebuild Benchmark
- **Observation Window:** `4.08 s` (`40` live 10 Hz ticks, `1474` dirty chunks rebuilt, avg `36.9` dirty chunks/tick)
- **Incremental Dirty-Chunk Mesh + Normal + Surfel Update:** **p50 = `0.724 ms`**, **p99 = `0.877 ms`**, mean = `0.721 ms`
- **Initial Full 64-Chunk Sync Build:** `1.856 ms`
- **Active Scene Geometry:** `35528` terrain tris, `431` instanced 3D meshes, `1880` surfel placements (`47959` GPU-instanced surfel draws)

## Captured Visuals
- `approach4_live_1080p.png` — Crisp `1920×1080` MOBA-angle live battle render (2× SS Pixel-Art Resolve).
- `approach4_orbit_4view.png` — `1920×1080` 2×2 multi-angle showcase (SE overview, close-up combat & surfel pines, northern ice crossing / carved ice raft, and Clean 3D Resolve comparison).
