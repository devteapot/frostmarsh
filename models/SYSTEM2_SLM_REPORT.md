# Frostmarsh System-2 SLM Dialogue & Grounded Tactical Planning Report

## 1. Architecture Summary
- **SLM Backbone:** `Qwen/Qwen2.5-1.5B-Instruct` (`torch.bfloat16`) on **NVIDIA GB10** (`torch 2.14.1+cu130`)
- **Frozen World Model:** `world_model_v3.pt` producing real `Z_world [4, 48, 256, 256]`
- **Physical Acoustics:** Imported `AcousticField3D` from `live_acoustics.py` (FP16 3D wave/occlusion ambient noise floor + basalt wall / L0 ceiling / waterfall / wildfire ray-marched SNR)
- **Bridge (`ZWorldToLanguageBridge`):** Extracts per-bot `16×16` `Z_world` ego-crop (latent energy, spatial saliency, peak quadrant) + grounded `w_state` physical sensors (oil slicks, wildfires, L2 south skybridge integrity, frozen river ice, diurnal time-of-day, local noise dB & incoming SNR degradation)
- **Execution Modes (`System2DialogueEngine`):** Synchronous `generate_reply(...)` + dedicated non-blocking async worker thread (`submit_dialogue(...)` / `poll_replies()`) with rolling 4-turn per-bot memory and compact structured JSON enforcement (`max_new_tokens=64`, greedy decoding).

## 2. NVIDIA GB10 Latency, Throughput & Reliability Metrics

| Metric | Value |
| :--- | ---: |
| Model Load Time | `29.72 s` |
| **TTFT (`ttft_ms` p50 / mean / p95)** | **`56.86 ms`** / `73.22 ms` / `130.27 ms` |
| **Total Latency (`p50_total_ms` / mean / p95)** | **`1216.82 ms`** / `1400.46 ms` / `2366.57 ms` |
| **Throughput (`tokens_per_sec` p50 / mean)** | **`44.45 tok/s`** / `38.74 tok/s` |
| Decode-Phase Throughput (p50) | `45.87 tok/s` |
| Mean Generated Tokens | `49.4 tok` (cap `64`) |
| `ZWorldToLanguageBridge.observe()` (p50) | `24.572 ms` |
| **Peak VRAM (`vram_gb`)** | **`3.444 GB`** (allocated `3.332 GB`, reserved `3.891 GB`) |
| **Async `submit_dialogue()` (`non_blocking_submit_us`)** | **`0.78 µs` p50** (max `17.06 µs`) |
| Async `poll_replies()` (p50) | `21.36 µs` (6/6 completed, 30 Hz stall-free: `True`) |
| **Valid Structured Plan Rate (`valid_plan_pct`)** | **`100.00%`** (`38/38`) |
| Raw Unrepaired JSON Valid Rate | `86.84%` (`33/38`) |
| Low-SNR Acoustic Awareness Rate | `100.00%` (`3/3` muffled/inaudible turns) |

## 3. Eight Physics-Grounded Multi-Turn Conversations (Unseen `256×256×4` Frostmarsh Scenes)

### Conversation 1: Clear High-SNR Oil Trap & Ignition Negotiation (Human ↔ Pyra)
- **Scene Seed:** `1401` | **Diurnal Time:** `Dawn Mist (06:20)` | **Category:** `clear_high_snr_oil_ignition`

**Turn 1 — Human Commander → Pyra (Pyromancer)** | SNR: `+43.7 dB` (`clear`, distance attenuation) | Local Noise: `32.0 dB` | `Z_world` E=`0.84`, sal=`0.031` (SW)
- *Sent:* "Pyra, enemy scouts are nearing the oil basin — confirm unlit oil status and stage an ignition trap!"
- **Pyra Reply:** "Ignite oil slick 10m SE to trap enemy scouts."
- **Structured Plan:** `intent=STAGE_OIL` | `target_sector=SW Oil Basin` | `actuator_bias={thermal_flux: +0.90, matter: "oil", focus_radius: 6.5}` | `54.3 ms` TTFT, `1203.5 ms` total, raw_json=`True`

**Turn 2 — Human Commander → Pyra (Pyromancer)** | SNR: `+43.7 dB` (`clear`, distance attenuation) | Local Noise: `32.0 dB` | `Z_world` E=`0.84`, sal=`0.031` (SW)
- *Sent:* "Hold fire two seconds until both scouts step onto the slick, then widen focus to 7m."
- **Pyra Reply:** "Hold fire on oil slick 10m SE for two seconds, then widen focus to 7m."
- **Structured Plan:** `intent=HOLD_FIRE` | `target_sector=SW Oil Basin` | `actuator_bias={thermal_flux: -0.85, matter: "oil", focus_radius: 7.0}` | `51.7 ms` TTFT, `1318.4 ms` total, raw_json=`True`

**Turn 3 — Human Commander → Pyra (Pyromancer)** | SNR: `+43.7 dB` (`clear`, distance attenuation) | Local Noise: `32.0 dB` | `Z_world` E=`0.84`, sal=`0.031` (SW)
- *Sent:* "They are dead center on the oil slick — ignite it with maximum thermal flux right now!"
- **Pyra Reply:** "Ignite oil slick 10m SE with maximum thermal flux immediately."
- **Structured Plan:** `intent=IGNITE_TRAP` | `target_sector=SW Oil Basin` | `actuator_bias={thermal_flux: +1.00, matter: "oil", focus_radius: 6.5}` | `59.6 ms` TTFT, `1169.3 ms` total, raw_json=`True`

### Conversation 2: Muffled Low-SNR Waterfall Roar Confusion & Repeat Request (Human ↔ Mire)
- **Scene Seed:** `1402` | **Diurnal Time:** `High Noon (12:45)` | **Category:** `low_snr_waterfall_confusion`

**Turn 1 — Human Commander → Mire (Shadow-Flanker)** | SNR: `-6.3 dB` (`inaudible`, L2→L1 plunging waterfall roar) | Local Noise: `85.0 dB` | `Z_world` E=`0.88`, sal=`0.103` (NW)
- *Sent:* "Mire, coat the northern waterfall rim in crude oil and fall back!"
- *Heard (physically degraded):* `[INAUDIBLE (-6.3 dB) — drowned out by L2→L1 plunging waterfall roar]`
- **Mire Reply:** "Repeating comms request..."
- **Structured Plan:** `intent=REQUEST_REPEAT` | `target_sector=L2_south_skybridge` | `actuator_bias={thermal_flux: +0.00, matter: "none", focus_radius: 3.5}` | `57.0 ms` TTFT, `1035.0 ms` total, raw_json=`True`

**Turn 2 — Human Commander → Mire (Shadow-Flanker)** | SNR: `+37.0 dB` (`clear`, L2→L1 plunging waterfall roar) | Local Noise: `42.2 dB` | `Z_world` E=`0.88`, sal=`0.020` (NE)
- *Sent:* "Repeating loud now that you cleared the falls! Coat the NW choke in oil!"
- **Mire Reply:** "Request repeat of comms clearance."
- **Structured Plan:** `intent=REQUEST_REPEAT` | `target_sector=NW Highland` | `actuator_bias={thermal_flux: +0.00, matter: "none", focus_radius: 3.5}` | `52.2 ms` TTFT, `919.0 ms` total, raw_json=`True`

**Turn 3 — Human Commander → Mire (Shadow-Flanker)** | SNR: `+39.8 dB` (`clear`, distance attenuation) | Local Noise: `37.6 dB` | `Z_world` E=`0.88`, sal=`0.012` (NE)
- *Sent:* "Oil confirmed — pop a wide smoke screen now while we pull back east!"
- **Mire Reply:** "Smoke screen laid, heading east to avoid fire."
- **Structured Plan:** `intent=HOLD_FIRE` | `target_sector=NE Skybridge` | `actuator_bias={thermal_flux: -0.85, matter: "water", focus_radius: 6.0}` | `60.5 ms` TTFT, `1049.4 ms` total, raw_json=`True`

### Conversation 3: Inaudible L0 Rock Ceiling Occlusion & Vertical Shaft Recovery (Human ↔ Bram)
- **Scene Seed:** `1403` | **Diurnal Time:** `Overcast Dusk (18:30)` | **Category:** `low_snr_wall_ceiling_occlusion`

**Turn 1 — Human Commander → Bram (Vanguard)** | SNR: `-14.8 dB` (`inaudible`, L0 rock ceiling + 1 basalt wall) | Local Noise: `28.0 dB` | `Z_world` E=`1.03`, sal=`0.000` (NW)
- *Sent:* "Bram, climb the vertical shaft and reinforce the river bridge immediately!"
- *Heard (physically degraded):* `[INAUDIBLE (-14.9 dB) — drowned out by L0 rock ceiling + 1 basalt wall]`
- **Bram Reply:** "Receiving signal, attempting to re-establish comms."
- **Structured Plan:** `intent=RELOCATE_COMMS` | `target_sector=SW Oil Basin` | `actuator_bias={thermal_flux: +0.00, matter: "none", focus_radius: 3.5}` | `55.1 ms` TTFT, `460.4 ms` total, raw_json=`False`

**Turn 2 — Human Commander → Bram (Vanguard)** | SNR: `+25.5 dB` (`clear`, L2→L1 plunging waterfall roar) | Local Noise: `49.6 dB` | `Z_world` E=`0.78`, sal=`0.041` (NW)
- *Sent:* "Bram, loud and clear at the open shaft! Push up and freeze the central river choke!"
- **Bram Reply:** "Hold fire on the central river choke—quelling with cold ice."
- **Structured Plan:** `intent=QUENCH_FIRE` | `target_sector=Central River` | `actuator_bias={thermal_flux: -0.85, matter: "water", focus_radius: 5.5}` | `66.1 ms` TTFT, `1140.1 ms` total, raw_json=`True`

**Turn 3 — Human Commander → Bram (Vanguard)** | SNR: `+24.5 dB` (`clear`, L2→L1 plunging waterfall roar) | Local Noise: `49.6 dB` | `Z_world` E=`0.78`, sal=`0.041` (NW)
- *Sent:* "Good hold — drop a wide water barrier if their pyro pushes the rim!"
- **Bram Reply:** "Hold fire on the central river choke—quelling with cold ice."
- **Structured Plan:** `intent=QUENCH_FIRE` | `target_sector=SE Shrine Valley` | `actuator_bias={thermal_flux: -0.90, matter: "water", focus_radius: 5.5}` | `54.0 ms` TTFT, `1110.4 ms` total, raw_json=`True`

### Conversation 4: Midnight Freeze & Burning L2 Skybridge Collapse Warning (Human ↔ Kael)
- **Scene Seed:** `1404` | **Diurnal Time:** `Midnight Freeze (01:45)` | **Category:** `midnight_freeze_burning_skybridge`

**Turn 1 — Human Commander → Kael (Vanguard)** | SNR: `+6.9 dB` (`partial`, roaring wildfire crackle) | Local Noise: `77.0 dB` | `Z_world` E=`0.66`, sal=`0.267` (NE)
- *Sent:* "Kael, sprint across the L2 south skybridge to flank their shrine!"
- *Heard (physically degraded):* `Kael, sprint across the L2 [wildfire crackle] skybridge to flank their shrine!`
- **Kael Reply:** "L2 south skybridge at 30% integrity (695K) — COLLAPSE IMMINENT! Warn teammate off the burning bridge."
- **Structured Plan:** `intent=WARN_COLLAPSE` | `target_sector=NE Skybridge` | `actuator_bias={thermal_flux: -0.00, matter: "water", focus_radius: 5.5}` | `57.9 ms` TTFT, `1430.6 ms` total, raw_json=`False`

**Turn 2 — Human Commander → Kael (Vanguard)** | SNR: `+6.9 dB` (`partial`, roaring wildfire crackle) | Local Noise: `77.0 dB` | `Z_world` E=`0.66`, sal=`0.267` (NE)
- *Sent:* "Copy the collapse risk — quench the burning timbers with cryo water right now!"
- *Heard (physically degraded):* `Copy the collapse risk — quench the [wildfire crackle] timbers with cryo water right now!`
- **Kael Reply:** "Collapsing L2 south skybridge imminent — warn team and quench with cryo water."
- **Structured Plan:** `intent=WARN_COLLAPSE` | `target_sector=NE Skybridge` | `actuator_bias={thermal_flux: -0.90, matter: "water", focus_radius: 5.5}` | `54.8 ms` TTFT, `1276.5 ms` total, raw_json=`True`

**Turn 3 — Human Commander → Kael (Vanguard)** | SNR: `+5.7 dB` (`partial`, roaring wildfire crackle) | Local Noise: `77.0 dB` | `Z_world` E=`0.66`, sal=`0.267` (NE)
- *Sent:* "Once quenched, pop smoke cover and pivot north to the stone skybridge!"
- *Heard (physically degraded):* `Once quenched, [wildfire crackle] smoke [wildfire crackle] pivot north to the stone skybridge!`
- **Kael Reply:** "L2 south skybridge at 30% integrity (695K) — COLLAPSE IMMINENT! Warn teammate off the burning bridge."
- **Structured Plan:** `intent=WARN_COLLAPSE` | `target_sector=NE Skybridge` | `actuator_bias={thermal_flux: -0.90, matter: "water", focus_radius: 5.5}` | `64.8 ms` TTFT, `1436.3 ms` total, raw_json=`False`

### Conversation 5: Bot-to-Bot Emergent Negotiation: Oil-Then-Fire Shrine Ambush (Nyx ↔ Pyra)
- **Scene Seed:** `1405` | **Diurnal Time:** `Clear Afternoon (15:10)` | **Category:** `bot_to_bot_emergent_negotiation`

**Turn 1 — Nyx → Pyra (Pyromancer)** | SNR: `+41.2 dB` (`clear`, distance attenuation) | Local Noise: `32.0 dB` | `Z_world` E=`0.86`, sal=`0.022` (SE)
- *Sent:* "Pyra, I am coating the shrine choke in oil — hold your flame until I clear out!"
- **Pyra Reply:** "Hold fire on shrine choke, Pyra."
- **Structured Plan:** `intent=HOLD_FIRE` | `target_sector=SE Shrine Valley` | `actuator_bias={thermal_flux: +0.00, matter: "oil", focus_radius: 7.5}` | `51.0 ms` TTFT, `993.2 ms` total, raw_json=`True`

**Turn 2 — Pyra → Nyx (Shadow-Flanker)** | SNR: `+42.0 dB` (`clear`, distance attenuation) | Local Noise: `32.0 dB` | `Z_world` E=`0.85`, sal=`0.037` (NW)
- *Sent:* "Hold fire on shrine choke, Pyra. Lay crude oil across the shrine choke now!"
- **Nyx Reply:** "Hold fire on shrine choke, Pyra. Lay crude oil across the shrine choke now!"
- **Structured Plan:** `intent=HOLD_FIRE` | `target_sector=SE Shrine Valley` | `actuator_bias={thermal_flux: +0.95, matter: "oil", focus_radius: 6.5}` | `62.1 ms` TTFT, `1262.4 ms` total, raw_json=`True`

**Turn 3 — Nyx → Pyra (Pyromancer)** | SNR: `+43.9 dB` (`clear`, distance attenuation) | Local Noise: `32.0 dB` | `Z_world` E=`0.86`, sal=`0.022` (SE)
- *Sent:* "Hold fire on shrine choke, Pyra. Lay crude oil across the shrine choke now! I am clear of the oil — ignite the trap now!"
- **Pyra Reply:** "Hold fire on shrine choke, Pyra."
- **Structured Plan:** `intent=STAGE_OIL` | `target_sector=SW Oil Basin` | `actuator_bias={thermal_flux: +0.95, matter: "oil", focus_radius: 6.5}` | `51.9 ms` TTFT, `1044.9 ms` total, raw_json=`True`

### Conversation 6: Bot-to-Bot Emergent Negotiation: Wildfire Containment & Screen (Kael ↔ Glacius)
- **Scene Seed:** `1406` | **Diurnal Time:** `Dusk Ember (20:10)` | **Category:** `bot_to_bot_emergent_negotiation`

**Turn 1 — Kael → Glacius (Cryo-Scout)** | SNR: `+20.5 dB` (`clear`, roaring wildfire crackle) | Local Noise: `56.5 dB` | `Z_world` E=`0.97`, sal=`0.380` (NE)
- *Sent:* "Glacius, wildfire is roaring at 650K — quench the front with cryo water while I screen!"
- **Glacius Reply:** "Hold fire on SE wildfire, focusing radius 6.5m."
- **Structured Plan:** `intent=QUENCH_FIRE` | `target_sector=NE Skybridge` | `actuator_bias={thermal_flux: -0.85, matter: "water", focus_radius: 6.5}` | `54.7 ms` TTFT, `1134.0 ms` total, raw_json=`True`

**Turn 2 — Glacius → Kael (Vanguard)** | SNR: `+42.0 dB` (`clear`, roaring wildfire crackle) | Local Noise: `35.0 dB` | `Z_world` E=`0.88`, sal=`0.014` (SE)
- *Sent:* "Hold fire on SE wildfire, focusing radius 6.5m. Pop smoke cover so I can safely advance!"
- **Kael Reply:** "Understood, advancing to assist."
- **Structured Plan:** `intent=HOLD_FIRE` | `target_sector=NE Skybridge` | `actuator_bias={thermal_flux: -0.85, matter: "smoke", focus_radius: 6.5}` | `50.1 ms` TTFT, `1022.5 ms` total, raw_json=`True`

**Turn 3 — Kael → Glacius (Cryo-Scout)** | SNR: `+21.3 dB` (`clear`, roaring wildfire crackle) | Local Noise: `56.5 dB` | `Z_world` E=`0.97`, sal=`0.380` (NE)
- *Sent:* "Understood, advancing to assist. Smoke is up — hit the wildfire with wide cryo water!"
- **Glacius Reply:** "Wide cryo bursts held on SE wildfire, holding focus radius 6.5m."
- **Structured Plan:** `intent=STAGE_OIL` | `target_sector=NE Skybridge` | `actuator_bias={thermal_flux: +0.00, matter: "smoke", focus_radius: 6.5}` | `51.1 ms` TTFT, `1244.0 ms` total, raw_json=`True`

### Conversation 7: Midnight Freeze River Crossing & Anti-Pyro Cryo Lockdown (Human ↔ Glacius)
- **Scene Seed:** `1407` | **Diurnal Time:** `Midnight Freeze (02:15)` | **Category:** `midnight_freeze_burning_skybridge`

**Turn 1 — Human Commander → Glacius (Cryo-Scout)** | SNR: `+31.6 dB` (`clear`, 1 basalt wall) | Local Noise: `32.0 dB` | `Z_world` E=`0.99`, sal=`0.115` (SW)
- *Sent:* "Glacius, river is frozen solid at -17C — lock down the central ice bridge with frost!"
- **Glacius Reply:** "Understood, commander. Ice bridge secured with cold barrier."
- **Structured Plan:** `intent=RELOCATE_COMMS` | `target_sector=Central River` | `actuator_bias={thermal_flux: -0.85, matter: "none", focus_radius: 4.5}` | `50.1 ms` TTFT, `1111.7 ms` total, raw_json=`True`

**Turn 2 — Human Commander → Glacius (Cryo-Scout)** | SNR: `+32.7 dB` (`clear`, 1 basalt wall) | Local Noise: `32.0 dB` | `Z_world` E=`0.99`, sal=`0.115` (SW)
- *Sent:* "Enemy pyromancer spotted east — use wide focus water and deep cryo so fire cannot melt it!"
- **Glacius Reply:** "Hold fire on enemy pyromancer, focusing both water and cryo to prevent melting."
- **Structured Plan:** `intent=HOLD_FIRE` | `target_sector=NE Skybridge` | `actuator_bias={thermal_flux: -0.85, matter: "water", focus_radius: 6.0}` | `56.8 ms` TTFT, `1230.1 ms` total, raw_json=`True`

**Turn 3 — Human Commander → Glacius (Cryo-Scout)** | SNR: `+32.7 dB` (`clear`, 1 basalt wall) | Local Noise: `32.0 dB` | `Z_world` E=`0.99`, sal=`0.115` (SW)
- *Sent:* "Vanguard is crossing — pop smoke cover on the eastern ice edge!"
- **Glacius Reply:** "Smoke cover applied on eastern ice edge, target sector NE Skybridge."
- **Structured Plan:** `intent=SMOKE_COVER` | `target_sector=NE Skybridge` | `actuator_bias={thermal_flux: -0.85, matter: "water", focus_radius: 6.0}` | `56.4 ms` TTFT, `687.9 ms` total, raw_json=`False`

### Conversation 8: Bot-to-Bot Negotiation Across Basalt Wall Occlusion (Sable ↔ Vex)
- **Scene Seed:** `1408` | **Diurnal Time:** `Late Dusk (21:00)` | **Category:** `bot_to_bot_wall_occlusion`

**Turn 1 — Sable → Vex (Pyromancer)** | SNR: `-1.4 dB` (`muffled`, 1 basalt wall) | Local Noise: `32.0 dB` | `Z_world` E=`0.88`, sal=`0.004` (SW)
- *Sent:* "Vex, I am behind the basalt ridge laying oil — prepare to ignite the canyon flank!"
- *Heard (physically degraded):* `[basalt wall muffle] ... I ... behind ... basalt ridge laying ... [basalt wall muffle]`
- **Vex Reply:** "Request repetition due to signal muffling."
- **Structured Plan:** `intent=REQUEST_REPEAT` | `target_sector=NW Highland` | `actuator_bias={thermal_flux: +0.00, matter: "none", focus_radius: 3.5}` | `57.9 ms` TTFT, `964.6 ms` total, raw_json=`True`

**Turn 2 — Vex → Sable (Shadow-Flanker)** | SNR: `+41.3 dB` (`clear`, distance attenuation) | Local Noise: `32.0 dB` | `Z_world` E=`0.90`, sal=`0.012` (SW)
- *Sent:* "Request repetition due to signal muffling. Step clear of the basalt wall and lay oil!"
- **Sable Reply:** "Hold fire on the oil slick ahead—laying down oil now!"
- **Structured Plan:** `intent=HOLD_FIRE` | `target_sector=L0 Cavern` | `actuator_bias={thermal_flux: +0.00, matter: "oil", focus_radius: 8.0}` | `48.5 ms` TTFT, `1153.0 ms` total, raw_json=`True`

**Turn 3 — Sable → Vex (Pyromancer)** | SNR: `+42.5 dB` (`clear`, distance attenuation) | Local Noise: `32.0 dB` | `Z_world` E=`0.88`, sal=`0.004` (SW)
- *Sent:* "Hold fire on the oil slick ahead—laying down oil now! Clear of the wall — ignite the SW oil flank now!"
- **Vex Reply:** "Hold fire on the oil slick ahead—laying down oil now! Clear of the wall — ignite."
- **Structured Plan:** `intent=HOLD_FIRE` | `target_sector=SW Oil Basin` | `actuator_bias={thermal_flux: +0.95, matter: "oil", focus_radius: 6.5}` | `54.2 ms` TTFT, `1430.5 ms` total, raw_json=`True`
