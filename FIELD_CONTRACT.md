# Frostmarsh Neural World Model — Shared Contract (v1)

Single source of truth shared between `crates/field-core` (Rust) and `ml/` (PyTorch on `spark-2b4e`).

## 1. Grid & Units
- Spatial grid: `W × H` (default playable arena `128 × 128`; benchmarks also run `256 × 256`).
- Chunk size: `16 × 16` cells.
- Field tick rate: `10 Hz` (`dt = 0.1 s`).
- Nominal cell mass: `C_NOM = 1000` (`u16` per species).
- Hard cell volume ceiling: `H_CAP = 4000` (`u32`, `v_s = 1` for all species).
- Thermal energy: `energy: i64` in `(cK * c)` units where `T_cK = energy / C_cell`, `C_cell = Σ_s m_s * c_s`.
  - Ambient temperature: `T_AMB = 29000` (`290.00 K` = `16.85 °C`).
  - Freezing point: `27315` (`0.00 °C`).
  - Boiling point: `37315` (`100.00 °C`).
  - Max clamped temperature: `T_MAX = 150000` (`1500.00 K`).
  - Min clamped temperature: `T_MIN = 15000` (`150.00 K`).

## 2. Species (`S = 10`)
| ID | Name  | Mobile | `c` | `k` | `u0` (per unit, `cK*c`) | Role |
|---:|-------|:------:|----:|----:|------------------------:|------|
| 0  | water | Yes    | 42  | 6   | 0                       | River liquid |
| 1  | ice   | No     | 21  | 22  | -336000                 | Walkable frozen bridge / raft |
| 2  | steam | Yes    | 20  | 2   | +840000                 | Rising hot gas |
| 3  | soil  | No     | 15  | 10  | 0                       | Bank terrain (soil+water = mud) |
| 4  | rock  | No     | 10  | 20  | 0                       | Solid wall obstacle |
| 5  | wood  | No     | 17  | 3   | +450000                 | Flammable trees/bridges |
| 6  | ash   | No     | 10  | 2   | 0                       | Burn residue |
| 7  | smoke | Yes    | 12  | 2   | 0                       | Rising smoke (dissipates to atmosphere) |
| 8  | oil   | Yes    | 20  | 3   | +750000                 | Flammable liquid (floats/spreads) |
| 9  | plant | No     | 25  | 4   | +300000                 | Flammable grass/marsh brush |

Mobile species order (`M = 4`): `MOBILE = [0 (water), 2 (steam), 7 (smoke), 8 (oil)]`.

## 3. Stoichiometric Reactions (`R = 7`)
All reactions satisfy `Σ ν_r = 0` (exact mass conservation).
Thermal energy released per integer reaction unit `ξ_r`: `ΔE_th = -(Σ_s ν_{r,s} * u0_s) * ξ_r`.

| ID | Name      | `ν`                          | Temp range (`cK`) | `ΔE_th / ξ` |
|---:|-----------|------------------------------|-------------------|------------:|
| 0  | Freeze    | `water: -1, ice: +1`         | `T < 27315`       | `+336000`   |
| 1  | Melt      | `ice: -1, water: +1`         | `T > 27315`       | `-336000`   |
| 2  | Boil      | `water: -1, steam: +1`       | `T > 37315`       | `-840000`   |
| 3  | Condense  | `steam: -1, water: +1`       | `T < 37000`       | `+840000`   |
| 4  | WoodBurn  | `wood: -2, ash: +1, smoke:+1`| `T > 52000`       | `+900000`   |
| 5  | OilBurn   | `oil: -1, smoke: +1`         | `T > 44000`       | `+750000`   |
| 6  | PlantBurn | `plant: -2, ash: +1, smoke:+1`| `T > 46000`      | `+600000`   |

## 4. Neural World Model ONNX I/O Contract
- **Input name:** `state` — `float32 [B, 18, H, W]`
  - `0..10`: `mass[0..10] / 1000.0`
  - `10`: `(T_cK - 29000.0) / 10000.0`
  - `11`: `ln(1.0 + total_mass / 1000.0)`
  - `12`: `flow_e / 1000.0` (last tick projected liquid flux East)
  - `13`: `flow_s / 1000.0` (last tick projected liquid flux South)
  - `14`: `elevation / 31.0`
  - `15`: `rock_wall` (`1.0` if `mass[rock] >= 500` else `0.0`)
  - `16`: `forcing_energy / 1e8` (clamped to `[-10.0, 10.0]`)
  - `17`: `forcing_mass / 1000.0` (clamped to `[-4.0, 4.0]`)
- **Output name:** `deltas` — `float32 [B, 17, H, W]`
  - `0..4`: `flux_e[0..4]` for `[water, steam, smoke, oil]` in `mass / 1000.0` units (signed; positive = East `x -> x+1`)
  - `4..8`: `flux_s[0..4]` for `[water, steam, smoke, oil]` in `mass / 1000.0` units (signed; positive = South `y -> y+1`)
  - `8`: `heat_e` conduction flux East in `1e6` energy units (signed)
  - `9`: `heat_s` conduction flux South in `1e6` energy units (signed)
  - `10..17`: `rxn_extent[0..7]` in `100.0` reaction units (`>= 0`, `softplus` in model)

## 5. Edge Ownership & Integer Projection
- Cell `(x, y)` owns its **East edge** `(x,y) -> (x+1,y)` (zero for `x == W-1`) and **South edge** `(x,y) -> (x,y+1)` (zero for `y == H-1`).
- Every edge flux is rounded once (`round_ties_even`), clamped edge-locally (`|q_s| <= floor(m_donor_s / 4)`, total inflow per edge `<= floor((H_CAP - V_recv) / 4)`), and applied equal-and-opposite (`donor -= q`, `recv += q`).
- Heat conduction obeys discrete 2nd-law maximum principle: `sign(G) == sign(T_d - T_r)` and `|G| <= floor(|T_d - T_r| * C_d * C_r / (4 * (C_d + C_r)))`.
- Advected thermal energy is computed deterministically: `E_adv = floor(q_s * c_s * E_d / C_d)`.
- Reactions run on post-transport masses with temperature gating, limiting-reagent scaling, and endothermic energy bounds.
- Ambient cooling / smoke dissipation is tracked explicitly in `atmosphere_mass` and `atmosphere_energy` accounts so total conserved mass and energy remain **identically 0 error**.
