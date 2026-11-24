/* tslint:disable */
/* eslint-disable */

export class FrostmarshApp {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Apply a live binary `0x01` `16×16×D` (`D = 16..=32`) volumetric brick from WebSocket!
     * Performs incremental Surface-Nets brick rebuild + GPU `queue.write_texture` sub-brick upload.
     * Returns elapsed rebuild+upload time in milliseconds.
     */
    apply_brick3d_raw(bx: number, by: number, depth: number, density_rgba: Uint8Array, albedo_rgba: Uint8Array): number;
    /**
     * Apply a live `0x02` multiplayer JSON entity, actuator beam, & evolution frame from WebSocket!
     * Updates live 5-archetype hero figurines, glowing projectiles, diagonal beam orb-streams,
     * world evolution stats, and telemetry HUD.
     * Returns total active rendered entities (`players + projectiles + beams`).
     */
    apply_live_entities_json(json_str: string): number;
    /**
     * Wave 24A: Unified WebSocket binary frame dispatcher for `0x01` full bricks,
     * `0x02` live entities JSON, and `0x03` sparse delta bricks. Returns handled frame tag.
     */
    apply_ws_binary_frame(frame: Uint8Array): number;
    /**
     * Wave 24A: Apply a live binary `0x03` sparse delta-brick payload (10-byte packed records)
     * onto `16×16×D` brick `(bx, by)`, triggering incremental occupancy, Surface-Nets, and GPU
     * 3D texture sub-region upload! Returns elapsed rebuild+upload time in milliseconds.
     */
    apply_ws_brick_delta(bx: number, by: number, depth: number, payload: Uint8Array): number;
    clear_keys(): void;
    /**
     * Create and initialize the Frostmarsh WebGPU application on the given `<canvas>` id.
     * If `fm3d_bytes` is provided, parses it as an `FM3D` volume (`Z=16..32+`); otherwise
     * generates the built-in isotropic `256×256×16` Frostmarsh True-3D world in WASM.
     */
    static create(canvas_id: string, fm3d_bytes?: Uint8Array | null): Promise<FrostmarshApp>;
    /**
     * Cycle live client-side WGSL 4-Style Neural Compute (`Y`-key):
     * `0 = NORDIC`, `1 = VOLCANIC EMBER`, `2 = ARCANE THERMAL`, `3 = EMERALD CAVERN`.
     */
    cycle_style_mode(): number;
    /**
     * Return compact JSON telemetry for the browser HUD overlay.
     */
    get_hud_stats(): string;
    get_style_name(): string;
    /**
     * Hot-swap to an external `FM3D` neural volume (`Z=16..32+`) in browser.
     */
    load_fm3d_volume(bytes: Uint8Array, label: string): void;
    /**
     * Hot-swap back to the built-in procedural isotropic `256×256×16` showcase world.
     */
    load_procedural_showcase(): void;
    /**
     * Generate and hot-swap a procedural isotropic `256×256×D` (`D = 16..=32`) showcase world.
     */
    load_procedural_showcase_depth(depth: number): void;
    orbit_drag(dx: number, dy: number): void;
    resize(width: number, height: number): void;
    set_camera_preset(preset: number): void;
    set_key(code: string, pressed: boolean): void;
    set_style_mode(mode: number): number;
    tick_and_render(dt_sec: number): void;
    /**
     * Toggle 3D Capsule Collision Walk Mode (`C`-key) with Smart 3D Cave Ceiling Cutaway.
     * Returns `true` when Capsule Walk is active.
     */
    toggle_capsule_mode(): boolean;
    /**
     * Toggle stylized `Pixel-2×SS` ↔ `Clean-3D` (`T`-key). Returns `true` when `Pixel-2×SS` is active.
     */
    toggle_pixel_style(): boolean;
    zoom_wheel(delta_y: number): void;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_frostmarshapp_free: (a: number, b: number) => void;
    readonly frostmarshapp_apply_brick3d_raw: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => void;
    readonly frostmarshapp_apply_live_entities_json: (a: number, b: number, c: number, d: number) => void;
    readonly frostmarshapp_apply_ws_binary_frame: (a: number, b: number, c: number, d: number) => void;
    readonly frostmarshapp_apply_ws_brick_delta: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
    readonly frostmarshapp_clear_keys: (a: number) => void;
    readonly frostmarshapp_create: (a: number, b: number, c: number) => number;
    readonly frostmarshapp_cycle_style_mode: (a: number) => number;
    readonly frostmarshapp_get_hud_stats: (a: number, b: number) => void;
    readonly frostmarshapp_get_style_name: (a: number, b: number) => void;
    readonly frostmarshapp_load_fm3d_volume: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly frostmarshapp_load_procedural_showcase: (a: number) => void;
    readonly frostmarshapp_load_procedural_showcase_depth: (a: number, b: number) => void;
    readonly frostmarshapp_orbit_drag: (a: number, b: number, c: number) => void;
    readonly frostmarshapp_resize: (a: number, b: number, c: number) => void;
    readonly frostmarshapp_set_camera_preset: (a: number, b: number) => void;
    readonly frostmarshapp_set_key: (a: number, b: number, c: number, d: number) => void;
    readonly frostmarshapp_set_style_mode: (a: number, b: number) => number;
    readonly frostmarshapp_tick_and_render: (a: number, b: number) => void;
    readonly frostmarshapp_toggle_capsule_mode: (a: number) => number;
    readonly frostmarshapp_toggle_pixel_style: (a: number) => number;
    readonly frostmarshapp_zoom_wheel: (a: number, b: number) => void;
    readonly __wasm_bindgen_func_elem_6613: (a: number, b: number, c: number, d: number) => void;
    readonly __wasm_bindgen_func_elem_6634: (a: number, b: number, c: number, d: number) => void;
    readonly __wasm_bindgen_func_elem_841: (a: number, b: number, c: number, d: number) => void;
    readonly __wasm_bindgen_func_elem_841_24: (a: number, b: number, c: number, d: number) => void;
    readonly __wasm_bindgen_func_elem_841_25: (a: number, b: number, c: number, d: number) => void;
    readonly __wbindgen_export: (a: number, b: number) => number;
    readonly __wbindgen_export2: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_export3: (a: number) => void;
    readonly __wbindgen_export4: (a: number, b: number, c: number) => void;
    readonly __wbindgen_export5: (a: number, b: number) => void;
    readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
