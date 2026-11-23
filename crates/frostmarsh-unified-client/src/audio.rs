//! Real-Time Procedural 3D Spatial Audio Engine (Zero External Audio Files).
//!
//! 100% pure Rust real-time stereo DSP synthesis at native device rate (44.1 / 48 kHz) via `cpal`:
//! - **Headless / No-Device Safe:** Never initialized in `--headless` mode and gracefully degrades
//!   to a no-op when no audio output device is available.
//! - **Continuous 3D Ambient Bed (lock-free atomics updated from main loop):**
//!   - Plunging waterfall low-pass pink noise panned toward the nearest 3D waterfall
//!   - Crackling wildfire bandpass noise + stochastic crackle impulses proportional to nearby fire
//!   - Dynamic weather wind whoosh (`Clear Frost`, `Blizzard`, `Squall`, `Ember Dusk`)
//! - **Spatialized One-Shot SFX (lock-free SPSC atomic ring buffer):**
//!   - Elemental casts (`Fire` whoosh+crackle, `Frost` harmonic ice chime, `Oil` viscous squelch)
//!   - Combat & world events (`Steam/Combo` resonant boom, `Kill` crisp harmonic chord, `Shrine/Relic` horn)
//!   - **SNR-Filtered Neural-Voice Synth Chirps:** 2-to-3-tone vocal formant synth panned to the
//!     speaker's 3D bearing and dynamically filtered by `AcousticField3D` `snr_db` (crystal-clear
//!     formant chime at `SNR >= +10 dB`, band-pass muffled radio-static burst at `SNR < +2 dB`).

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream, StreamConfig};
use frostmarsh_proto::{WEATHER_BLIZZARD, WEATHER_EMBER, WEATHER_SQUALL};
use std::f32::consts::{FRAC_PI_2, TAU};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;

const SFX_QUEUE_CAP: usize = 64;
const MAX_VOICES: usize = 24;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SfxKind {
    FireCast = 0,
    FrostCast = 1,
    OilCast = 2,
    SteamCombo = 3,
    KillChord = 4,
    VoiceChirp = 5,
    ShrineOrRelic = 6,
}

impl SfxKind {
    #[inline]
    fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::FireCast,
            1 => Self::FrostCast,
            2 => Self::OilCast,
            3 => Self::SteamCombo,
            4 => Self::KillChord,
            5 => Self::VoiceChirp,
            _ => Self::ShrineOrRelic,
        }
    }
}

struct AudioShared {
    muted: AtomicBool,
    waterfall_gain: AtomicU32,
    waterfall_pan: AtomicU32,
    fire_gain: AtomicU32,
    fire_pan: AtomicU32,
    weather_kind: AtomicU8,
    wind_strength: AtomicU32,
    cave_enclosure_bits: AtomicU32,
    // Lock-free SPSC ring buffer (main thread -> audio callback thread)
    q_write: AtomicUsize,
    q_read: AtomicUsize,
    q_word0: [AtomicU32; SFX_QUEUE_CAP],
    q_word1: [AtomicU32; SFX_QUEUE_CAP],
}

impl AudioShared {
    fn new() -> Self {
        Self {
            muted: AtomicBool::new(false),
            waterfall_gain: AtomicU32::new(0.0f32.to_bits()),
            waterfall_pan: AtomicU32::new(0.0f32.to_bits()),
            fire_gain: AtomicU32::new(0.0f32.to_bits()),
            fire_pan: AtomicU32::new(0.0f32.to_bits()),
            weather_kind: AtomicU8::new(0),
            wind_strength: AtomicU32::new(0.65f32.to_bits()),
            cave_enclosure_bits: AtomicU32::new(0.0f32.to_bits()),
            q_write: AtomicUsize::new(0),
            q_read: AtomicUsize::new(0),
            q_word0: std::array::from_fn(|_| AtomicU32::new(0)),
            q_word1: std::array::from_fn(|_| AtomicU32::new(0)),
        }
    }

    fn push_event(&self, kind: SfxKind, param: u8, gain: f32, pan: f32, extra_f32: f32) {
        let w = self.q_write.load(Ordering::Relaxed);
        let r = self.q_read.load(Ordering::Acquire);
        if w.wrapping_sub(r) >= SFX_QUEUE_CAP {
            return;
        }
        let gain_u8 = (gain.clamp(0.0, 1.0) * 255.0).round() as u8;
        let pan_u8 = ((pan.clamp(-1.0, 1.0) * 0.5 + 0.5) * 255.0).round() as u8;
        let word0 = (kind as u32)
            | ((param as u32) << 8)
            | ((gain_u8 as u32) << 16)
            | ((pan_u8 as u32) << 24);
        let idx = w & (SFX_QUEUE_CAP - 1);
        self.q_word0[idx].store(word0, Ordering::Relaxed);
        self.q_word1[idx].store(extra_f32.to_bits(), Ordering::Relaxed);
        self.q_write.store(w.wrapping_add(1), Ordering::Release);
    }
}

pub struct SpatialAudioEngine {
    shared: Arc<AudioShared>,
    _stream: Stream,
}

impl SpatialAudioEngine {
    /// Initializes native stereo DSP output (`cpal`). Returns `None` gracefully if no audio device
    /// is available or stream setup fails (e.g. headless CI / remote container).
    pub fn try_new() -> Option<Self> {
        let host = cpal::default_host();
        let device = host.default_output_device()?;
        let supported = device.default_output_config().ok()?;
        let sample_format = supported.sample_format();
        let config: StreamConfig = supported.into();
        let sample_rate = config.sample_rate.0.max(8000) as f32;
        let channels = config.channels.max(1) as usize;

        let shared = Arc::new(AudioShared::new());
        let dsp_shared = shared.clone();

        let err_fn = |err| {
            eprintln!("[Audio] Output stream error: {}", err);
        };

        let stream_res = match sample_format {
            SampleFormat::F32 => {
                let mut dsp = DspState::new(sample_rate);
                device.build_output_stream(
                    &config,
                    move |data: &mut [f32], _| {
                        dsp.render_interleaved_f32(data, channels, &dsp_shared);
                    },
                    err_fn,
                    None,
                )
            }
            SampleFormat::I16 => {
                let mut dsp = DspState::new(sample_rate);
                device.build_output_stream(
                    &config,
                    move |data: &mut [i16], _| {
                        dsp.render_interleaved_i16(data, channels, &dsp_shared);
                    },
                    err_fn,
                    None,
                )
            }
            SampleFormat::U16 => {
                let mut dsp = DspState::new(sample_rate);
                device.build_output_stream(
                    &config,
                    move |data: &mut [u16], _| {
                        dsp.render_interleaved_u16(data, channels, &dsp_shared);
                    },
                    err_fn,
                    None,
                )
            }
            _ => return None,
        };

        let stream = stream_res.ok()?;
        stream.play().ok()?;
        println!(
            "[Audio] Spatial 3D DSP active ({:.0} Hz, {} ch, {:?})",
            sample_rate, channels, sample_format
        );
        Some(Self {
            shared,
            _stream: stream,
        })
    }

    pub fn set_muted(&self, muted: bool) {
        self.shared.muted.store(muted, Ordering::Relaxed);
    }

    pub fn is_muted(&self) -> bool {
        self.shared.muted.load(Ordering::Relaxed)
    }

    pub fn toggle_mute(&self) -> bool {
        let next = !self.is_muted();
        self.set_muted(next);
        next
    }

    /// Updates continuous 3D ambient bed parameters lock-free from the render loop.
    pub fn update_ambient(
        &self,
        waterfall_gain: f32,
        waterfall_pan: f32,
        fire_gain: f32,
        fire_pan: f32,
        weather_kind: u8,
        wind_strength: f32,
    ) {
        self.shared
            .waterfall_gain
            .store(waterfall_gain.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
        self.shared
            .waterfall_pan
            .store(waterfall_pan.clamp(-1.0, 1.0).to_bits(), Ordering::Relaxed);
        self.shared
            .fire_gain
            .store(fire_gain.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
        self.shared
            .fire_pan
            .store(fire_pan.clamp(-1.0, 1.0).to_bits(), Ordering::Relaxed);
        self.shared.weather_kind.store(weather_kind, Ordering::Relaxed);
        self.shared
            .wind_strength
            .store(wind_strength.clamp(0.0, 2.5).to_bits(), Ordering::Relaxed);
    }

    /// Updates physical 3D cave enclosure (`0.0` open sky → `1.0` deep vaulted cavern) lock-free
    /// to drive the real-time Schroeder stereo cave reverb (`wet = enclosure * 0.42`).
    pub fn set_cave_enclosure(&self, enc: f32) {
        let clamped = if enc.is_finite() { enc.clamp(0.0, 1.0) } else { 0.0 };
        self.shared
            .cave_enclosure_bits
            .store(clamped.to_bits(), Ordering::Relaxed);
    }

    pub fn trigger_sfx(&self, kind: SfxKind, param: u8, gain: f32, pan: f32) {
        if gain > 0.02 {
            self.shared.push_event(kind, param, gain, pan, 0.0);
        }
    }

    /// Triggers an SNR-filtered procedural 2-to-3-tone neural-voice synth chirp at the speaker's 3D bearing.
    pub fn trigger_voice_chirp(&self, symbol_id: u8, snr_db: f32, gain: f32, pan: f32) {
        if gain > 0.02 {
            self.shared
                .push_event(SfxKind::VoiceChirp, symbol_id, gain, pan, snr_db);
        }
    }
}

// =========================================================================
// REAL-TIME AUDIO THREAD DSP SYNTHESIS (ZERO ALLOCATIONS, ZERO LOCKS)
// =========================================================================

#[inline]
fn equal_power_pan(pan: f32) -> (f32, f32) {
    let ang = (pan.clamp(-1.0, 1.0) * 0.5 + 0.5) * FRAC_PI_2;
    (ang.cos(), ang.sin())
}

#[derive(Clone, Copy)]
struct OneShotVoice {
    active: bool,
    kind: SfxKind,
    param: u8,
    gain: f32,
    pan_l: f32,
    pan_r: f32,
    extra: f32,
    age: f32,
    duration: f32,
    phase0: f32,
    phase1: f32,
    phase2: f32,
    bp_lp: f32,
    bp_bp: f32,
}

impl Default for OneShotVoice {
    fn default() -> Self {
        Self {
            active: false,
            kind: SfxKind::FireCast,
            param: 0,
            gain: 0.0,
            pan_l: 0.707,
            pan_r: 0.707,
            extra: 0.0,
            age: 0.0,
            duration: 0.25,
            phase0: 0.0,
            phase1: 0.0,
            phase2: 0.0,
            bp_lp: 0.0,
            bp_bp: 0.0,
        }
    }
}

#[inline]
fn flush_denormal(x: f32) -> f32 {
    if x.is_finite() && x.abs() >= 1e-20 {
        x
    } else {
        0.0
    }
}

/// Pure-Rust zero-allocation Schroeder Stereo Cave Reverb (~16.2 KB fixed ring buffers):
/// 4 tuned prime-length damped feedback comb filters + 2 short stereo allpass diffusers.
pub(crate) struct SchroederCaveReverb {
    comb0: [f32; 677],
    comb1: [f32; 797],
    comb2: [f32; 919],
    comb3: [f32; 1049],
    comb_idx: [usize; 4],
    comb_lp: [f32; 4],
    ap0_l: [f32; 113],
    ap1_l: [f32; 179],
    ap0_r: [f32; 127],
    ap1_r: [f32; 193],
    ap_idx: [usize; 4],
}

impl SchroederCaveReverb {
    pub(crate) fn new() -> Box<Self> {
        Box::new(Self {
            comb0: [0.0; 677],
            comb1: [0.0; 797],
            comb2: [0.0; 919],
            comb3: [0.0; 1049],
            comb_idx: [0; 4],
            comb_lp: [0.0; 4],
            ap0_l: [0.0; 113],
            ap1_l: [0.0; 179],
            ap0_r: [0.0; 127],
            ap1_r: [0.0; 193],
            ap_idx: [0; 4],
        })
    }

    #[inline]
    fn step_comb(buf: &mut [f32], idx: &mut usize, lp: &mut f32, input: f32, fb: f32, damp: f32) -> f32 {
        let i = *idx;
        let out = buf[i];
        *lp = flush_denormal(out * (1.0 - damp) + *lp * damp);
        buf[i] = flush_denormal(input + *lp * fb);
        let next = i + 1;
        *idx = if next >= buf.len() { 0 } else { next };
        out
    }

    #[inline]
    fn step_allpass(buf: &mut [f32], idx: &mut usize, input: f32, g: f32) -> f32 {
        let i = *idx;
        let delayed = buf[i];
        let out = flush_denormal(delayed - input * g);
        buf[i] = flush_denormal(input + delayed * g);
        let next = i + 1;
        *idx = if next >= buf.len() { 0 } else { next };
        out
    }

    #[inline]
    pub(crate) fn process(&mut self, in_l: f32, in_r: f32) -> (f32, f32) {
        let il = if in_l.is_finite() { in_l.clamp(-4.0, 4.0) } else { 0.0 };
        let ir = if in_r.is_finite() { in_r.clamp(-4.0, 4.0) } else { 0.0 };

        // Decorrelated stereo feeds into 4 prime-length damped feedback comb filters
        let x0 = il * 0.72 + ir * 0.28;
        let x1 = ir * 0.72 + il * 0.28;
        let x2 = il * 0.65 + ir * 0.35;
        let x3 = ir * 0.65 + il * 0.35;

        let c0 = Self::step_comb(&mut self.comb0, &mut self.comb_idx[0], &mut self.comb_lp[0], x0, 0.79, 0.28);
        let c1 = Self::step_comb(&mut self.comb1, &mut self.comb_idx[1], &mut self.comb_lp[1], x1, 0.80, 0.30);
        let c2 = Self::step_comb(&mut self.comb2, &mut self.comb_idx[2], &mut self.comb_lp[2], x2, 0.81, 0.32);
        let c3 = Self::step_comb(&mut self.comb3, &mut self.comb_idx[3], &mut self.comb_lp[3], x3, 0.82, 0.34);

        let sum_l = 0.32 * (c0 + c2) + 0.18 * (c1 - c3);
        let sum_r = 0.32 * (c1 + c3) + 0.18 * (c0 - c2);

        // 2 cascaded short prime-length allpass diffusers per channel for lush rocky cave tail
        let dl0 = Self::step_allpass(&mut self.ap0_l, &mut self.ap_idx[0], sum_l, 0.50);
        let dl1 = Self::step_allpass(&mut self.ap1_l, &mut self.ap_idx[1], dl0, 0.46);
        let dr0 = Self::step_allpass(&mut self.ap0_r, &mut self.ap_idx[2], sum_r, 0.50);
        let dr1 = Self::step_allpass(&mut self.ap1_r, &mut self.ap_idx[3], dr0, 0.46);

        (dl1, dr1)
    }
}

struct DspState {
    inv_sr: f32,
    rng: u32,
    // Smoothers
    master_gain: f32,
    wf_gain: f32,
    wf_pan: f32,
    fire_gain: f32,
    fire_pan: f32,
    wind_gain: f32,
    wind_freq: f32,
    cave_enclosure: f32,
    // Pink noise state (Paul Kellet 3-pole economy filter)
    pink_b0: f32,
    pink_b1: f32,
    pink_b2: f32,
    // Ambient filters & LFOs
    wf_lpf: f32,
    wf_lfo: f32,
    fire_lp: f32,
    fire_bp: f32,
    crackle_env: f32,
    crackle_phase: f32,
    wind_lp: f32,
    wind_bp: f32,
    wind_lfo1: f32,
    wind_lfo2: f32,
    ember_lp: f32,
    // Pure-Rust ~16 KB Schroeder Cave Reverb
    cave_reverb: Box<SchroederCaveReverb>,
    // Polyphonic spatial one-shot voice pool
    voices: [OneShotVoice; MAX_VOICES],
    next_voice_rr: usize,
}

impl DspState {
    fn new(sample_rate: f32) -> Self {
        Self {
            inv_sr: 1.0 / sample_rate,
            rng: 0x9E37_79B9,
            master_gain: 1.0,
            wf_gain: 0.0,
            wf_pan: 0.0,
            fire_gain: 0.0,
            fire_pan: 0.0,
            wind_gain: 0.03,
            wind_freq: 280.0,
            cave_enclosure: 0.0,
            pink_b0: 0.0,
            pink_b1: 0.0,
            pink_b2: 0.0,
            wf_lpf: 0.0,
            wf_lfo: 0.0,
            fire_lp: 0.0,
            fire_bp: 0.0,
            crackle_env: 0.0,
            crackle_phase: 0.0,
            wind_lp: 0.0,
            wind_bp: 0.0,
            wind_lfo1: 0.0,
            wind_lfo2: 0.0,
            ember_lp: 0.0,
            cave_reverb: SchroederCaveReverb::new(),
            voices: [OneShotVoice::default(); MAX_VOICES],
            next_voice_rr: 0,
        }
    }

    #[inline]
    fn next_white(&mut self) -> f32 {
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.rng = x;
        (x as i32 as f32) * (1.0 / 2_147_483_648.0)
    }

    #[inline]
    fn next_pink(&mut self, white: f32) -> f32 {
        self.pink_b0 = 0.99765 * self.pink_b0 + white * 0.0990460;
        self.pink_b1 = 0.96300 * self.pink_b1 + white * 0.2965164;
        self.pink_b2 = 0.57000 * self.pink_b2 + white * 1.0526913;
        (self.pink_b0 + self.pink_b1 + self.pink_b2 + white * 0.1848) * 0.22
    }

    fn drain_queue(&mut self, shared: &AudioShared) {
        let mut r = shared.q_read.load(Ordering::Relaxed);
        let w = shared.q_write.load(Ordering::Acquire);
        while r != w {
            let idx = r & (SFX_QUEUE_CAP - 1);
            let word0 = shared.q_word0[idx].load(Ordering::Relaxed);
            let word1 = shared.q_word1[idx].load(Ordering::Relaxed);
            r = r.wrapping_add(1);

            let kind = SfxKind::from_u8((word0 & 0xFF) as u8);
            let param = ((word0 >> 8) & 0xFF) as u8;
            let gain = ((word0 >> 16) & 0xFF) as f32 / 255.0;
            let pan = (((word0 >> 24) & 0xFF) as f32 / 255.0) * 2.0 - 1.0;
            let extra = f32::from_bits(word1);

            self.spawn_voice(kind, param, gain, pan, extra);
        }
        shared.q_read.store(r, Ordering::Release);
    }

    fn spawn_voice(&mut self, kind: SfxKind, param: u8, gain: f32, pan: f32, extra: f32) {
        let slot = self
            .voices
            .iter()
            .position(|v| !v.active)
            .unwrap_or_else(|| {
                let s = self.next_voice_rr;
                self.next_voice_rr = (self.next_voice_rr + 1) % MAX_VOICES;
                s
            });
        let (pan_l, pan_r) = equal_power_pan(pan);
        let duration = match kind {
            SfxKind::FireCast => 0.24,
            SfxKind::FrostCast => 0.34,
            SfxKind::OilCast => 0.20,
            SfxKind::SteamCombo => 0.48,
            SfxKind::KillChord => 0.56,
            SfxKind::VoiceChirp => 0.34,
            SfxKind::ShrineOrRelic => 0.52,
        };
        self.voices[slot] = OneShotVoice {
            active: true,
            kind,
            param,
            gain,
            pan_l,
            pan_r,
            extra,
            age: 0.0,
            duration,
            phase0: 0.0,
            phase1: 0.0,
            phase2: 0.0,
            bp_lp: 0.0,
            bp_bp: 0.0,
        };
    }

    #[inline]
    fn step_stereo(&mut self, shared: &AudioShared) -> (f32, f32) {
        let muted = shared.muted.load(Ordering::Relaxed);
        let tgt_master = if muted { 0.0 } else { 0.72 };
        self.master_gain += (tgt_master - self.master_gain) * 0.0025;

        let tgt_wf_g = f32::from_bits(shared.waterfall_gain.load(Ordering::Relaxed));
        let tgt_wf_p = f32::from_bits(shared.waterfall_pan.load(Ordering::Relaxed));
        let tgt_fi_g = f32::from_bits(shared.fire_gain.load(Ordering::Relaxed));
        let tgt_fi_p = f32::from_bits(shared.fire_pan.load(Ordering::Relaxed));
        let weather = shared.weather_kind.load(Ordering::Relaxed);
        let wind_str = f32::from_bits(shared.wind_strength.load(Ordering::Relaxed));
        let tgt_cave = f32::from_bits(shared.cave_enclosure_bits.load(Ordering::Relaxed)).clamp(0.0, 1.0);

        self.wf_gain += (tgt_wf_g - self.wf_gain) * 0.0008;
        self.wf_pan += (tgt_wf_p - self.wf_pan) * 0.0008;
        self.fire_gain += (tgt_fi_g - self.fire_gain) * 0.0008;
        self.fire_pan += (tgt_fi_p - self.fire_pan) * 0.0008;
        self.cave_enclosure += (tgt_cave - self.cave_enclosure) * 0.0008;

        let (tgt_wg, tgt_wf) = match weather {
            WEATHER_BLIZZARD => (0.115 * (0.7 + 0.3 * wind_str), 620.0),
            WEATHER_SQUALL => (0.085 * (0.7 + 0.3 * wind_str), 880.0),
            WEATHER_EMBER => (0.095 * (0.7 + 0.3 * wind_str), 145.0),
            _ => (0.036 * (0.7 + 0.3 * wind_str), 270.0),
        };
        self.wind_gain += (tgt_wg - self.wind_gain) * 0.0005;
        self.wind_freq += (tgt_wf - self.wind_freq) * 0.0005;

        let white = self.next_white();
        let pink = self.next_pink(white);

        // 1. Plunging Waterfall Bed (low-pass pink noise + slow surge LFO, stereo panned)
        self.wf_lfo = (self.wf_lfo + 0.24 * self.inv_sr) % 1.0;
        let wf_mod = 0.86 + 0.14 * (self.wf_lfo * TAU).sin();
        let wf_cut = (560.0 + 180.0 * wf_mod) * TAU * self.inv_sr;
        self.wf_lpf += (pink - self.wf_lpf) * wf_cut.clamp(0.01, 0.45);
        let wf_sig = self.wf_lpf * self.wf_gain * 0.24 * wf_mod;
        let (wf_l, wf_r) = equal_power_pan(self.wf_pan);

        // 2. Crackling Wildfire Bed (bandpass roar + stochastic sharp crackle bursts)
        let f_cut = (1180.0 * TAU * self.inv_sr).clamp(0.02, 0.45);
        let f_hp = white - self.fire_lp - 0.65 * self.fire_bp;
        self.fire_bp += f_cut * f_hp;
        self.fire_lp += f_cut * self.fire_bp;
        // Stochastic crackle pop proportional to fire_gain
        let crackle_prob = 0.00035 + 0.0018 * self.fire_gain;
        if self.fire_gain > 0.03 && white.abs() > (1.0 - crackle_prob) {
            self.crackle_env = (self.crackle_env + 0.65).min(1.0);
        }
        self.crackle_env *= 0.986;
        self.crackle_phase = (self.crackle_phase + 2950.0 * self.inv_sr) % 1.0;
        let crackle = (self.crackle_phase * TAU).sin() * self.crackle_env * 0.35;
        let fire_sig = (self.fire_bp * 0.32 + crackle) * self.fire_gain * 0.24;
        let (fi_l, fi_r) = equal_power_pan(self.fire_pan);

        // 3. Dynamic Weather Wind Whoosh (dual-LFO resonant bandpass + Ember sub-rumble)
        self.wind_lfo1 = (self.wind_lfo1 + 0.14 * self.inv_sr) % 1.0;
        self.wind_lfo2 = (self.wind_lfo2 + 0.37 * self.inv_sr) % 1.0;
        let lfo_mix = 0.65 + 0.25 * (self.wind_lfo1 * TAU).sin() + 0.10 * (self.wind_lfo2 * TAU).cos();
        let w_freq = (self.wind_freq * lfo_mix).clamp(75.0, 2200.0);
        let w_f = (w_freq * TAU * self.inv_sr).clamp(0.005, 0.55);
        let w_q = if weather == WEATHER_BLIZZARD { 0.22 } else { 0.55 };
        let w_hp = pink - self.wind_lp - w_q * self.wind_bp;
        self.wind_bp += w_f * w_hp;
        self.wind_lp += w_f * self.wind_bp;
        // Extra deep volcanic rumble during Ember Dusk
        let ember_cut = (115.0 * TAU * self.inv_sr).clamp(0.004, 0.25);
        self.ember_lp += (pink - self.ember_lp) * ember_cut;
        let wind_sig = if weather == WEATHER_EMBER {
            (self.ember_lp * 1.45 + self.wind_bp * 0.30) * self.wind_gain
        } else {
            (self.wind_bp * 0.72 + self.wind_lp * 0.28) * self.wind_gain
        };
        let wind_pan_lfo = 0.22 * (self.wind_lfo1 * TAU).cos();
        let (wn_l, wn_r) = equal_power_pan(wind_pan_lfo);

        let wind_shelter = 1.0 - self.cave_enclosure * 0.62;
        let mut out_l = wf_sig * wf_l + fire_sig * fi_l + wind_sig * wind_shelter * wn_l;
        let mut out_r = wf_sig * wf_r + fire_sig * fi_r + wind_sig * wind_shelter * wn_r;

        // 4. Spatialized Polyphonic One-Shot SFX & SNR-Filtered Neural-Voice Chirps
        let inv_sr = self.inv_sr;
        for i in 0..MAX_VOICES {
            if !self.voices[i].active {
                continue;
            }
            let noise = self.next_white();
            let v = &mut self.voices[i];
            let t = v.age;
            let norm = t / v.duration.max(0.001);
            if norm >= 1.0 {
                v.active = false;
                continue;
            }
            let sig = match v.kind {
                SfxKind::FireCast => {
                    // Low whoosh + warm thump + sharp crackle
                    let env = (t * 65.0).min(1.0) * (1.0 - norm).powf(1.8);
                    let freq = 480.0 - 330.0 * norm;
                    let fc = (freq * TAU * inv_sr).clamp(0.01, 0.45);
                    let hp = noise - v.bp_lp - 0.55 * v.bp_bp;
                    v.bp_bp += fc * hp;
                    v.bp_lp += fc * v.bp_bp;
                    v.phase0 = (v.phase0 + (125.0 - 55.0 * norm) * inv_sr) % 1.0;
                    let thump = (v.phase0 * TAU).sin() * (1.0 - norm).powf(3.0) * 0.45;
                    let crack = if noise.abs() > 0.86 { noise * 0.35 } else { 0.0 };
                    (v.bp_bp * 0.85 + thump + crack) * env * 0.42
                }
                SfxKind::FrostCast => {
                    // Crystalline harmonic ice chime (bell partials)
                    let pitch_mul = 0.94 + (v.param as f32 % 5.0) * 0.03;
                    let env = (t * 140.0).min(1.0) * (-t * 9.5).exp();
                    v.phase0 = (v.phase0 + 1180.0 * pitch_mul * inv_sr) % 1.0;
                    v.phase1 = (v.phase1 + 1770.0 * pitch_mul * inv_sr) % 1.0;
                    v.phase2 = (v.phase2 + 2665.0 * pitch_mul * inv_sr) % 1.0;
                    let chime = (v.phase0 * TAU).sin() * 0.50
                        + (v.phase1 * TAU).sin() * 0.32
                        + (v.phase2 * TAU).sin() * 0.18 * (-t * 14.0).exp();
                    chime * env * 0.30
                }
                SfxKind::OilCast => {
                    // Viscous wet squelch (swept resonant bandpass + low bubble pitch bend)
                    let env = (t * 80.0).min(1.0) * (1.0 - norm).powf(1.5);
                    let sweep = 240.0 + 460.0 * (norm * std::f32::consts::PI).sin();
                    let fc = (sweep * TAU * inv_sr).clamp(0.01, 0.45);
                    let hp = noise - v.bp_lp - 0.32 * v.bp_bp;
                    v.bp_bp += fc * hp;
                    v.bp_lp += fc * v.bp_bp;
                    v.phase0 = (v.phase0 + (175.0 - 80.0 * norm) * inv_sr) % 1.0;
                    let bub = (v.phase0 * TAU).sin() * 0.45;
                    (v.bp_bp * 0.72 + bub) * env * 0.36
                }
                SfxKind::SteamCombo => {
                    // Resonant deep boom + pressurized scalding steam hiss
                    let boom_env = (t * 120.0).min(1.0) * (-t * 8.0).exp();
                    let boom_f = 130.0 * (-t * 5.0).exp() + 42.0;
                    v.phase0 = (v.phase0 + boom_f * inv_sr) % 1.0;
                    let boom = (v.phase0 * TAU).sin() * boom_env * 0.68;

                    let hiss_env = (t * 35.0).min(1.0) * (1.0 - norm).powf(1.4);
                    let fc = (2400.0 * TAU * inv_sr).clamp(0.05, 0.55);
                    let hp = noise - v.bp_lp - 0.65 * v.bp_bp;
                    v.bp_bp += fc * hp;
                    v.bp_lp += fc * v.bp_bp;
                    let hiss = (v.bp_bp + hp * 0.35) * hiss_env * 0.36;
                    (boom + hiss) * 0.48
                }
                SfxKind::KillChord => {
                    // Crisp arpeggiated heroic triad (C5 -> E5 -> G5 + shimmer)
                    let env0 = (t * 120.0).min(1.0) * (-t * 5.5).exp();
                    let t1 = (t - 0.045).max(0.0);
                    let env1 = if t >= 0.045 { (t1 * 120.0).min(1.0) * (-t1 * 5.2).exp() } else { 0.0 };
                    let t2 = (t - 0.095).max(0.0);
                    let env2 = if t >= 0.095 { (t2 * 120.0).min(1.0) * (-t2 * 4.8).exp() } else { 0.0 };
                    v.phase0 = (v.phase0 + 523.25 * inv_sr) % 1.0;
                    v.phase1 = (v.phase1 + 659.25 * inv_sr) % 1.0;
                    v.phase2 = (v.phase2 + 783.99 * inv_sr) % 1.0;
                    let s0 = (v.phase0 * TAU).sin() * env0;
                    let s1 = (v.phase1 * TAU).sin() * env1;
                    let s2 = (v.phase2 * TAU).sin() * env2;
                    (s0 * 0.38 + s1 * 0.34 + s2 * 0.36) * 0.32
                }
                SfxKind::ShrineOrRelic => {
                    // Resonant fifth horn/bell swell
                    let env = (t * 18.0).min(1.0) * (1.0 - norm).powf(1.3);
                    v.phase0 = (v.phase0 + 329.63 * inv_sr) % 1.0;
                    v.phase1 = (v.phase1 + 493.88 * inv_sr) % 1.0;
                    v.phase2 = (v.phase2 + 659.25 * inv_sr) % 1.0;
                    let horn = (v.phase0 * TAU).sin() * 0.50
                        + (v.phase1 * TAU).sin() * 0.32
                        + (v.phase2 * TAU).sin() * 0.18;
                    horn * env * 0.30
                }
                SfxKind::VoiceChirp => {
                    // 2-to-3-tone procedural neural-voice synth chirp filtered by AcousticField3D snr_db!
                    // SNR >= +10 dB -> crystal-clear resonant formant chime
                    // SNR < +2 dB   -> band-pass muffled radio-static burst
                    let snr_db = v.extra;
                    let clarity = ((snr_db - 1.5) / 8.5).clamp(0.0, 1.0);
                    let sym = v.param;
                    let num_tones: usize = if (sym & 1) == 0 { 3 } else { 2 };
                    let tone_len = v.duration / (num_tones as f32);
                    let tone_idx = ((t / tone_len) as usize).min(num_tones - 1);
                    let tone_t = (t - tone_idx as f32 * tone_len) / tone_len;
                    let syl_env = (tone_t * 12.0).min(1.0) * ((1.0 - tone_t) * 8.0).min(1.0);

                    // Distinct pentatonic/formant contour per symbol_id
                    let base_hz = 290.0 + ((sym.wrapping_mul(37)) % 7) as f32 * 34.0;
                    let step_ratios: [[f32; 3]; 4] = [
                        [1.00, 1.25, 1.50], // rising alert
                        [1.33, 1.12, 1.00], // descending confirmation
                        [1.00, 1.50, 1.25], // tactical callout
                        [1.12, 1.33, 1.66], // urgent high call
                    ];
                    let ratio = step_ratios[(sym as usize) & 3][tone_idx];
                    let f0 = base_hz * ratio * (1.0 + 0.04 * (tone_t * std::f32::consts::PI).sin());

                    v.phase0 = (v.phase0 + f0 * inv_sr) % 1.0;
                    v.phase1 = (v.phase1 + f0 * 2.0 * inv_sr) % 1.0;
                    v.phase2 = (v.phase2 + f0 * 3.0 * inv_sr) % 1.0;
                    let clear_vocal = ((v.phase0 * TAU).sin() * 0.56
                        + (v.phase1 * TAU).sin() * 0.30
                        + (v.phase2 * TAU).sin() * 0.14)
                        * syl_env;

                    // Bandpass filter for muffled low-SNR radio voice + static burst
                    let muffle_fc = ((580.0 + clarity * 1650.0) * TAU * inv_sr).clamp(0.02, 0.48);
                    let static_mix = (1.0 - clarity).powf(1.3);
                    let radio_in = clear_vocal * (0.45 + 0.55 * clarity) + noise * static_mix * 0.42 * syl_env;
                    let hp = radio_in - v.bp_lp - (0.35 + 0.35 * clarity) * v.bp_bp;
                    v.bp_bp += muffle_fc * hp;
                    v.bp_lp += muffle_fc * v.bp_bp;

                    let blended = clear_vocal * clarity + v.bp_bp * (1.0 - clarity) * 1.45;
                    blended * 0.34
                }
            };

            v.age += inv_sr;
            let scaled = sig * v.gain;
            out_l += scaled * v.pan_l;
            out_r += scaled * v.pan_r;
        }

        // 5. Physical 3D Schroeder Cave Reverb (mixed cleanly at `wet = enclosure * 0.42`)
        let (rev_l, rev_r) = self.cave_reverb.process(out_l, out_r);
        let wet = self.cave_enclosure * 0.42;
        let dry = 1.0 - wet * 0.22;
        out_l = out_l * dry + rev_l * wet;
        out_r = out_r * dry + rev_r * wet;

        // Master gain + smooth soft limiter (`tanh`)
        let l = (out_l * self.master_gain).tanh();
        let r = (out_r * self.master_gain).tanh();
        (l, r)
    }

    fn render_interleaved_f32(&mut self, out: &mut [f32], channels: usize, shared: &AudioShared) {
        self.drain_queue(shared);
        for frame in out.chunks_mut(channels) {
            let (l, r) = self.step_stereo(shared);
            if channels == 1 {
                frame[0] = (l + r) * 0.5;
            } else {
                frame[0] = l;
                frame[1] = r;
                for extra in frame.iter_mut().skip(2) {
                    *extra = 0.0;
                }
            }
        }
    }

    fn render_interleaved_i16(&mut self, out: &mut [i16], channels: usize, shared: &AudioShared) {
        self.drain_queue(shared);
        for frame in out.chunks_mut(channels) {
            let (l, r) = self.step_stereo(shared);
            if channels == 1 {
                frame[0] = (((l + r) * 0.5) * i16::MAX as f32) as i16;
            } else {
                frame[0] = (l * i16::MAX as f32) as i16;
                frame[1] = (r * i16::MAX as f32) as i16;
                for extra in frame.iter_mut().skip(2) {
                    *extra = 0;
                }
            }
        }
    }

    fn render_interleaved_u16(&mut self, out: &mut [u16], channels: usize, shared: &AudioShared) {
        self.drain_queue(shared);
        for frame in out.chunks_mut(channels) {
            let (l, r) = self.step_stereo(shared);
            let to_u16 = |s: f32| -> u16 { ((s * 0.5 + 0.5).clamp(0.0, 1.0) * u16::MAX as f32) as u16 };
            if channels == 1 {
                frame[0] = to_u16((l + r) * 0.5);
            } else {
                frame[0] = to_u16(l);
                frame[1] = to_u16(r);
                for extra in frame.iter_mut().skip(2) {
                    *extra = 32768;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::z16_world::Z16World;

    #[test]
    fn test_cave_enclosure_and_schroeder_reverb_finite() {
        // 1. Verify 3D cave enclosure: > 0.7 under a stone roof vs < 0.1 under open sky
        let mut world = Z16World::new(32, 32, 16);
        let (w, d) = (world.width, world.depth);
        for iz in 4..20 {
            for ix in 4..20 {
                // Solid floor at iy=0..2 and solid vaulted roof at iy=6..10
                for iy in [0usize, 1, 6, 7, 8, 9] {
                    let off = ((iz * d + iy) * w + ix) * 4;
                    world.density_mat[off] = 255;
                }
            }
        }
        let enc_roof = world.sample_cave_enclosure(12.0, 12.0, 2.5);
        let enc_sky = world.sample_cave_enclosure(26.0, 26.0, 2.5);
        assert!(
            enc_roof > 0.7,
            "Expected cave enclosure > 0.7 under solid roof, got {enc_roof}"
        );
        assert!(
            enc_sky < 0.1,
            "Expected cave enclosure < 0.1 under open sky, got {enc_sky}"
        );

        // 2. Verify pure-Rust Schroeder Cave Reverb produces finite, non-NaN stereo reverberation
        let mut reverb = SchroederCaveReverb::new();
        let mut tail_energy = 0.0f32;
        for i in 0..8192 {
            let (in_l, in_r) = if i == 0 {
                (1.0, 0.75)
            } else if i == 120 {
                (-0.5, 0.9)
            } else {
                (0.0, 0.0)
            };
            let (out_l, out_r) = reverb.process(in_l, in_r);
            assert!(out_l.is_finite() && out_r.is_finite(), "Reverb produced non-finite sample at {i}");
            assert!(out_l.abs() < 4.0 && out_r.abs() < 4.0, "Reverb blew up at {i}: ({out_l}, {out_r})");
            if i > 600 {
                tail_energy += out_l * out_l + out_r * out_r;
            }
        }
        assert!(tail_energy > 0.01, "Expected audible cave reverb tail energy, got {tail_energy}");
    }
}
