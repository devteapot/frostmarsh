//! Live 3D Volumetric QUIC (`quinn`) Client for Frostmarsh Unified v3 (`256×256×4` = 262,144 voxels!).
//! Receives sparse layered `APlusChunk { layer, cx, cy, lz4_cells }` across all 4 vertical strata
//! plus 30 Hz 3D `EntitySnapshot`s (with automatic WAN/Tailscale MTU uni-stream fallback).

use frostmarsh_proto::*;
use quinn::{ClientConfig, Connection, Endpoint};
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{DigitallySignedStruct, Error as TlsError, SignatureScheme};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug)]
struct SkipServerVerification;

impl ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, TlsError> {
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, TlsError> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, TlsError> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        vec![
            SignatureScheme::RSA_PKCS1_SHA256,
            SignatureScheme::ECDSA_NISTP256_SHA256,
            SignatureScheme::RSA_PKCS1_SHA384,
            SignatureScheme::ECDSA_NISTP384_SHA384,
            SignatureScheme::RSA_PKCS1_SHA512,
            SignatureScheme::RSA_PSS_SHA256,
            SignatureScheme::RSA_PSS_SHA384,
            SignatureScheme::RSA_PSS_SHA512,
            SignatureScheme::ED25519,
        ]
    }
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct LiveEntityState {
    pub sim_tick: u64,
    pub match_time_secs: u32,
    pub your_player_id: u32,
    pub echo_client_time_ms: u32,
    pub relic_x: f32,
    pub relic_y: f32,
    pub players: Vec<PlayerSnap>,
    pub projectiles: Vec<ProjectileSnap>,
    pub bodies: Vec<CellBodySnap>,
    pub shrines: Vec<ShrineSnap>,
    pub events: Vec<CombatEvent>,
    pub telemetry: TelemetrySnap,
    pub bot_brains: Vec<BotBrainSnap>,
    pub kill_feed: Vec<KillFeedEntry>,
}

pub struct LiveWorldState {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub chunks_x: usize,
    pub chunks_y: usize,
    pub your_player_id: u32,
    /// Layered 4-byte `A+` cells: `aplus_layers[layer][y * width + x]` (defaults to empty air `[0xff, 0, 26, 1]`).
    pub aplus_layers: Vec<Vec<[u8; 4]>>,
    /// Received chunk flags for surface layer (`surf_layer`) to track initial sync progress.
    pub received_surf_chunks: Vec<bool>,
    /// Dirty `(layer, chunk_idx)` flags across all `depth` layers.
    pub dirty_chunks: Vec<Vec<bool>>,
    pub latest_field_tick: u64,
    pub entities: Option<LiveEntityState>,
    pub chunks_rx: u64,
    pub snapshots_rx: u64,
    pub bytes_rx: u64,
    pub start_instant: Instant,
    pub rtt_ms: f32,
}

impl LiveWorldState {
    pub fn new(width: usize, height: usize, depth: usize) -> Self {
        let d = depth.max(1);
        let cx = chunks_for_dim(width);
        let cy = chunks_for_dim(height);
        Self {
            width,
            height,
            depth: d,
            chunks_x: cx,
            chunks_y: cy,
            your_player_id: 0,
            aplus_layers: vec![vec![[0xff, 0, 26, 1]; width * height]; d],
            received_surf_chunks: vec![false; cx * cy],
            dirty_chunks: vec![vec![false; cx * cy]; d],
            latest_field_tick: 0,
            entities: None,
            chunks_rx: 0,
            snapshots_rx: 0,
            bytes_rx: 0,
            start_instant: Instant::now(),
            rtt_ms: 0.0,
        }
    }

    pub fn surf_layer(&self) -> usize {
        if self.depth >= 2 { 1 } else { 0 }
    }

    pub fn resize_if_needed(&mut self, w: usize, h: usize, d: usize) {
        let d = d.max(1);
        if w == 0 || h == 0 || (self.width == w && self.height == h && self.depth == d) {
            return;
        }
        self.width = w;
        self.height = h;
        self.depth = d;
        self.chunks_x = chunks_for_dim(w);
        self.chunks_y = chunks_for_dim(h);
        self.aplus_layers = vec![vec![[0xff, 0, 26, 1]; w * h]; d];
        self.received_surf_chunks = vec![false; self.chunks_x * self.chunks_y];
        self.dirty_chunks = vec![vec![false; self.chunks_x * self.chunks_y]; d];
    }

    pub fn total_surf_chunks(&self) -> usize {
        self.chunks_x * self.chunks_y
    }

    pub fn received_surf_count(&self) -> usize {
        self.received_surf_chunks.iter().filter(|&&b| b).count()
    }

    pub fn take_dirty_list(&mut self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for (l, row) in self.dirty_chunks.iter_mut().enumerate() {
            for (ci, d) in row.iter_mut().enumerate() {
                if *d {
                    *d = false;
                    out.push((l, ci));
                }
            }
        }
        out
    }

    pub fn client_time_ms(&self) -> u32 {
        self.start_instant.elapsed().as_millis() as u32
    }
}

pub struct QuicClientHandle {
    pub state: Arc<Mutex<LiveWorldState>>,
    pub conn: Connection,
    _endpoint: Endpoint,
}

impl QuicClientHandle {
    pub async fn connect(
        addr: SocketAddr,
        name: &str,
        team: u8,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let _ = rustls::crypto::ring::default_provider().install_default();

        let crypto = rustls::ClientConfig::builder()
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(SkipServerVerification))
            .with_no_client_auth();

        let quic_crypto = quinn::crypto::rustls::QuicClientConfig::try_from(crypto)?;
        let mut client_cfg = ClientConfig::new(Arc::new(quic_crypto));
        let mut transport = quinn::TransportConfig::default();
        transport.max_idle_timeout(Some(Duration::from_secs(30).try_into()?));
        transport.keep_alive_interval(Some(Duration::from_secs(5)));
        transport.datagram_receive_buffer_size(Some(12 * 1024 * 1024));
        transport.datagram_send_buffer_size(4 * 1024 * 1024);
        client_cfg.transport_config(Arc::new(transport));

        let mut endpoint = Endpoint::client("0.0.0.0:0".parse()?)?;
        endpoint.set_default_client_config(client_cfg);

        let conn = endpoint.connect(addr, "localhost")?.await?;
        let state = Arc::new(Mutex::new(LiveWorldState::new(GRID_W, GRID_H, GRID_D)));

        Self::send_reliable_on(&conn, &ClientReliable::Join { name: name.to_string(), team }).await?;
        Self::send_reliable_on(&conn, &ClientReliable::RequestFullSync).await?;

        let apply_snapshot = |st: &mut LiveWorldState,
                              raw_len: u64,
                              sim_tick: u64,
                              match_time_secs: u32,
                              your_player_id: u32,
                              echo_client_time_ms: u32,
                              relic_x: f32,
                              relic_y: f32,
                              players: Vec<PlayerSnap>,
                              projectiles: Vec<ProjectileSnap>,
                              bodies: Vec<CellBodySnap>,
                              shrines: Vec<ShrineSnap>,
                              events: Vec<CombatEvent>,
                              telemetry: TelemetrySnap,
                              bot_brains: Vec<BotBrainSnap>,
                              kill_feed: Vec<KillFeedEntry>| {
            st.resize_if_needed(
                telemetry.grid_w as usize,
                telemetry.grid_h as usize,
                telemetry.grid_d as usize,
            );
            st.bytes_rx += raw_len;
            st.snapshots_rx += 1;
            st.your_player_id = your_player_id;
            if echo_client_time_ms > 0 {
                let now_ms = st.client_time_ms();
                let rtt = now_ms.saturating_sub(echo_client_time_ms) as f32;
                if rtt < 2000.0 {
                    st.rtt_ms = if st.rtt_ms <= 0.1 { rtt } else { st.rtt_ms * 0.85 + rtt * 0.15 };
                }
            }
            st.entities = Some(LiveEntityState {
                sim_tick,
                match_time_secs,
                your_player_id,
                echo_client_time_ms,
                relic_x,
                relic_y,
                players,
                projectiles,
                bodies,
                shrines,
                events,
                telemetry,
                bot_brains,
                kill_feed,
            });
        };

        // Reliable uni-stream listener (Welcome + WAN MTU-fallback EntitySnapshots)
        let st_rel = state.clone();
        let conn_rel = conn.clone();
        tokio::spawn(async move {
            while let Ok(mut recv) = conn_rel.accept_uni().await {
                if let Ok(bytes) = recv.read_to_end(65536).await {
                    if bytes.first() == Some(&0xFE) {
                        if let Some(ServerDatagram::EntitySnapshot {
                            sim_tick,
                            match_time_secs,
                            your_player_id,
                            echo_client_time_ms,
                            relic_x,
                            relic_y,
                            players,
                            projectiles,
                            bodies,
                            shrines,
                            events,
                            telemetry,
                            bot_brains,
                            kill_feed,
                        }) = decode_server_datagram(&bytes[1..])
                        {
                            let mut st = st_rel.lock().unwrap();
                            apply_snapshot(
                                &mut st,
                                bytes.len() as u64,
                                sim_tick,
                                match_time_secs,
                                your_player_id,
                                echo_client_time_ms,
                                relic_x,
                                relic_y,
                                players,
                                projectiles,
                                bodies,
                                shrines,
                                events,
                                telemetry,
                                bot_brains,
                                kill_feed,
                            );
                        }
                    } else if let Some(ServerReliable::Welcome {
                        your_player_id,
                        width,
                        height,
                        depth,
                    }) = decode_msg::<ServerReliable>(&bytes)
                    {
                        let mut st = st_rel.lock().unwrap();
                        st.your_player_id = your_player_id;
                        st.resize_if_needed(width as usize, height as usize, depth as usize);
                    }
                }
            }
        });

        // Unreliable datagram reader (3D layered APlusChunk + EntitySnapshot)
        let st_dg = state.clone();
        let conn_dg = conn.clone();
        tokio::spawn(async move {
            while let Ok(bytes) = conn_dg.read_datagram().await {
                let raw_len = bytes.len() as u64;
                let Some(pkt) = decode_server_datagram(&bytes) else {
                    continue;
                };
                match pkt {
                    ServerDatagram::APlusChunk {
                        field_tick,
                        layer,
                        chunk_id: _,
                        cx,
                        cy,
                        lz4_cells,
                    } => {
                        let Some(raw) = decompress_aplus_cells(&lz4_cells) else {
                            continue;
                        };
                        if raw.len() != CHUNK_SIZE * CHUNK_SIZE * 4 {
                            continue;
                        }
                        let mut st = st_dg.lock().unwrap();
                        let l_u = layer as usize;
                        let cx_u = cx as usize;
                        let cy_u = cy as usize;
                        if l_u >= st.depth || cx_u >= st.chunks_x || cy_u >= st.chunks_y {
                            continue;
                        }
                        st.bytes_rx += raw_len;
                        st.chunks_rx += 1;
                        st.latest_field_tick = st.latest_field_tick.max(field_tick);

                        let x0 = cx_u * CHUNK_SIZE;
                        let y0 = cy_u * CHUNK_SIZE;
                        let w = st.width;
                        let h = st.height;
                        let grid = &mut st.aplus_layers[l_u];
                        for dy in 0..CHUNK_SIZE {
                            let gy = (y0 + dy).min(h - 1);
                            for dx in 0..CHUNK_SIZE {
                                let gx = (x0 + dx).min(w - 1);
                                let src = (dy * CHUNK_SIZE + dx) * 4;
                                grid[gy * w + gx] = [raw[src], raw[src + 1], raw[src + 2], raw[src + 3]];
                            }
                        }
                        let ci = cy_u * st.chunks_x + cx_u;
                        if l_u == st.surf_layer() {
                            st.received_surf_chunks[ci] = true;
                        }
                        st.dirty_chunks[l_u][ci] = true;
                    }
                    ServerDatagram::EntitySnapshot {
                        sim_tick,
                        match_time_secs,
                        your_player_id,
                        echo_client_time_ms,
                        relic_x,
                        relic_y,
                        players,
                        projectiles,
                        bodies,
                        shrines,
                        events,
                        telemetry,
                        bot_brains,
                        kill_feed,
                    } => {
                        let mut st = st_dg.lock().unwrap();
                        apply_snapshot(
                            &mut st,
                            raw_len,
                            sim_tick,
                            match_time_secs,
                            your_player_id,
                            echo_client_time_ms,
                            relic_x,
                            relic_y,
                            players,
                            projectiles,
                            bodies,
                            shrines,
                            events,
                            telemetry,
                            bot_brains,
                            kill_feed,
                        );
                    }
                }
            }
        });

        Ok(Self {
            state,
            conn,
            _endpoint: endpoint,
        })
    }

    pub async fn send_reliable_on(
        conn: &Connection,
        msg: &ClientReliable,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut send = conn.open_uni().await?;
        let raw = encode_msg(msg);
        send.write_all(&raw).await?;
        send.finish()?;
        Ok(())
    }

    pub fn send_reliable_detached(&self, msg: ClientReliable) {
        let conn = self.conn.clone();
        tokio::spawn(async move {
            let _ = Self::send_reliable_on(&conn, &msg).await;
        });
    }

    pub fn send_input(&self, inp: &ClientInputDatagram) {
        let raw = encode_msg(inp);
        let _ = self.conn.send_datagram(bytes::Bytes::from(raw));
    }

    pub async fn wait_for_sync(&self, min_frac: f32, max_wait: Duration) -> (usize, usize, bool) {
        let start = Instant::now();
        let mut last_req = Instant::now();
        loop {
            let (rc, tot, has_ent) = {
                let st = self.state.lock().unwrap();
                (st.received_surf_count(), st.total_surf_chunks(), st.entities.is_some())
            };
            let target = ((tot as f32) * min_frac).ceil() as usize;
            if rc >= target && has_ent {
                return (rc, tot, true);
            }
            if start.elapsed() >= max_wait {
                return (rc, tot, has_ent);
            }
            if last_req.elapsed() >= Duration::from_millis(900) && rc < target {
                let _ = Self::send_reliable_on(&self.conn, &ClientReliable::RequestFullSync).await;
                last_req = Instant::now();
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }
}
