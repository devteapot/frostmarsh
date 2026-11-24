//! Wave 22B: 100-Client Real Concurrent QUIC + RFC 6455 WebSocket MMO Load Tester (`mmo-loadtest`).
//!
//! Spawns `N` real concurrent QUIC (`quinn`) players streaming 10 Hz movement/aim inputs and receiving
//! `VolumetricBrick3D` / `VolumetricBrick3DDelta` / `EntitySnapshot` datagrams alongside `M` real
//! concurrent RFC 6455 WebSocket browser clients receiving live `0x01` brick & `0x02` entity frames.

use bytes::Bytes;
use clap::Parser;
use frostmarsh_proto::{
    decode_server_datagram, encode_msg, ClientInputDatagram, ClientReliable, ServerDatagram,
};
use quinn::{ClientConfig, Endpoint};
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{DigitallySignedStruct, Error as TlsError, SignatureScheme};
use std::fs;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

#[derive(Parser, Debug)]
#[command(
    name = "mmo-loadtest",
    about = "Real Concurrent QUIC + RFC 6455 WebSocket MMO Load Tester for Frostmarsh"
)]
struct Args {
    /// Target Frostmarsh QUIC server address (e.g. 127.0.0.1:4433)
    #[arg(long, default_value = "127.0.0.1:4433")]
    quic_addr: SocketAddr,
    /// Target Frostmarsh RFC 6455 WebSocket bridge address (e.g. 127.0.0.1:4434)
    #[arg(long, default_value = "127.0.0.1:4434")]
    ws_addr: String,
    /// Number of concurrent real QUIC player clients
    #[arg(long, default_value_t = 50)]
    quic_clients: usize,
    /// Number of concurrent real RFC 6455 WebSocket clients
    #[arg(long, default_value_t = 50)]
    ws_clients: usize,
    /// Test observation duration in seconds after clients start
    #[arg(long, default_value_t = 12)]
    duration_secs: u64,
    /// Optional path to write structured JSON load-test summary
    #[arg(long)]
    out_json: Option<PathBuf>,
}

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

#[derive(Default, Debug)]
struct QuicClientMetrics {
    connected: bool,
    bytes_rx: u64,
    bricks_full: u64,
    bricks_delta: u64,
    chunks_aplus: u64,
    snapshots_rx: u64,
    snap_intervals_ms: Vec<f64>,
    max_players_in_snap: usize,
    last_active_bots: u16,
    last_server_wire_kbps: f32,
    elapsed_secs: f64,
}

#[derive(Default, Debug)]
struct WsClientMetrics {
    connected: bool,
    bytes_rx: u64,
    bricks_full: u64,
    bricks_delta: u64,
    entity_frames_rx: u64,
    max_online_players: usize,
    elapsed_secs: f64,
}

fn build_quic_client_config() -> Result<ClientConfig, Box<dyn std::error::Error>> {
    let crypto = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(SkipServerVerification))
        .with_no_client_auth();
    let quic_crypto = quinn::crypto::rustls::QuicClientConfig::try_from(crypto)?;
    let mut cfg = ClientConfig::new(Arc::new(quic_crypto));
    let mut transport = quinn::TransportConfig::default();
    transport.max_idle_timeout(Some(Duration::from_secs(30).try_into()?));
    transport.keep_alive_interval(Some(Duration::from_secs(5)));
    transport.datagram_receive_buffer_size(Some(4 * 1024 * 1024));
    transport.datagram_send_buffer_size(1024 * 1024);
    cfg.transport_config(Arc::new(transport));
    Ok(cfg)
}

fn record_server_datagram(
    m: &mut QuicClientMetrics,
    raw_len: u64,
    pkt: ServerDatagram,
    last_snap_at: &mut Option<Instant>,
) {
    m.bytes_rx += raw_len;
    match pkt {
        ServerDatagram::VolumetricBrick3D(_)
        | ServerDatagram::VolumetricSubBrick3D { .. }
        | ServerDatagram::PlanetFaceBrick3D { .. } => {
            m.bricks_full += 1;
        }
        ServerDatagram::VolumetricBrick3DDelta { .. }
        | ServerDatagram::VolumetricSubBrick3DDelta { .. }
        | ServerDatagram::PlanetFaceBrick3DDelta { .. } => {
            m.bricks_delta += 1;
        }
        ServerDatagram::APlusChunk { .. } => {
            m.chunks_aplus += 1;
        }
        ServerDatagram::EntitySnapshot {
            players,
            telemetry,
            ..
        } => {
            let now = Instant::now();
            if let Some(prev) = *last_snap_at {
                m.snap_intervals_ms.push(now.duration_since(prev).as_secs_f64() * 1000.0);
            }
            *last_snap_at = Some(now);
            m.snapshots_rx += 1;
            m.max_players_in_snap = m.max_players_in_snap.max(players.len());
            m.last_active_bots = telemetry.active_bots;
            m.last_server_wire_kbps = telemetry.wire_kbps;
        }
    }
}

async fn run_quic_client(
    idx: usize,
    addr: SocketAddr,
    client_cfg: ClientConfig,
    deadline: tokio::time::Instant,
) -> QuicClientMetrics {
    let mut metrics = QuicClientMetrics::default();
    let Ok(mut endpoint) = Endpoint::client("0.0.0.0:0".parse().unwrap()) else {
        return metrics;
    };
    endpoint.set_default_client_config(client_cfg);

    let connect_fut = async {
        let connecting = endpoint.connect(addr, "localhost").ok()?;
        connecting.await.ok()
    };
    let conn = match tokio::time::timeout(Duration::from_secs(8), connect_fut).await {
        Ok(Some(c)) => c,
        _ => return metrics,
    };

    let t_start = Instant::now();
    metrics.connected = true;

    // Send Join + RequestFullSync over reliable uni streams
    for msg in [
        ClientReliable::Join {
            name: format!("LoadQ-{:03}", idx),
            team: (idx % 2) as u8,
        },
        ClientReliable::RequestFullSync,
    ] {
        if let Ok(mut send) = conn.open_uni().await {
            let raw = encode_msg(&msg);
            let _ = send.write_all(&raw).await;
            let _ = send.finish();
        }
    }

    // 10 Hz realistic movement/aim input stream
    let conn_in = conn.clone();
    let input_task = tokio::spawn(async move {
        let mut seq = 0u32;
        let mut interval = tokio::time::interval(Duration::from_millis(100));
        let phase = (idx as f32) * 0.37;
        while tokio::time::Instant::now() < deadline {
            interval.tick().await;
            seq = seq.wrapping_add(1);
            let t = (seq as f32) * 0.1 + phase;
            let inp = ClientInputDatagram {
                seq,
                move_x: t.cos(),
                move_y: t.sin(),
                aim_x: (t * 0.7).cos(),
                aim_y: (t * 0.7).sin(),
                dash: seq % 45 == (idx as u32 % 45),
                melee: false,
                cast_bolt: seq % 25 == (idx as u32 % 25),
                bolt_element: (idx % 3) as u8,
                carve_ice: false,
                cast_ability_e: false,
                cast_ultimate_r: false,
                client_time_ms: t_start.elapsed().as_millis() as u32,
            };
            let raw = encode_msg(&inp);
            if conn_in.send_datagram(Bytes::from(raw)).is_err() {
                break;
            }
        }
    });

    // Concurrently receive datagrams and 0xFE-framed uni-streams until deadline
    let mut last_snap_at: Option<Instant> = None;
    loop {
        let now = tokio::time::Instant::now();
        if now >= deadline {
            break;
        }
        let rem = deadline - now;
        tokio::select! {
            _ = tokio::time::sleep(rem) => break,
            dg = conn.read_datagram() => {
                match dg {
                    Ok(bytes) => {
                        let raw_len = bytes.len() as u64;
                        if let Some(pkt) = decode_server_datagram(&bytes) {
                            record_server_datagram(&mut metrics, raw_len, pkt, &mut last_snap_at);
                        }
                    }
                    Err(_) => break,
                }
            }
            uni = conn.accept_uni() => {
                match uni {
                    Ok(mut recv) => {
                        if let Ok(Ok(bytes)) = tokio::time::timeout(rem, recv.read_to_end(524_288)).await {
                            if bytes.first() == Some(&0xFE) {
                                let raw_len = bytes.len() as u64;
                                if let Some(pkt) = decode_server_datagram(&bytes[1..]) {
                                    record_server_datagram(&mut metrics, raw_len, pkt, &mut last_snap_at);
                                }
                            }
                        }
                    }
                    Err(_) => break,
                }
            }
        }
    }

    input_task.abort();
    conn.close(0u32.into(), b"loadtest-done");
    metrics.elapsed_secs = t_start.elapsed().as_secs_f64().max(0.001);
    metrics
}

fn base64_encode_16(input: &[u8; 16]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(24);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(TABLE[((triple >> 18) & 0x3F) as usize] as char);
        out.push(TABLE[((triple >> 12) & 0x3F) as usize] as char);
        out.push(if chunk.len() > 1 {
            TABLE[((triple >> 6) & 0x3F) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            TABLE[(triple & 0x3F) as usize] as char
        } else {
            '='
        });
    }
    out
}

fn encode_masked_ws_frame(opcode: u8, payload: &[u8]) -> Vec<u8> {
    let mask = [0x37u8, 0xFA, 0x21, 0x3D];
    let len = payload.len();
    let mut out = Vec::with_capacity(14 + len);
    out.push(0x80 | (opcode & 0x0F));
    if len < 126 {
        out.push(0x80 | (len as u8));
    } else if len <= u16::MAX as usize {
        out.push(0x80 | 126);
        out.extend_from_slice(&(len as u16).to_be_bytes());
    } else {
        out.push(0x80 | 127);
        out.extend_from_slice(&(len as u64).to_be_bytes());
    }
    out.extend_from_slice(&mask);
    for (i, &b) in payload.iter().enumerate() {
        out.push(b ^ mask[i & 3]);
    }
    out
}

fn parse_online_players(json: &str) -> Option<usize> {
    let needle = "\"online_players\":";
    let pos = json.find(needle)?;
    let rest = json[pos + needle.len()..].trim_start();
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse::<usize>().ok()
}

async fn read_server_ws_frame(stream: &mut TcpStream) -> std::io::Result<Option<(u8, Vec<u8>, usize)>> {
    let mut hdr = [0u8; 2];
    match stream.read_exact(&mut hdr).await {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let opcode = hdr[0] & 0x0F;
    let masked = (hdr[1] & 0x80) != 0;
    let len_code = hdr[1] & 0x7F;
    let mut wire_bytes = 2usize;
    let payload_len = match len_code {
        0..=125 => len_code as usize,
        126 => {
            let mut ext = [0u8; 2];
            stream.read_exact(&mut ext).await?;
            wire_bytes += 2;
            u16::from_be_bytes(ext) as usize
        }
        _ => {
            let mut ext = [0u8; 8];
            stream.read_exact(&mut ext).await?;
            wire_bytes += 8;
            u64::from_be_bytes(ext) as usize
        }
    };
    if payload_len > 4 * 1024 * 1024 {
        return Ok(None);
    }
    let mut mask = [0u8; 4];
    if masked {
        stream.read_exact(&mut mask).await?;
        wire_bytes += 4;
    }
    let mut payload = vec![0u8; payload_len];
    if payload_len > 0 {
        stream.read_exact(&mut payload).await?;
        wire_bytes += payload_len;
        if masked {
            for (i, b) in payload.iter_mut().enumerate() {
                *b ^= mask[i & 3];
            }
        }
    }
    Ok(Some((opcode, payload, wire_bytes)))
}

async fn run_ws_client(
    idx: usize,
    ws_addr: String,
    deadline: tokio::time::Instant,
) -> WsClientMetrics {
    let mut metrics = WsClientMetrics::default();
    let Ok(Ok(mut stream)) = tokio::time::timeout(Duration::from_secs(8), TcpStream::connect(&ws_addr)).await else {
        return metrics;
    };
    let _ = stream.set_nodelay(true);

    let mut nonce = [0u8; 16];
    for (i, b) in nonce.iter_mut().enumerate() {
        *b = ((idx.wrapping_mul(31) + i.wrapping_mul(17) + 7) & 0xFF) as u8;
    }
    let ws_key = base64_encode_16(&nonce);
    let req = format!(
        "GET / HTTP/1.1\r\nHost: {}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {}\r\nSec-WebSocket-Version: 13\r\n\r\n",
        ws_addr, ws_key
    );
    if stream.write_all(req.as_bytes()).await.is_err() {
        return metrics;
    }

    // Read HTTP 101 response header
    let mut hdr_buf = Vec::with_capacity(512);
    let mut byte = [0u8; 1];
    let handshake_ok = loop {
        let now = tokio::time::Instant::now();
        if now >= deadline || hdr_buf.len() > 8192 {
            break false;
        }
        match tokio::time::timeout(Duration::from_secs(5), stream.read_exact(&mut byte)).await {
            Ok(Ok(_)) => {
                hdr_buf.push(byte[0]);
                if hdr_buf.ends_with(b"\r\n\r\n") {
                    let s = String::from_utf8_lossy(&hdr_buf);
                    break s.starts_with("HTTP/1.1 101");
                }
            }
            _ => break false,
        }
    };
    if !handshake_ok {
        return metrics;
    }

    let t_start = Instant::now();
    metrics.connected = true;

    loop {
        let now = tokio::time::Instant::now();
        if now >= deadline {
            break;
        }
        let rem = deadline - now;
        match tokio::time::timeout(rem, read_server_ws_frame(&mut stream)).await {
            Ok(Ok(Some((opcode, payload, wire_bytes)))) => {
                metrics.bytes_rx += wire_bytes as u64;
                match opcode {
                    0x02 => {
                        match payload.first().copied() {
                            Some(0x01) => {
                                metrics.bricks_full += 1;
                            }
                            Some(0x03) => {
                                metrics.bricks_delta += 1;
                            }
                            Some(0x02) => {
                                metrics.entity_frames_rx += 1;
                                if let Ok(json) = std::str::from_utf8(&payload[1..]) {
                                    if let Some(n) = parse_online_players(json) {
                                        metrics.max_online_players = metrics.max_online_players.max(n);
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    0x09 => {
                        let pong = encode_masked_ws_frame(0x0A, &payload);
                        let _ = stream.write_all(&pong).await;
                    }
                    0x08 => break,
                    _ => {}
                }
            }
            _ => break,
        }
    }

    let close = encode_masked_ws_frame(0x08, &[]);
    let _ = stream.write_all(&close).await;
    metrics.elapsed_secs = t_start.elapsed().as_secs_f64().max(0.001);
    metrics
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (((sorted.len() - 1) as f64) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = rustls::crypto::ring::default_provider().install_default();
    let args = Args::parse();
    let client_cfg = build_quic_client_config()?;
    let duration = Duration::from_secs(args.duration_secs.max(1));
    let t_global = Instant::now();
    let deadline = tokio::time::Instant::now() + duration;

    println!(
        "[mmo-loadtest] Launching {} QUIC ({}) + {} WS ({}) clients for {}s...",
        args.quic_clients, args.quic_addr, args.ws_clients, args.ws_addr, args.duration_secs
    );

    let mut quic_handles = Vec::with_capacity(args.quic_clients);
    for idx in 0..args.quic_clients {
        let cfg = client_cfg.clone();
        let addr = args.quic_addr;
        quic_handles.push(tokio::spawn(async move {
            // Mild 4 ms ramp stagger so 100 handshakes don't overflow kernel UDP/TCP listen backlog
            tokio::time::sleep(Duration::from_millis((idx as u64) * 4)).await;
            run_quic_client(idx, addr, cfg, deadline).await
        }));
    }

    let mut ws_handles = Vec::with_capacity(args.ws_clients);
    for idx in 0..args.ws_clients {
        let ws_addr = args.ws_addr.clone();
        ws_handles.push(tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis((idx as u64) * 4)).await;
            run_ws_client(idx, ws_addr, deadline).await
        }));
    }

    let mut quic_results = Vec::with_capacity(args.quic_clients);
    for h in quic_handles {
        if let Ok(m) = h.await {
            quic_results.push(m);
        }
    }
    let mut ws_results = Vec::with_capacity(args.ws_clients);
    for h in ws_handles {
        if let Ok(m) = h.await {
            ws_results.push(m);
        }
    }

    let wall_secs = t_global.elapsed().as_secs_f64().max(0.001);

    let quic_ok = quic_results.iter().filter(|m| m.connected).count();
    let quic_bytes_total: u64 = quic_results.iter().map(|m| m.bytes_rx).sum();
    let quic_bricks_full: u64 = quic_results.iter().map(|m| m.bricks_full).sum();
    let quic_bricks_delta: u64 = quic_results.iter().map(|m| m.bricks_delta).sum();
    let quic_chunks_aplus: u64 = quic_results.iter().map(|m| m.chunks_aplus).sum();
    let quic_snaps_total: u64 = quic_results.iter().map(|m| m.snapshots_rx).sum();
    let quic_max_players_in_snap = quic_results
        .iter()
        .map(|m| m.max_players_in_snap)
        .max()
        .unwrap_or(0);
    let quic_last_active_bots = quic_results
        .iter()
        .map(|m| m.last_active_bots)
        .max()
        .unwrap_or(0);
    let quic_per_client_kbps = if quic_ok > 0 {
        quic_results
            .iter()
            .filter(|m| m.connected)
            .map(|m| (m.bytes_rx as f64 / 1024.0) / m.elapsed_secs)
            .sum::<f64>()
            / (quic_ok as f64)
    } else {
        0.0
    };
    let quic_total_kbps = (quic_bytes_total as f64 / 1024.0) / wall_secs;

    let mut all_intervals: Vec<f64> = quic_results
        .iter()
        .flat_map(|m| m.snap_intervals_ms.iter().copied())
        .collect();
    all_intervals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let snap_p50_ms = percentile(&all_intervals, 0.50);
    let snap_p99_ms = percentile(&all_intervals, 0.99);

    let ws_ok = ws_results.iter().filter(|m| m.connected).count();
    let ws_bytes_total: u64 = ws_results.iter().map(|m| m.bytes_rx).sum();
    let ws_bricks_full: u64 = ws_results.iter().map(|m| m.bricks_full).sum();
    let ws_bricks_delta: u64 = ws_results.iter().map(|m| m.bricks_delta).sum();
    let ws_bricks_total = ws_bricks_full + ws_bricks_delta;
    let ws_delta_share_pct = if ws_bricks_total > 0 {
        (ws_bricks_delta as f64 * 100.0) / (ws_bricks_total as f64)
    } else {
        0.0
    };
    let ws_entities_total: u64 = ws_results.iter().map(|m| m.entity_frames_rx).sum();
    let ws_max_online_players = ws_results
        .iter()
        .map(|m| m.max_online_players)
        .max()
        .unwrap_or(0);
    let ws_per_client_kbps = if ws_ok > 0 {
        ws_results
            .iter()
            .filter(|m| m.connected)
            .map(|m| (m.bytes_rx as f64 / 1024.0) / m.elapsed_secs)
            .sum::<f64>()
            / (ws_ok as f64)
    } else {
        0.0
    };
    let ws_total_kbps = (ws_bytes_total as f64 / 1024.0) / wall_secs;

    let total_bricks_quic = quic_bricks_full + quic_bricks_delta;
    let delta_share_pct = if total_bricks_quic > 0 {
        (quic_bricks_delta as f64 * 100.0) / (total_bricks_quic as f64)
    } else {
        0.0
    };

    println!("\n+------------------------+-------------------+-------------------+");
    println!("| Metric                 | QUIC Clients      | WebSocket Clients |");
    println!("+------------------------+-------------------+-------------------+");
    println!(
        "| Connected / Requested  | {:>7} / {:<7} | {:>7} / {:<7} |",
        quic_ok, args.quic_clients, ws_ok, args.ws_clients
    );
    println!(
        "| Avg Per-Client KB/s    | {:>12.1} KB/s | {:>12.1} KB/s |",
        quic_per_client_kbps, ws_per_client_kbps
    );
    println!(
        "| Aggregate RX KB/s      | {:>12.1} KB/s | {:>12.1} KB/s |",
        quic_total_kbps, ws_total_kbps
    );
    println!(
        "| Full Bricks RX         | {:>17} | {:>17} |",
        quic_bricks_full, ws_bricks_full
    );
    println!(
        "| Delta Bricks RX        | {:>17} | {:>17} |",
        quic_bricks_delta, ws_bricks_delta
    );
    println!(
        "| Delta Brick Share      | {:>16.1}% | {:>16.1}% |",
        delta_share_pct, ws_delta_share_pct
    );
    println!(
        "| Entity Snapshots RX    | {:>17} | {:>17} |",
        quic_snaps_total, ws_entities_total
    );
    println!(
        "| Snap Inter-Arrival p50 | {:>14.2} ms | {:>17} |",
        snap_p50_ms, "-"
    );
    println!(
        "| Snap Inter-Arrival p99 | {:>14.2} ms | {:>17} |",
        snap_p99_ms, "-"
    );
    println!(
        "| Max Players Visible    | {:>17} | {:>17} |",
        quic_max_players_in_snap, ws_max_online_players
    );
    println!("+------------------------+-------------------+-------------------+");
    println!(
        "Total Connected: {}/{} | Wall: {:.2}s | APlusChunks: {} | Active Bots: {}",
        quic_ok + ws_ok,
        args.quic_clients + args.ws_clients,
        wall_secs,
        quic_chunks_aplus,
        quic_last_active_bots
    );

    if let Some(ref out_path) = args.out_json {
        let json = format!(
            concat!(
                "{{\n",
                "  \"wall_secs\": {:.3},\n",
                "  \"quic\": {{\n",
                "    \"requested\": {},\n",
                "    \"connected\": {},\n",
                "    \"bytes_rx_total\": {},\n",
                "    \"per_client_kbps\": {:.2},\n",
                "    \"total_kbps\": {:.2},\n",
                "    \"bricks_full\": {},\n",
                "    \"bricks_delta\": {},\n",
                "    \"delta_share_pct\": {:.2},\n",
                "    \"chunks_aplus\": {},\n",
                "    \"snapshots_rx\": {},\n",
                "    \"snap_interval_p50_ms\": {:.3},\n",
                "    \"snap_interval_p99_ms\": {:.3},\n",
                "    \"max_players_in_snap\": {},\n",
                "    \"active_bots\": {}\n",
                "  }},\n",
                "  \"ws\": {{\n",
                "    \"requested\": {},\n",
                "    \"connected\": {},\n",
                "    \"bytes_rx_total\": {},\n",
                "    \"per_client_kbps\": {:.2},\n",
                "    \"total_kbps\": {:.2},\n",
                "    \"bricks_rx\": {},\n",
                "    \"bricks_full\": {},\n",
                "    \"bricks_delta\": {},\n",
                "    \"delta_share_pct\": {:.2},\n",
                "    \"entity_frames_rx\": {},\n",
                "    \"max_online_players\": {}\n",
                "  }}\n",
                "}}\n"
            ),
            wall_secs,
            args.quic_clients,
            quic_ok,
            quic_bytes_total,
            quic_per_client_kbps,
            quic_total_kbps,
            quic_bricks_full,
            quic_bricks_delta,
            delta_share_pct,
            quic_chunks_aplus,
            quic_snaps_total,
            snap_p50_ms,
            snap_p99_ms,
            quic_max_players_in_snap,
            quic_last_active_bots,
            args.ws_clients,
            ws_ok,
            ws_bytes_total,
            ws_per_client_kbps,
            ws_total_kbps,
            ws_bricks_total,
            ws_bricks_full,
            ws_bricks_delta,
            ws_delta_share_pct,
            ws_entities_total,
            ws_max_online_players,
        );
        if let Some(parent) = out_path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(out_path, json)?;
        println!("[mmo-loadtest] Wrote JSON summary to {}", out_path.display());
    }

    Ok(())
}
