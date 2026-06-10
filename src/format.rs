// .hodge binary format
//
// v1 — uncompressed:
//   header + per-frame: gradient[N]f32 + solenoidal[N]f32 + harmonic f32 + xi f32 + sc u8
//   Frame size: 8*N + 9 bytes (N=1024 → 8201 bytes/frame)
//
// v2 — SC-aware compression (symmetric i8, deprecated):
//   SC1: [0x01] harmonic f32 + xi f32 + sc u8          =  10 bytes
//   SC2: [0x02] grad_scaler f32 + grad i8[N] + sol_scaler f32 + sol i8[N] DPCM + meta = 2N+18
//   SC3: [0x03] gradient f32[N] + solenoidal f32[N] + meta = 8N+10
//
// v3 — Phase 8: Asymmetric bit allocation (Shannon-compliant):
//   ∇φ (tonal, 90% energy)  → i16 Block-FP  → ~96 dB SNR
//   δψ (transient, sparse)  → i8  Block-FP  → ~48 dB + sparsity Zstd boost
//   h  (harmonic / DC)      → f32            → lossless
//   SC1: [0x11] harmonic f32 + xi f32 + sc u8                    = 10 bytes
//   SC2: [0x12] g_scaler f32 + grad i16[N]LE + s_scaler f32 + sol i8[N] + meta = 2N*2+N+14
//   SC3: [0x13] gradient f32[N] + solenoidal f32[N] + meta        = 8N+10
//
// Target (v3, 44.1kHz stereo 5min, SC2≈99.5%):
//   Raw data ≈ 34MB → after sparsity threshold Zstd → 8-10MB

use std::io::{self, Read, Write};
use crate::hodge_math::{HodgeFrame, ScClass};

pub const MAGIC: &[u8; 6] = b"HODGE\x01";
pub const VERSION:    u32 = 1;
pub const VERSION_V2: u32 = 2;
pub const VERSION_V3: u32 = 3; // Phase 8: asymmetric i16/i8 allocation

pub const I8_SCALE:  f32 = 127.0;
const I16_SCALE: f32 = 32767.0;
// V4 additions
pub const GAIN_SCALE: f32 = 1.0; // placeholder, to be tuned
pub const V4_SC2_TYPE: u8 = 0x21; // new frame type identifier for V4 SC2

// Metadata block (after fixed 28-byte header):
//   0x4D + u16_le(len) + utf8_bytes  →  author string present
//   0x00                             →  no metadata (legacy files)
const META_MARKER: u8 = 0x4D; // 'M'

#[derive(Debug, Clone)]
pub struct HodgeFileHeader {
    pub version: u32,
    pub sample_rate: u32,
    pub channels: u16,
    pub frame_size: u32,
    pub total_frames: u64,
    /// Optional author/copyright string embedded in header
    pub author: Option<String>,
}

pub fn write_header<W: Write>(w: &mut W, h: &HodgeFileHeader) -> io::Result<()> {
    w.write_all(MAGIC)?;
    w.write_all(&h.version.to_le_bytes())?;
    w.write_all(&h.sample_rate.to_le_bytes())?;
    w.write_all(&h.channels.to_le_bytes())?;
    w.write_all(&h.frame_size.to_le_bytes())?;
    w.write_all(&h.total_frames.to_le_bytes())?;
    match &h.author {
        Some(a) => {
            let bytes = a.as_bytes();
            w.write_all(&[META_MARKER])?;
            w.write_all(&(bytes.len() as u16).to_le_bytes())?;
            w.write_all(bytes)?;
        }
        None => w.write_all(&[0u8])?,
    }
    Ok(())
}

pub fn read_header<R: Read>(r: &mut R) -> io::Result<HodgeFileHeader> {
    let mut magic = [0u8; 6];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Not a .hodge file"));
    }
    let mut buf4 = [0u8; 4]; let mut buf2 = [0u8; 2]; let mut buf8 = [0u8; 8];
    r.read_exact(&mut buf4)?; let version = u32::from_le_bytes(buf4);
    r.read_exact(&mut buf4)?; let sample_rate = u32::from_le_bytes(buf4);
    r.read_exact(&mut buf2)?; let channels = u16::from_le_bytes(buf2);
    r.read_exact(&mut buf4)?; let frame_size = u32::from_le_bytes(buf4);
    r.read_exact(&mut buf8)?; let total_frames = u64::from_le_bytes(buf8);

    // Read optional metadata byte
    let mut meta_flag = [0u8; 1];
    let author = match r.read_exact(&mut meta_flag) {
        Ok(()) if meta_flag[0] == META_MARKER => {
            let mut len_buf = [0u8; 2];
            r.read_exact(&mut len_buf)?;
            let len = u16::from_le_bytes(len_buf) as usize;
            let mut abuf = vec![0u8; len];
            r.read_exact(&mut abuf)?;
            Some(String::from_utf8_lossy(&abuf).into_owned())
        }
        _ => None, // legacy file or no metadata
    };

    Ok(HodgeFileHeader { version, sample_rate, channels, frame_size, total_frames, author })
}

// ── v1: uncompressed ──────────────────────────────────────────────

pub fn write_frame<W: Write>(w: &mut W, f: &HodgeFrame) -> io::Result<()> {
    for &x in &f.gradient   { w.write_all(&x.to_le_bytes())?; }
    for &x in &f.solenoidal { w.write_all(&x.to_le_bytes())?; }
    w.write_all(&f.harmonic.to_le_bytes())?;
    w.write_all(&f.xi.to_le_bytes())?;
    w.write_all(&[f.sc_class as u8])?;
    Ok(())
}

pub fn read_frame<R: Read>(r: &mut R, frame_size: usize) -> io::Result<HodgeFrame> {
    let read_vec_f32 = |r: &mut R, v: &mut Vec<f32>| -> io::Result<()> {
        for x in v.iter_mut() {
            let mut buf = [0u8; 4]; r.read_exact(&mut buf)?; *x = f32::from_le_bytes(buf);
        }
        Ok(())
    };
    let mut gradient = vec![0.0f32; frame_size];
    let mut solenoidal = vec![0.0f32; frame_size];
    read_vec_f32(r, &mut gradient)?;
    read_vec_f32(r, &mut solenoidal)?;
    let mut buf4 = [0u8; 4];
    r.read_exact(&mut buf4)?; let harmonic = f32::from_le_bytes(buf4);
    r.read_exact(&mut buf4)?; let xi = f32::from_le_bytes(buf4);
    let mut sc_byte = [0u8; 1]; r.read_exact(&mut sc_byte)?;
    let sc_class = sc_from_byte(sc_byte[0]);
    Ok(HodgeFrame { gradient, solenoidal, harmonic, xi, sc_class })
}

// ── v2: SC-aware compressed ───────────────────────────────────────

pub fn write_frame_v2<W: Write>(w: &mut W, f: &HodgeFrame) -> io::Result<()> {
    match f.sc_class {
        ScClass::SC1 => {
            w.write_all(&[0x01])?;
            w.write_all(&f.harmonic.to_le_bytes())?;
            w.write_all(&f.xi.to_le_bytes())?;
            w.write_all(&[f.sc_class as u8])?;
        }
        ScClass::SC2 => {
            w.write_all(&[0x02])?;
            // block-FP + DPCM: normalize → quantize → delta-encode
            // DPCM makes tonal gradients ~N(0,7) instead of Uniform(-127,127) → ~3 bits/sym vs 8
            let g_max = f.gradient.iter().map(|x| x.abs()).fold(1e-9f32, f32::max);
            let s_max = f.solenoidal.iter().map(|x| x.abs()).fold(1e-9f32, f32::max);
            w.write_all(&g_max.to_le_bytes())?;
            w.write_all(&s_max.to_le_bytes())?;
            let g_i8: Vec<i8> = f.gradient.iter()
                .map(|&x| ((x / g_max).clamp(-1.0, 1.0) * I8_SCALE) as i8)
                .collect();
            let s_i8: Vec<i8> = f.solenoidal.iter()
                .map(|&x| ((x / s_max).clamp(-1.0, 1.0) * I8_SCALE) as i8)
                .collect();
            // DPCM encode
            let write_dpcm = |w: &mut W, v: &[i8]| -> io::Result<()> {
                if v.is_empty() { return Ok(()); }
                w.write_all(&[v[0] as u8])?;
                for i in 1..v.len() {
                    w.write_all(&[v[i].wrapping_sub(v[i-1]) as u8])?;
                }
                Ok(())
            };
            write_dpcm(w, &g_i8)?;
            write_dpcm(w, &s_i8)?;
            w.write_all(&f.harmonic.to_le_bytes())?;
            w.write_all(&f.xi.to_le_bytes())?;
            w.write_all(&[f.sc_class as u8])?;
        }
        ScClass::SC3 => {
            w.write_all(&[0x03])?;
            for &x in &f.gradient   { w.write_all(&x.to_le_bytes())?; }
            for &x in &f.solenoidal { w.write_all(&x.to_le_bytes())?; }
            w.write_all(&f.harmonic.to_le_bytes())?;
            w.write_all(&f.xi.to_le_bytes())?;
            w.write_all(&[f.sc_class as u8])?;
        }
    }
    Ok(())
}

pub fn read_frame_v2<R: Read>(r: &mut R, frame_size: usize) -> io::Result<HodgeFrame> {
    let mut type_byte = [0u8; 1];
    r.read_exact(&mut type_byte)?;
    let mut buf4 = [0u8; 4];

    match type_byte[0] {
        0x01 => {
            r.read_exact(&mut buf4)?; let harmonic = f32::from_le_bytes(buf4);
            r.read_exact(&mut buf4)?; let xi = f32::from_le_bytes(buf4);
            let mut sc_byte = [0u8; 1]; r.read_exact(&mut sc_byte)?;
            Ok(HodgeFrame {
                gradient:   vec![0.0; frame_size],
                solenoidal: vec![0.0; frame_size],
                harmonic, xi,
                sc_class: sc_from_byte(sc_byte[0]),
            })
        }
        0x02 => {
            // block-FP + DPCM decode
            r.read_exact(&mut buf4)?; let g_max = f32::from_le_bytes(buf4);
            r.read_exact(&mut buf4)?; let s_max = f32::from_le_bytes(buf4);
            let mut g_bytes = vec![0u8; frame_size];
            let mut s_bytes = vec![0u8; frame_size];
            r.read_exact(&mut g_bytes)?;
            r.read_exact(&mut s_bytes)?;
            // DPCM decode: cumulative sum over i8 deltas
            let dpcm_decode = |bytes: &[u8]| -> Vec<i8> {
                let mut out = vec![0i8; bytes.len()];
                if bytes.is_empty() { return out; }
                out[0] = bytes[0] as i8;
                for i in 1..bytes.len() {
                    out[i] = out[i-1].wrapping_add(bytes[i] as i8);
                }
                out
            };
            let gradient:   Vec<f32> = dpcm_decode(&g_bytes).iter().map(|&b| b as f32 / I8_SCALE * g_max).collect();
            let solenoidal: Vec<f32> = dpcm_decode(&s_bytes).iter().map(|&b| b as f32 / I8_SCALE * s_max).collect();
            r.read_exact(&mut buf4)?; let harmonic = f32::from_le_bytes(buf4);
            r.read_exact(&mut buf4)?; let xi = f32::from_le_bytes(buf4);
            let mut sc_byte = [0u8; 1]; r.read_exact(&mut sc_byte)?;
            Ok(HodgeFrame { gradient, solenoidal, harmonic, xi, sc_class: sc_from_byte(sc_byte[0]) })
        }
        0x03 => {
            let read_f32_vec = |r: &mut R| -> io::Result<Vec<f32>> {
                let mut v = vec![0.0f32; frame_size];
                for x in v.iter_mut() {
                    let mut b = [0u8; 4]; r.read_exact(&mut b)?; *x = f32::from_le_bytes(b);
                }
                Ok(v)
            };
            let gradient   = read_f32_vec(r)?;
            let solenoidal = read_f32_vec(r)?;
            r.read_exact(&mut buf4)?; let harmonic = f32::from_le_bytes(buf4);
            r.read_exact(&mut buf4)?; let xi = f32::from_le_bytes(buf4);
            let mut sc_byte = [0u8; 1]; r.read_exact(&mut sc_byte)?;
            Ok(HodgeFrame { gradient, solenoidal, harmonic, xi, sc_class: sc_from_byte(sc_byte[0]) })
        }
        b => Err(io::Error::new(io::ErrorKind::InvalidData, format!("unknown frame type 0x{b:02x}")))
    }
}

fn sc_from_byte(b: u8) -> ScClass {
    match b { 1 => ScClass::SC1, 3 => ScClass::SC3, _ => ScClass::SC2 }
}

// ── v3: Asymmetric i16/i8 Block-FP (Phase 8) ─────────────────────

/// Write a v3 frame with asymmetric quantization:
///   ∇φ → i16 Block-FP  (96 dB, tonal carrier)
///   δψ → i8  Block-FP  (48 dB, sparse transient)
///   h  → f32           (lossless DC)
pub fn write_frame_v3<W: Write>(w: &mut W, f: &HodgeFrame) -> io::Result<()> {
    match f.sc_class {
        ScClass::SC1 => {
            // Silence frame: only harmonic needed
            w.write_all(&[0x11])?;
            w.write_all(&f.harmonic.to_le_bytes())?;
            w.write_all(&f.xi.to_le_bytes())?;
            w.write_all(&[f.sc_class as u8])?;
        }
        ScClass::SC2 => {
            w.write_all(&[0x12])?;

            // ∇φ → i16 Block-FP: scaler (f32) + i16 samples LE
            let g_max = f.gradient.iter().map(|x| x.abs()).fold(1e-20f32, f32::max);
            w.write_all(&g_max.to_le_bytes())?;
            for &x in &f.gradient {
                let q = ((x / g_max).clamp(-1.0, 1.0) * I16_SCALE) as i16;
                w.write_all(&q.to_le_bytes())?;
            }

            // δψ → i8 Block-FP: scaler (f32) + i8 samples
            // Sparsity already applied in decompose() — most values are 0
            let s_max = f.solenoidal.iter().map(|x| x.abs()).fold(1e-20f32, f32::max);
            w.write_all(&s_max.to_le_bytes())?;
            for &x in &f.solenoidal {
                let q = ((x / s_max).clamp(-1.0, 1.0) * I8_SCALE) as i8;
                w.write_all(&[q as u8])?;
            }

            w.write_all(&f.harmonic.to_le_bytes())?;
            w.write_all(&f.xi.to_le_bytes())?;
            w.write_all(&[f.sc_class as u8])?;
        }
        ScClass::SC3 => {
            // Transient burst: keep full f32 precision
            w.write_all(&[0x13])?;
            for &x in &f.gradient   { w.write_all(&x.to_le_bytes())?; }
            for &x in &f.solenoidal { w.write_all(&x.to_le_bytes())?; }
            w.write_all(&f.harmonic.to_le_bytes())?;
            w.write_all(&f.xi.to_le_bytes())?;
            w.write_all(&[f.sc_class as u8])?;
        }
    }
    Ok(())
}

pub fn read_frame_v3<R: Read>(r: &mut R, frame_size: usize) -> io::Result<HodgeFrame> {
    let mut type_byte = [0u8; 1];
    r.read_exact(&mut type_byte)?;
    let mut buf4 = [0u8; 4];
    let mut buf2 = [0u8; 2];

    match type_byte[0] {
        0x11 => {
            r.read_exact(&mut buf4)?; let harmonic = f32::from_le_bytes(buf4);
            r.read_exact(&mut buf4)?; let xi = f32::from_le_bytes(buf4);
            let mut sc_byte = [0u8; 1]; r.read_exact(&mut sc_byte)?;
            Ok(HodgeFrame {
                gradient:   vec![0.0; frame_size],
                solenoidal: vec![0.0; frame_size],
                harmonic, xi,
                sc_class: sc_from_byte(sc_byte[0]),
            })
        }
        0x12 => {
            // ∇φ: i16 Block-FP
            r.read_exact(&mut buf4)?; let g_max = f32::from_le_bytes(buf4);
            let mut gradient = vec![0.0f32; frame_size];
            for g in gradient.iter_mut() {
                r.read_exact(&mut buf2)?;
                let q = i16::from_le_bytes(buf2);
                *g = (q as f32 / I16_SCALE) * g_max;
            }

            // δψ: i8 Block-FP
            r.read_exact(&mut buf4)?; let s_max = f32::from_le_bytes(buf4);
            let mut s_bytes = vec![0u8; frame_size];
            r.read_exact(&mut s_bytes)?;
            let solenoidal: Vec<f32> = s_bytes.iter()
                .map(|&b| (b as i8 as f32 / I8_SCALE) * s_max)
                .collect();

            r.read_exact(&mut buf4)?; let harmonic = f32::from_le_bytes(buf4);
            r.read_exact(&mut buf4)?; let xi = f32::from_le_bytes(buf4);
            let mut sc_byte = [0u8; 1]; r.read_exact(&mut sc_byte)?;
            Ok(HodgeFrame { gradient, solenoidal, harmonic, xi, sc_class: sc_from_byte(sc_byte[0]) })
        }
        0x13 => {
            let read_f32_vec = |r: &mut R| -> io::Result<Vec<f32>> {
                let mut v = vec![0.0f32; frame_size];
                for x in v.iter_mut() {
                    let mut b = [0u8; 4]; r.read_exact(&mut b)?; *x = f32::from_le_bytes(b);
                }
                Ok(v)
            };
            let gradient   = read_f32_vec(r)?;
            let solenoidal = read_f32_vec(r)?;
            r.read_exact(&mut buf4)?; let harmonic = f32::from_le_bytes(buf4);
            r.read_exact(&mut buf4)?; let xi = f32::from_le_bytes(buf4);
            let mut sc_byte = [0u8; 1]; r.read_exact(&mut sc_byte)?;
            Ok(HodgeFrame { gradient, solenoidal, harmonic, xi, sc_class: sc_from_byte(sc_byte[0]) })
        }
        b => Err(io::Error::new(io::ErrorKind::InvalidData,
            format!("unknown v3 frame type 0x{b:02x}")))
    }
}
