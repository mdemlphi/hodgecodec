// .hodge → WAV decoder with per-component gain manipulation
// hodge decode track.hodge out.wav --attack 2.0 --tone 0.5 --soul 1.0

use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;
use hound::{WavWriter, WavSpec, SampleFormat};
use crate::hodge_math::{reconstruct, ScClass};
use crate::format::{read_header, read_frame, read_frame_v2, read_frame_v3};

const ZSTD_MAGIC: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD];

/// Read file, transparently decompressing if zstd-wrapped.
pub fn load_hodge_bytes<P: AsRef<Path>>(path: P) -> io::Result<Vec<u8>> {
    let mut f = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    let mut tail = Vec::new();
    f.read_to_end(&mut tail)?;
    let all: Vec<u8> = magic.iter().chain(tail.iter()).copied().collect();
    if magic == ZSTD_MAGIC {
        zstd::decode_all(std::io::Cursor::new(all)).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    } else {
        Ok(all)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DecodeParams {
    pub tone_gain: f32,    // ∇φ gain (tonal/melody)
    pub attack_gain: f32,  // δψ gain (transients/drums)
    pub soul_gain: f32,    // h gain (DC/soul)
}

impl Default for DecodeParams {
    fn default() -> Self {
        Self { tone_gain: 1.0, attack_gain: 1.0, soul_gain: 1.0 }
    }
}

pub struct DecodeStats {
    pub total_frames: u64,
    pub total_samples: u64,
    pub sample_rate: u32,
}

pub fn decode_hodge<P: AsRef<Path>, Q: AsRef<Path>>(
    input: P,
    output: Q,
    params: DecodeParams,
) -> Result<DecodeStats, Box<dyn std::error::Error>> {
    let bytes = load_hodge_bytes(input)?;
    let mut reader = std::io::Cursor::new(bytes);
    let header = read_header(&mut reader)?;

    let spec = WavSpec {
        channels: header.channels,
        sample_rate: header.sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut writer = WavWriter::create(output, spec)?;
    let frame_size = header.frame_size as usize;
    let mut total_samples = 0u64;

    for _ in 0..header.total_frames {
        let hf = match header.version {
            3 => read_frame_v3(&mut reader, frame_size)?,
            2 => read_frame_v2(&mut reader, frame_size)?,
            1 => read_frame(&mut reader, frame_size)?,
            v => return Err(format!("Unsupported version: {}", v).into()),
        };
        let pcm = reconstruct(&hf, params.tone_gain, params.attack_gain, params.soul_gain);
        for sample in pcm {
            let s = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            writer.write_sample(s)?;
            total_samples += 1;
        }
    }
    writer.finalize()?;

    Ok(DecodeStats {
        total_frames: header.total_frames,
        total_samples,
        sample_rate: header.sample_rate,
    })
}

/// Returns (pcm, sample_rate, frame_size, per_frame_stats: Vec<(xi, sc_class)>)
pub fn decode_hodge_with_stats<P: AsRef<Path>>(
    input: P,
    params: DecodeParams,
) -> Result<(Vec<f32>, u32, usize, Vec<(f32, ScClass)>), Box<dyn std::error::Error>> {
    let bytes = load_hodge_bytes(input)?;
    let mut reader = std::io::Cursor::new(bytes);
    let header = read_header(&mut reader)?;

    let cap = (header.total_frames * header.frame_size as u64) as usize;
    let mut pcm = Vec::with_capacity(cap);
    let mut stats = Vec::with_capacity(header.total_frames as usize);

    let fs = header.frame_size as usize;
    for _ in 0..header.total_frames {
        let hf = match header.version {
            3 => read_frame_v3(&mut reader, fs)?,
            2 => read_frame_v2(&mut reader, fs)?,
            1 => read_frame(&mut reader, fs)?,
            v => return Err(format!("Unsupported version: {}", v).into()),
        };
        stats.push((hf.xi, hf.sc_class));
        let samples = reconstruct(&hf, params.tone_gain, params.attack_gain, params.soul_gain);
        pcm.extend_from_slice(&samples);
    }

    Ok((pcm, header.sample_rate, header.frame_size as usize, stats))
}

pub fn decode_hodge_to_pcm<P: AsRef<Path>>(
    input: P,
    params: DecodeParams,
) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    let bytes = load_hodge_bytes(input)?;
    let mut reader = std::io::Cursor::new(bytes);
    let header = read_header(&mut reader)?;

    let cap = (header.total_frames * header.frame_size as u64) as usize;
    let mut pcm = Vec::with_capacity(cap);
    let fs = header.frame_size as usize;

    for _ in 0..header.total_frames {
        let hf = match header.version {
            3 => read_frame_v3(&mut reader, fs)?,
            2 => read_frame_v2(&mut reader, fs)?,
            1 => read_frame(&mut reader, fs)?,
            v => return Err(format!("Unsupported version: {}", v).into()),
        };
        let samples = reconstruct(&hf, params.tone_gain, params.attack_gain, params.soul_gain);
        pcm.extend_from_slice(&samples);
    }

    Ok((pcm, header.sample_rate))
}

pub fn info_hodge<P: AsRef<Path>>(input: P) -> Result<String, Box<dyn std::error::Error>> {
    let bytes = load_hodge_bytes(input)?;
    let mut reader = std::io::Cursor::new(bytes);
    let header = read_header(&mut reader)?;

    let duration_s = (header.total_frames * header.frame_size as u64) as f64
        / header.sample_rate as f64;

    let mut sc1 = 0u64; let mut sc2 = 0u64; let mut sc3 = 0u64;
    let mut xi_sum = 0.0f64;

    let fs = header.frame_size as usize;
    for _ in 0..header.total_frames {
        let res = match header.version {
            3 => read_frame_v3(&mut reader, fs),
            2 => read_frame_v2(&mut reader, fs),
            1 => read_frame(&mut reader, fs),
            v => Err(io::Error::new(io::ErrorKind::InvalidData, format!("Unsupported version: {}", v))),
        };
        if let Ok(hf) = res {
            use crate::hodge_math::ScClass::*;
            match hf.sc_class { SC1 => sc1 += 1, SC2 => sc2 += 1, SC3 => sc3 += 1 }
            xi_sum += hf.xi as f64;
        }
    }

    let mean_xi = if header.total_frames > 0 { xi_sum / header.total_frames as f64 } else { 0.0 };

    let author_line = match &header.author {
        Some(a) => format!("\n  Author:      {}", a),
        None    => String::new(),
    };
    Ok(format!(
        ".hodge file info:{}\n  Version:     {}\n  Sample rate: {} Hz\n  Channels:    {}\n  Frame size:  {} samples\n  Total frames:{}\n  Duration:    {:.2}s\n  Mean ξ:      {:.4}\n  SC1 (silence):  {} frames ({:.1}%)\n  SC2 (tonal):    {} frames ({:.1}%)\n  SC3 (attack):   {} frames ({:.1}%)",
        author_line,
        header.version,
        header.sample_rate,
        header.channels,
        header.frame_size,
        header.total_frames,
        duration_s,
        mean_xi,
        sc1, 100.0 * sc1 as f64 / header.total_frames as f64,
        sc2, 100.0 * sc2 as f64 / header.total_frames as f64,
        sc3, 100.0 * sc3 as f64 / header.total_frames as f64,
    ))
}
