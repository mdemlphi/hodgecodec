// WAV → .hodge encoder

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use hound::WavReader;
use crate::hodge_math::{decompose, FRAME_SIZE};
use std::io;
use crate::hodge_math::{HodgeFrame, ScClass};
use crate::format::{write_header, write_frame, write_frame_v3, HodgeFileHeader, VERSION, VERSION_V3, I8_SCALE};

const ZSTD_LEVEL: i32 = 3;

pub struct EncodeStats {
    pub total_frames: u64,
    pub sc1_frames: u64,
    pub sc2_frames: u64,
    pub sc3_frames: u64,
    pub mean_xi: f32,
}

pub fn encode_wav<P: AsRef<Path>, Q: AsRef<Path>>(
    input: P, output: Q, author: Option<&str>,
) -> Result<EncodeStats, Box<dyn std::error::Error>> {
    let out = File::create(output)?;
    encode_to_writer(input, BufWriter::new(out), false, author)
}

pub fn encode_wav_compressed<P: AsRef<Path>, Q: AsRef<Path>>(
    input: P, output: Q, author: Option<&str>,
) -> Result<EncodeStats, Box<dyn std::error::Error>> {
    let out = File::create(output)?;
    encode_to_writer(input, BufWriter::new(out), true, author)
}

pub fn encode_wav_zstd<P: AsRef<Path>, Q: AsRef<Path>>(
    input: P, output: Q, author: Option<&str>,
) -> Result<EncodeStats, Box<dyn std::error::Error>> {
    let mut buf: Vec<u8> = Vec::new();
    let stats = encode_to_writer(input, &mut buf, true, author)?;
    let compressed = zstd::encode_all(std::io::Cursor::new(buf), ZSTD_LEVEL)?;
    std::fs::write(output, compressed)?;
    Ok(stats)
}

fn encode_to_writer<P: AsRef<Path>, W: Write>(
    input: P,
    mut writer: W,
    compress: bool,
    author: Option<&str>,
) -> Result<EncodeStats, Box<dyn std::error::Error>> {
    let mut reader = WavReader::open(input)?;
    let spec = reader.spec();

    let samples_i16: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap()).collect();
    let samples: Vec<f32> = samples_i16.iter()
        .map(|&s| s as f32 / i16::MAX as f32)
        .collect();

    let ch = spec.channels as usize;
    let mono: Vec<f32> = if ch == 1 {
        samples
    } else {
        samples.chunks(ch).map(|c| c[0]).collect()
    };

    let frames: Vec<Vec<f32>> = mono.chunks(FRAME_SIZE)
        .map(|chunk| {
            let mut f = chunk.to_vec();
            f.resize(FRAME_SIZE, 0.0);
            f
        })
        .collect();

    let total_frames = frames.len() as u64;
    let header = HodgeFileHeader {
        version: if compress { VERSION_V3 } else { VERSION },
        sample_rate: spec.sample_rate,
        channels: 1,
        frame_size: FRAME_SIZE as u32,
        total_frames,
        author: author.map(|s| s.to_string()),
    };

    write_header(&mut writer, &header)?;

    let mut sc1 = 0u64; let mut sc2 = 0u64; let mut sc3 = 0u64;
    let mut xi_sum = 0.0f32;

    for frame_data in &frames {
        let hf = decompose(frame_data);
        use crate::hodge_math::ScClass::*;
        match hf.sc_class { SC1 => sc1 += 1, SC2 => sc2 += 1, SC3 => sc3 += 1 }
        xi_sum += hf.xi;
        if compress { write_frame_v3(&mut writer, &hf)? } else { write_frame(&mut writer, &hf)? };
    }

    Ok(EncodeStats {
        total_frames,
        sc1_frames: sc1,
        sc2_frames: sc2,
        sc3_frames: sc3,
        mean_xi: if total_frames > 0 { xi_sum / total_frames as f32 } else { 0.0 },
    })
}

#[allow(dead_code)]
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
            for &x in &f.gradient { w.write_all(&x.to_le_bytes())?; }
            for &x in &f.solenoidal { w.write_all(&x.to_le_bytes())?; }
            w.write_all(&f.harmonic.to_le_bytes())?;
            w.write_all(&f.xi.to_le_bytes())?;
            w.write_all(&[f.sc_class as u8])?;
        }
    }
    Ok(())
}
