// hodge mix — blend two .hodge files at component level
// hodge mix a.hodge b.hodge out.hodge --tone-a 1.0 --attack-a 0.0 --soul-a 1.0
// Default: alpha=0.5 (equal blend of all components)

use std::fs::File;
use std::io::{BufWriter};
use std::path::Path;
use crate::decoder::load_hodge_bytes;
use crate::format::{read_header, read_frame, read_frame_v2, write_header, write_frame, HodgeFileHeader, VERSION};
use crate::hodge_math::{HodgeFrame, ScClass, compute_xi};

pub struct MixParams {
    /// blend ratio for ∇φ: 1.0 = 100% from A, 0.0 = 100% from B
    pub tone_a: f32,
    pub attack_a: f32,
    pub soul_a: f32,
}

impl Default for MixParams {
    fn default() -> Self { Self { tone_a: 0.5, attack_a: 0.5, soul_a: 0.5 } }
}

pub struct MixStats {
    pub total_frames: u64,
    pub sample_rate: u32,
}

pub fn mix_hodge<P, Q, R>(
    a: P, b: Q, output: R,
    params: MixParams,
) -> Result<MixStats, Box<dyn std::error::Error>>
where P: AsRef<Path>, Q: AsRef<Path>, R: AsRef<Path>
{
    let ba = load_hodge_bytes(a)?;
    let bb = load_hodge_bytes(b)?;
    let mut ra = std::io::Cursor::new(ba);
    let mut rb = std::io::Cursor::new(bb);
    let ha = read_header(&mut ra)?;
    let hb = read_header(&mut rb)?;

    if ha.frame_size != hb.frame_size {
        return Err(format!("frame_size mismatch: {} vs {}", ha.frame_size, hb.frame_size).into());
    }
    if ha.sample_rate != hb.sample_rate {
        return Err(format!("sample_rate mismatch: {} vs {}", ha.sample_rate, hb.sample_rate).into());
    }

    let total_frames = ha.total_frames.min(hb.total_frames);
    let header = HodgeFileHeader {
        version: VERSION,
        sample_rate: ha.sample_rate,
        channels: 1,
        frame_size: ha.frame_size,
        total_frames,
        author: ha.author.clone().or(hb.author.clone()),
    };

    let mut writer = BufWriter::new(File::create(output)?);
    write_header(&mut writer, &header)?;

    let frame_size = ha.frame_size as usize;
    let wa_t = params.tone_a;
    let wa_a = params.attack_a;
    let wa_s = params.soul_a;
    let v2a = ha.version == 2;
    let v2b = hb.version == 2;

    for _ in 0..total_frames {
        let fa = if v2a { read_frame_v2(&mut ra, frame_size)? } else { read_frame(&mut ra, frame_size)? };
        let fb = if v2b { read_frame_v2(&mut rb, frame_size)? } else { read_frame(&mut rb, frame_size)? };

        let gradient: Vec<f32> = fa.gradient.iter().zip(fb.gradient.iter())
            .map(|(a, b)| a * wa_t + b * (1.0 - wa_t))
            .collect();
        let solenoidal: Vec<f32> = fa.solenoidal.iter().zip(fb.solenoidal.iter())
            .map(|(a, b)| a * wa_a + b * (1.0 - wa_a))
            .collect();
        let harmonic = fa.harmonic * wa_s + fb.harmonic * (1.0 - wa_s);

        let xi = compute_xi(&gradient, &solenoidal);
        let centered_sq_sum: f32 = gradient.iter().zip(solenoidal.iter())
            .map(|(g, s)| (g + s) * (g + s)).sum::<f32>();
        let rms = (centered_sq_sum / frame_size as f32).sqrt();
        let sc_class = ScClass::from_energy_xi(rms, xi);

        write_frame(&mut writer, &HodgeFrame { gradient, solenoidal, harmonic, xi, sc_class })?;
    }

    Ok(MixStats { total_frames, sample_rate: ha.sample_rate })
}
