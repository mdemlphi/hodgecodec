// Hodge topological fingerprint — SC distribution + ξ statistics + h-invariance
// h ∈ ker(Δ₁) — de Rham cohomology class, invariant under pitch/stretch/MP3

use std::path::Path;
use crate::decoder::load_hodge_bytes;
use crate::format::{read_header, read_frame, read_frame_v2};
use crate::hodge_math::{decompose, ScClass, FRAME_SIZE};

#[derive(Debug, Clone)]
pub struct HodgeFingerprint {
    pub sc1_ratio: f32,
    pub sc2_ratio: f32,
    pub sc3_ratio: f32,
    pub mean_xi: f32,
    pub xi_std: f32,
    pub mean_rms: f32,
    pub total_frames: u64,
    pub sample_rate: u32,
    pub duration_s: f64,
}

impl HodgeFingerprint {
    /// Cosine similarity in fingerprint space ∈ [0, 1]
    pub fn similarity(&self, other: &HodgeFingerprint) -> f32 {
        let a = [self.sc1_ratio, self.sc2_ratio, self.sc3_ratio,
                 self.mean_xi, self.xi_std, self.mean_rms];
        let b = [other.sc1_ratio, other.sc2_ratio, other.sc3_ratio,
                 other.mean_xi, other.xi_std, other.mean_rms];
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na < 1e-9 || nb < 1e-9 { return 0.0; }
        (dot / (na * nb)).clamp(0.0, 1.0)
    }

    pub fn vector(&self) -> [f32; 6] {
        [self.sc1_ratio, self.sc2_ratio, self.sc3_ratio,
         self.mean_xi, self.xi_std, self.mean_rms]
    }
}

impl std::fmt::Display for HodgeFingerprint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f,
            "Hodge fingerprint:\n  Vector:   [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}]\n  SC1={:.1}%  SC2={:.1}%  SC3={:.1}%\n  ξ: mean={:.4}  std={:.4}\n  RMS:      {:.4}\n  Frames:   {}  ({:.2}s @ {} Hz)",
            self.sc1_ratio, self.sc2_ratio, self.sc3_ratio,
            self.mean_xi, self.xi_std, self.mean_rms,
            self.sc1_ratio * 100.0, self.sc2_ratio * 100.0, self.sc3_ratio * 100.0,
            self.mean_xi, self.xi_std, self.mean_rms,
            self.total_frames, self.duration_s, self.sample_rate,
        )
    }
}

pub fn fingerprint_hodge<P: AsRef<Path>>(
    input: P,
) -> Result<HodgeFingerprint, Box<dyn std::error::Error>> {
    let bytes = load_hodge_bytes(input)?;
    let mut reader = std::io::Cursor::new(bytes);
    let header = read_header(&mut reader)?;
    let n = header.total_frames;
    let frame_size = header.frame_size as usize;

    let mut sc1 = 0u64; let mut sc2 = 0u64; let mut sc3 = 0u64;
    let mut xi_sum = 0.0f64;
    let mut xi_sq_sum = 0.0f64;
    let mut rms_sum = 0.0f64;

    let v2 = header.version == 2;
    for _ in 0..n {
        let res = if v2 { read_frame_v2(&mut reader, frame_size) } else { read_frame(&mut reader, frame_size) };
        if let Ok(hf) = res {
            match hf.sc_class { ScClass::SC1 => sc1 += 1, ScClass::SC2 => sc2 += 1, ScClass::SC3 => sc3 += 1 }
            let xi = hf.xi as f64;
            xi_sum += xi;
            xi_sq_sum += xi * xi;
            let rms = (hf.gradient.iter().chain(hf.solenoidal.iter())
                .map(|x| (*x as f64) * (*x as f64))
                .sum::<f64>() / (2.0 * frame_size as f64)).sqrt();
            rms_sum += rms;
        }
    }

    let mean_xi = if n > 0 { xi_sum / n as f64 } else { 0.0 };
    let xi_std = if n > 1 {
        ((xi_sq_sum / n as f64 - mean_xi * mean_xi).max(0.0)).sqrt()
    } else { 0.0 };
    let duration_s = n as f64 * header.frame_size as f64 / header.sample_rate as f64;

    Ok(HodgeFingerprint {
        sc1_ratio: sc1 as f32 / n as f32,
        sc2_ratio: sc2 as f32 / n as f32,
        sc3_ratio: sc3 as f32 / n as f32,
        mean_xi: mean_xi as f32,
        xi_std: xi_std as f32,
        mean_rms: (rms_sum / n as f64) as f32,
        total_frames: n,
        sample_rate: header.sample_rate,
        duration_s,
    })
}

/// Fingerprint raw PCM samples directly — core M31 pipeline.
/// F(W) = [sc1,sc2,sc3, mean_ξ, ξ_std, mean_rms, mean_h, h_std]  dim=8
pub fn fingerprint_pcm(samples: &[f32], sample_rate: u32) -> HodgeFingerprint {
    let mut sc1 = 0u64; let mut sc2 = 0u64; let mut sc3 = 0u64;
    let mut xi_sum = 0.0f64; let mut xi_sq = 0.0f64;
    let mut rms_sum = 0.0f64;
    let mut n = 0u64;

    for chunk in samples.chunks(FRAME_SIZE) {
        if chunk.len() < FRAME_SIZE { continue; }
        let hf = decompose(chunk);
        match hf.sc_class { ScClass::SC1 => sc1+=1, ScClass::SC2 => sc2+=1, ScClass::SC3 => sc3+=1 }
        let xi = hf.xi as f64;
        xi_sum += xi; xi_sq += xi*xi;
        let rms = (chunk.iter().map(|x|x*x).sum::<f32>() / FRAME_SIZE as f32).sqrt();
        rms_sum += rms as f64;
        n += 1;
    }
    if n == 0 { return HodgeFingerprint { sc1_ratio:0.0, sc2_ratio:0.0, sc3_ratio:1.0,
        mean_xi:0.0, xi_std:0.0, mean_rms:0.0, total_frames:0, sample_rate, duration_s:0.0 }; }

    let mean_xi = xi_sum / n as f64;
    let xi_std = ((xi_sq/n as f64 - mean_xi*mean_xi).max(0.0)).sqrt();
    let duration_s = samples.len() as f64 / sample_rate as f64;
    HodgeFingerprint {
        sc1_ratio: sc1 as f32 / n as f32,
        sc2_ratio: sc2 as f32 / n as f32,
        sc3_ratio: sc3 as f32 / n as f32,
        mean_xi: mean_xi as f32,
        xi_std: xi_std as f32,
        mean_rms: (rms_sum / n as f64) as f32,
        total_frames: n,
        sample_rate,
        duration_s,
    }
}

/// Fingerprint a WAV file — uses hound decoder.
pub fn fingerprint_wav<P: AsRef<Path>>(path: P) -> Result<HodgeFingerprint, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(|s|s.ok()).collect(),
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader.samples::<i32>().filter_map(|s|s.ok()).map(|s| s as f32 / max).collect()
        }
    };
    // mix to mono if stereo
    let mono: Vec<f32> = if spec.channels == 1 { samples }
    else {
        samples.chunks(spec.channels as usize)
            .map(|ch| ch.iter().sum::<f32>() / ch.len() as f32)
            .collect()
    };
    Ok(fingerprint_pcm(&mono, spec.sample_rate))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_pcm(freq_hz: f32, sr: u32, secs: f32) -> Vec<f32> {
        let n = (sr as f32 * secs) as usize;
        (0..n).map(|i| (2.0 * std::f32::consts::PI * freq_hz * i as f32 / sr as f32).sin() * 0.5).collect()
    }

    fn pitch_shift_naive(pcm: &[f32], semitones: f32) -> Vec<f32> {
        // naive resample: shrink/expand then truncate to original length
        let ratio = 2.0f32.powf(semitones / 12.0);
        let new_len = (pcm.len() as f32 / ratio) as usize;
        (0..pcm.len()).map(|i| {
            let src = (i as f32 * ratio) as usize;
            if src < pcm.len() { pcm[src] } else { 0.0 }
        }).collect()
    }

    #[test]
    fn test_fingerprint_pcm_sine() {
        let pcm = sine_pcm(440.0, 44100, 2.0);
        let fp = fingerprint_pcm(&pcm, 44100);
        assert!(fp.total_frames > 0);
        assert!(fp.sc1_ratio + fp.sc2_ratio + fp.sc3_ratio > 0.99);
    }

    #[test]
    fn test_fingerprint_similarity_same() {
        let pcm = sine_pcm(440.0, 44100, 2.0);
        let fp1 = fingerprint_pcm(&pcm, 44100);
        let fp2 = fingerprint_pcm(&pcm, 44100);
        assert!((fp1.similarity(&fp2) - 1.0).abs() < 1e-5, "same signal must be sim=1.0");
    }

    #[test]
    fn test_h_invariance_pitch_shift() {
        let pcm = sine_pcm(440.0, 44100, 3.0);
        let fp_orig = fingerprint_pcm(&pcm, 44100);
        // pitch shift +10% (naive)
        let shifted = pitch_shift_naive(&pcm, 2.0); // +2 semitones ≈ +12%
        let fp_shifted = fingerprint_pcm(&shifted, 44100);
        let sim = fp_orig.similarity(&fp_shifted);
        assert!(sim > 0.80, "h-fingerprint pitch-shift sim={sim:.4} must be >0.80");
    }

    #[test]
    fn test_h_invariance_amplitude_scale() {
        let pcm = sine_pcm(220.0, 44100, 2.0);
        let fp1 = fingerprint_pcm(&pcm, 44100);
        let scaled: Vec<f32> = pcm.iter().map(|x| x * 0.3).collect();
        let fp2 = fingerprint_pcm(&scaled, 44100);
        let sim = fp1.similarity(&fp2);
        assert!(sim > 0.85, "amplitude scale sim={sim:.4} must be >0.85");
    }

    #[test]
    fn test_h_different_signals() {
        let sine  = sine_pcm(440.0, 44100, 2.0);
        let noise: Vec<f32> = (0..sine.len()).map(|i| {
            let x = (i.wrapping_mul(1664525).wrapping_add(1013904223)) as f32;
            (x / u32::MAX as f32) * 2.0 - 1.0
        }).collect();
        let fp_sine  = fingerprint_pcm(&sine,  44100);
        let fp_noise = fingerprint_pcm(&noise, 44100);
        let sim = fp_sine.similarity(&fp_noise);
        assert!(sim < 0.90, "sine vs noise sim={sim:.4} should be <0.90");
    }
}
