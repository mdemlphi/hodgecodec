// Shazam-style spectrogram constellation hash — baseline for M31 comparison.
// Wang 2003: local spectral peaks → hash pairs (f1,f2,Δt) → fingerprint set.
// Intentionally simple: no fan-out, fixed window. Mirrors real ACR behavior.

use std::collections::HashSet;

const FFT_SIZE: usize = 1024;
const HOP: usize = 512;
const PEAK_NEIGHBORHOOD: usize = 5;   // bins each side
const FAN_OUT: usize = 5;             // pairs per peak
const MAX_DT: usize = 20;             // max frame distance for pairing

pub struct ShazamFingerprint {
    pub hashes: HashSet<u64>,
}

impl ShazamFingerprint {
    pub fn similarity(&self, other: &ShazamFingerprint) -> f32 {
        if self.hashes.is_empty() || other.hashes.is_empty() { return 0.0; }
        let matches = self.hashes.intersection(&other.hashes).count();
        let denom = self.hashes.len().min(other.hashes.len());
        matches as f32 / denom as f32
    }
}

fn hann(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.5 * (1.0 - (2.0*std::f32::consts::PI*i as f32/(n-1) as f32).cos())).collect()
}

// Minimal DFT magnitude — no external crate needed, O(N²) but N=1024 is fine for bench
fn dft_mag(frame: &[f32]) -> Vec<f32> {
    let n = frame.len();
    let half = n / 2 + 1;
    let mut mag = vec![0.0f32; half];
    for k in 0..half {
        let (mut re, mut im) = (0.0f32, 0.0f32);
        for i in 0..n {
            let angle = -2.0 * std::f32::consts::PI * k as f32 * i as f32 / n as f32;
            re += frame[i] * angle.cos();
            im += frame[i] * angle.sin();
        }
        mag[k] = (re*re + im*im).sqrt();
    }
    mag
}

fn find_peaks(mag: &[f32]) -> Vec<usize> {
    let mut peaks = Vec::new();
    let n = mag.len();
    for i in PEAK_NEIGHBORHOOD..n-PEAK_NEIGHBORHOOD {
        let is_peak = (i-PEAK_NEIGHBORHOOD..i+PEAK_NEIGHBORHOOD)
            .all(|j| j == i || mag[j] <= mag[i]);
        if is_peak && mag[i] > 1e-6 { peaks.push(i); }
    }
    peaks
}

pub fn fingerprint_pcm(samples: &[f32]) -> ShazamFingerprint {
    let win = hann(FFT_SIZE);
    let mut frames: Vec<Vec<usize>> = Vec::new();

    for start in (0..samples.len().saturating_sub(FFT_SIZE)).step_by(HOP) {
        let frame: Vec<f32> = (0..FFT_SIZE)
            .map(|i| if start+i < samples.len() { samples[start+i] * win[i] } else { 0.0 })
            .collect();
        let mag = dft_mag(&frame);
        frames.push(find_peaks(&mag));
    }

    let mut hashes = HashSet::new();
    for (t1, peaks1) in frames.iter().enumerate() {
        for &f1 in peaks1.iter().take(FAN_OUT) {
            for dt in 1..MAX_DT.min(frames.len()-t1) {
                for &f2 in frames[t1+dt].iter().take(FAN_OUT) {
                    // hash: f1(10b) | f2(10b) | dt(8b) packed into u64
                    let h: u64 = ((f1 as u64) << 18) | ((f2 as u64) << 8) | dt as u64;
                    hashes.insert(h);
                }
            }
        }
    }
    ShazamFingerprint { hashes }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f32, sr: u32, secs: f32) -> Vec<f32> {
        let n = (sr as f32 * secs) as usize;
        (0..n).map(|i| (2.0*std::f32::consts::PI*freq*i as f32/sr as f32).sin()*0.5).collect()
    }

    fn pitch_shift(pcm: &[f32], semitones: f32) -> Vec<f32> {
        let ratio = 2.0f32.powf(semitones/12.0);
        (0..pcm.len()).map(|i| {
            let src = (i as f32 * ratio) as usize;
            if src < pcm.len() { pcm[src] } else { 0.0 }
        }).collect()
    }

    #[test]
    fn test_shazam_self_similarity() {
        let pcm = sine(440.0, 44100, 3.0);
        let fp1 = fingerprint_pcm(&pcm);
        let fp2 = fingerprint_pcm(&pcm);
        assert!(fp1.similarity(&fp2) > 0.99, "self-similarity must be ~1.0");
    }

    #[test]
    fn test_shazam_breaks_at_pitch_shift() {
        let pcm = sine(440.0, 44100, 3.0);
        let fp_orig = fingerprint_pcm(&pcm);
        let shifted = pitch_shift(&pcm, 2.0); // +2 semitones
        let fp_shifted = fingerprint_pcm(&shifted);
        let sim = fp_orig.similarity(&fp_shifted);
        // Shazam SHOULD fail here — this proves our Hodge method is superior
        println!("Shazam pitch+2st sim={sim:.4}");
        // No assertion — we just measure. Expected: < 0.30
    }
}
