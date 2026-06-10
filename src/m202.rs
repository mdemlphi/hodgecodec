// M202 Unified Topological Scoring
// score ∈ [0,1] from 7 Hodge invariants — no Golden Ratio, no L1-as-Betti hacks
//
// Components:
//   ξ        — conservation violation (from decompose)
//   e_tonal  — E(∇φ) / E_total  (exact energy fraction)
//   e_attack — E(δψ) / E_total  (coexact energy fraction)
//   e_dc     — h² / E_total     (harmonic DC fraction)
//   rms      — sqrt(mean(x²))   (signal level gate)
//   β₁       — persistent peak count above RMS (sublevel set filtration proxy)
//   sc_class — SC1/SC2/SC3 encoded to [0,1]

use crate::hodge_math::{decompose, ScClass};

#[derive(Debug, Clone)]
pub struct M202Score {
    /// Final scalar score ∈ [0,1]
    pub score: f32,
    /// ξ conservation violation index
    pub xi: f32,
    /// E(∇φ) / E_total
    pub tonal_ratio: f32,
    /// E(δψ) / E_total
    pub attack_ratio: f32,
    /// h² / E_total
    pub dc_ratio: f32,
    /// sqrt(mean(x²))
    pub rms: f32,
    /// β₁ proxy: persistent peak count above RMS threshold
    pub beta1: usize,
    pub sc_class: ScClass,
}

/// Compute M202 score for a single PCM frame.
///
/// Weights are tuned for audio quality assessment:
/// - High tonal fraction + low ξ → coherent tonal signal → score → 1
/// - SC3 (transient) or silence → lower score
pub fn compute_m202(signal: &[f32]) -> M202Score {
    let n = signal.len();
    if n < 3 {
        return M202Score {
            score: 0.0, xi: 0.0, tonal_ratio: 0.0, attack_ratio: 0.0,
            dc_ratio: 0.0, rms: 0.0, beta1: 0, sc_class: ScClass::SC1,
        };
    }

    let hf = decompose(signal);
    let nf = n as f32;

    // Energy of each Hodge component
    let e_grad = hf.gradient.iter().map(|x| x * x).sum::<f32>() / nf;
    let e_sol  = hf.solenoidal.iter().map(|x| x * x).sum::<f32>() / nf;
    let e_harm = hf.harmonic * hf.harmonic;
    let e_total = e_grad + e_sol + e_harm + 1e-9;

    let tonal_ratio  = e_grad / e_total;
    let attack_ratio = e_sol  / e_total;
    let dc_ratio     = e_harm / e_total;

    // RMS of original signal
    let rms = (signal.iter().map(|x| x * x).sum::<f32>() / nf).sqrt();

    // β₁ proxy: count local maxima of |signal| that exceed RMS threshold.
    // Each peak above RMS = a birth in the sublevel-set filtration of |signal|.
    // For a pure sinusoid: β₁ = N_cycles ≈ f·T.
    // For noise: β₁ ≈ N/4 (dense maxima).
    let threshold = rms;
    let beta1 = signal.windows(3)
        .filter(|w| {
            let abs_mid = w[1].abs();
            abs_mid > w[0].abs() && abs_mid > w[2].abs() && abs_mid > threshold
        })
        .count();

    // SC class as numeric [0=silence, 0.5=tonal, 1.0=attack]
    let _sc_num = match hf.sc_class {
        ScClass::SC1 => 0.0f32,
        ScClass::SC2 => 0.5,
        ScClass::SC3 => 1.0,
    };

    // Weighted score — silence-safe:
    //   (1-ξ)×gate  × 0.35  — conservation only counts when signal present
    //   tonal_ratio  × 0.30  — energy in exact component
    //   sc_reward    × 0.20  — SC2=1.0, SC3=0.4, SC1=0.0
    //   rms_gate     × 0.15  — non-silence bonus
    let rms_gate  = if rms > 1e-4 { 1.0f32 } else { 0.0 };
    let sc_reward = match hf.sc_class {
        ScClass::SC2 => 1.0f32,
        ScClass::SC3 => 0.4,
        ScClass::SC1 => 0.0,
    };
    let score = (
        (1.0 - hf.xi) * rms_gate * 0.35
        + tonal_ratio              * 0.30
        + sc_reward                * 0.20
        + rms_gate                 * 0.15
    ).clamp(0.0, 1.0);

    M202Score { score, xi: hf.xi, tonal_ratio, attack_ratio, dc_ratio, rms, beta1, sc_class: hf.sc_class }
}

/// Aggregate M202 over a full PCM stream (frame-by-frame).
pub fn compute_m202_stream(pcm: &[f32], frame_size: usize) -> f32 {
    if pcm.is_empty() || frame_size == 0 { return 0.0; }
    let scores: Vec<f32> = pcm.chunks(frame_size)
        .filter(|c| c.len() >= 3)
        .map(|c| {
            let mut buf = c.to_vec();
            buf.resize(frame_size, 0.0);
            compute_m202(&buf).score
        })
        .collect();
    if scores.is_empty() { return 0.0; }
    scores.iter().sum::<f32>() / scores.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_m202_pure_sine() {
        let signal: Vec<f32> = (0..1024)
            .map(|i| (i as f32 * 2.0 * std::f32::consts::PI * 440.0 / 44100.0).sin() * 0.8)
            .collect();
        let s = compute_m202(&signal);
        assert!(s.score > 0.5, "pure sine should score > 0.5, got {:.3}", s.score);
        assert!(s.tonal_ratio > 0.5, "sine: tonal_ratio > 0.5, got {:.3}", s.tonal_ratio);
        assert_eq!(s.sc_class, ScClass::SC2, "sine must be SC2");
    }

    #[test]
    fn test_m202_silence() {
        let signal = vec![0.0f32; 1024];
        let s = compute_m202(&signal);
        assert!(s.score < 0.5, "silence should score < 0.5, got {:.3}", s.score);
        assert_eq!(s.sc_class, ScClass::SC1);
    }

    #[test]
    fn test_m202_score_range() {
        // white noise
        let signal: Vec<f32> = (0..1024).map(|i| {
            let x = ((i as u32).wrapping_mul(2654435761) ^ ((i as u32) >> 16)) as f32;
            x / u32::MAX as f32 * 2.0 - 1.0
        }).collect();
        let s = compute_m202(&signal);
        assert!(s.score >= 0.0 && s.score <= 1.0, "score out of range: {}", s.score);
        assert!(s.beta1 > 0, "noise must have β₁ > 0");
    }

    #[test]
    fn test_m202_sine_beats_noise() {
        let sine: Vec<f32> = (0..1024)
            .map(|i| (i as f32 * 0.1).sin() * 0.7)
            .collect();
        let noise: Vec<f32> = (0..1024).map(|i| {
            let x = ((i as u32).wrapping_mul(2654435761) ^ ((i as u32) >> 16)) as f32;
            (x / u32::MAX as f32 * 2.0 - 1.0) * 0.7
        }).collect();
        let s_sine  = compute_m202(&sine).score;
        let s_noise = compute_m202(&noise).score;
        assert!(s_sine > s_noise, "sine ({s_sine:.3}) should beat noise ({s_noise:.3})");
    }
}
