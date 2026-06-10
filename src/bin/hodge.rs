// hodge CLI — encode/decode/info/stem/fingerprint/mix for .hodge files

use clap::{Parser, Subcommand, ValueEnum};

#[derive(Clone, ValueEnum)]
enum WmAction { Embed, Decode }
use hodgecodec::{encode_wav, encode_wav_compressed, encode_wav_zstd, decode_hodge, info_hodge,
                  fingerprint_hodge, mix_hodge, DecodeParams, MixParams,
                  watermark_encode, watermark_decode};
use hound::WavReader;

#[derive(Parser)]
#[command(name = "hodge", version = "0.5.0", about = "HodgeCodec — Hodge decomposition audio codec")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Encode WAV to .hodge format
    Encode {
        input: String,
        output: String,
        /// SC-aware Block-FP compression (v2): SC1→10B, SC2→i8, SC3→f32
        #[arg(long)] compress: bool,
        /// zstd entropy coding on top of --compress (smaller file, slower decode)
        #[arg(long)] zstd: bool,
        /// Embed author/copyright into .hodge header, e.g. "MDIACH.EML © 2026 |HodgeCodec"
        #[arg(long, short)] author: Option<String>,
    },
    /// Decode .hodge to WAV with optional component manipulation
    Decode {
        input: String,
        output: String,
        #[arg(long, default_value_t = 1.0)] attack: f32,
        #[arg(long, default_value_t = 1.0)] tone: f32,
        #[arg(long, default_value_t = 1.0)] soul: f32,
    },
    /// Show .hodge file statistics and Hodge decomposition info
    Info {
        input: String,
    },
    /// Split .hodge into 3 component stems (tone / attack / soul)
    Stem {
        /// Input .hodge file
        input: String,
        /// Output prefix — produces <prefix>_tone.wav, <prefix>_attack.wav, <prefix>_soul.wav
        prefix: String,
    },
    /// Compute topological fingerprint vector of a .hodge file
    Fingerprint {
        input: String,
        /// Optional second file to compare similarity
        compare: Option<String>,
    },
    /// Compare two WAV files — SNR, RMSE, max error (quality audit)
    Compare {
        /// Reference (original) WAV
        reference: String,
        /// Decoded/processed WAV to compare against
        decoded: String,
    },
    /// Embed author watermark into WAV (inaudible LSB steganography)
    Watermark {
        /// embed / decode
        #[arg(value_enum)] action: WmAction,
        /// Input WAV file
        input: String,
        /// Output WAV (only for embed)
        output: Option<String>,
        /// Author string to embed, e.g. "MDIACH.EML"
        #[arg(long, short)] author: Option<String>,
    },
    /// Blend two .hodge files at component level
    Mix {
        input_a: String,
        input_b: String,
        output: String,
        /// ∇φ (tone) blend — 1.0 = 100% from A, 0.0 = 100% from B
        #[arg(long, default_value_t = 0.5)] tone_a: f32,
        /// δψ (attack) blend
        #[arg(long, default_value_t = 0.5)] attack_a: f32,
        /// h (soul) blend
        #[arg(long, default_value_t = 0.5)] soul_a: f32,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Encode { input, output, compress, zstd, author } => {
            let label = if zstd { " [v2+zstd]" } else if compress { " [v2 compressed]" } else { "" };
            print!("Encoding {} → {}{}...", input, output, label);
            let auth = author.as_deref();
            let result = if zstd { encode_wav_zstd(&input, &output, auth) }
                         else if compress { encode_wav_compressed(&input, &output, auth) }
                         else { encode_wav(&input, &output, auth) };
            match result {
                Ok(s) => {
                    println!(" done.");
                    println!("  Frames:   {}", s.total_frames);
                    println!("  Mean ξ:   {:.4}", s.mean_xi);
                    println!("  SC1 (silence): {} ({:.1}%)", s.sc1_frames,
                        100.0 * s.sc1_frames as f64 / s.total_frames as f64);
                    println!("  SC2 (tonal):   {} ({:.1}%)", s.sc2_frames,
                        100.0 * s.sc2_frames as f64 / s.total_frames as f64);
                    println!("  SC3 (attack):  {} ({:.1}%)", s.sc3_frames,
                        100.0 * s.sc3_frames as f64 / s.total_frames as f64);
                }
                Err(e) => eprintln!("Error: {e}"),
            }
        }

        Commands::Decode { input, output, attack, tone, soul } => {
            let params = DecodeParams { tone_gain: tone, attack_gain: attack, soul_gain: soul };
            println!("Decoding {} → {} (∇φ×{tone:.2} δψ×{attack:.2} h×{soul:.2})", input, output);
            match decode_hodge(&input, &output, params) {
                Ok(s) => println!("  Done. {} frames, {} samples, {} Hz",
                    s.total_frames, s.total_samples, s.sample_rate),
                Err(e) => eprintln!("Error: {e}"),
            }
        }

        Commands::Info { input } => {
            match info_hodge(&input) {
                Ok(s) => println!("{s}"),
                Err(e) => eprintln!("Error: {e}"),
            }
        }

        Commands::Stem { input, prefix } => {
            let stems = [
                ("_tone.wav",   1.0f32, 0.0f32, 0.0f32),
                ("_attack.wav", 0.0,    1.0,    0.0),
                ("_soul.wav",   0.0,    0.0,    1.0),
            ];
            for (suffix, tone, attack, soul) in &stems {
                let out = format!("{prefix}{suffix}");
                let params = DecodeParams { tone_gain: *tone, attack_gain: *attack, soul_gain: *soul };
                print!("  Stem {} → {}...", suffix.trim_start_matches('_').trim_end_matches(".wav"), out);
                match decode_hodge(&input, &out, params) {
                    Ok(s) => println!(" {} samples", s.total_samples),
                    Err(e) => eprintln!(" Error: {e}"),
                }
            }
        }

        Commands::Fingerprint { input, compare } => {
            match fingerprint_hodge(&input) {
                Ok(fp) => {
                    println!("{}\n  File: {}", fp, input);
                    if let Some(cmp) = compare {
                        match fingerprint_hodge(&cmp) {
                            Ok(fp2) => {
                                println!("\n{}\n  File: {}", fp2, cmp);
                                println!("\n  Similarity: {:.4} ({:.1}%)",
                                    fp.similarity(&fp2), fp.similarity(&fp2) * 100.0);
                            }
                            Err(e) => eprintln!("Error comparing: {e}"),
                        }
                    }
                }
                Err(e) => eprintln!("Error: {e}"),
            }
        }

        Commands::Compare { reference, decoded } => {
            fn read_wav_f32(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
                let mut r = WavReader::open(path)?;
                let s: Vec<f32> = match r.spec().sample_format {
                    hound::SampleFormat::Int => r.samples::<i16>()
                        .map(|s| s.map(|x| x as f32 / i16::MAX as f32))
                        .collect::<Result<_, _>>()?,
                    hound::SampleFormat::Float => r.samples::<f32>()
                        .collect::<Result<_, _>>()?,
                };
                Ok(s)
            }
            match (read_wav_f32(&reference), read_wav_f32(&decoded)) {
                (Ok(ref_s), Ok(dec_s)) => {
                    let n = ref_s.len().min(dec_s.len()) as f64;
                    let sig_pow: f64 = ref_s.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / n;
                    let noise_pow: f64 = ref_s.iter().zip(dec_s.iter())
                        .map(|(&r, &d)| ((r - d) as f64).powi(2)).sum::<f64>() / n;
                    let rmse = noise_pow.sqrt();
                    let snr_db = if noise_pow > 1e-15 { 10.0 * (sig_pow / noise_pow).log10() } else { f64::INFINITY };
                    let max_err = ref_s.iter().zip(dec_s.iter())
                        .map(|(&r, &d)| (r - d).abs())
                        .fold(0.0f32, f32::max);
                    let samples_cmp = ref_s.len().min(dec_s.len());
                    println!("Quality comparison: {} vs {}", reference, decoded);
                    println!("  Samples compared: {}", samples_cmp);
                    println!("  SNR:      {:.2} dB", snr_db);
                    println!("  RMSE:     {:.6}", rmse);
                    println!("  Max err:  {:.6}", max_err);
                    println!("  Verdict:  {}", match snr_db as u32 {
                        u32::MAX        => "lossless",
                        50..            => "transparent (studio quality)",
                        40..=49         => "excellent (broadcast quality)",
                        30..=39         => "good (MP3 320kbps equivalent)",
                        20..=29         => "acceptable (MP3 128kbps equivalent)",
                        _               => "audible degradation",
                    });
                }
                (Err(e), _) | (_, Err(e)) => eprintln!("Error: {e}"),
            }
        }

        Commands::Watermark { action, input, output, author } => {
            match action {
                WmAction::Embed => {
                    let out = output.as_deref().unwrap_or("watermarked.wav");
                    let auth = author.as_deref().unwrap_or("MDIACH.EML");
                    match watermark_encode(&input, out, auth) {
                        Ok(n) => {
                            println!("Watermark embedded: \"{}\" ({} bytes)", auth, n);
                            println!("  Output: {}", out);
                            println!("  Method: LSB steganography, stride-spread, CRC32 verified");
                            println!("  Audible diff: ±1 LSB = 0.003 dB (inaudible)");
                        }
                        Err(e) => eprintln!("Error: {e}"),
                    }
                }
                WmAction::Decode => {
                    match watermark_decode(&input) {
                        Ok((auth, verified)) => {
                            println!("Watermark found:");
                            println!("  Author:   {}", auth);
                            println!("  CRC32:    {}", if verified { "✓ verified" } else { "✗ corrupted" });
                        }
                        Err(e) => println!("  {e}"),
                    }
                }
            }
        }

        Commands::Mix { input_a, input_b, output, tone_a, attack_a, soul_a } => {
            println!("Mixing:\n  A={input_a}\n  B={input_b}");
            println!("  ∇φ: {tone_a:.2}A+{:.2}B  δψ: {attack_a:.2}A+{:.2}B  h: {soul_a:.2}A+{:.2}B",
                1.0 - tone_a, 1.0 - attack_a, 1.0 - soul_a);
            let params = MixParams { tone_a, attack_a, soul_a };
            match mix_hodge(&input_a, &input_b, &output, params) {
                Ok(s) => println!("  Done. {} frames @ {} Hz → {}", s.total_frames, s.sample_rate, output),
                Err(e) => eprintln!("Error: {e}"),
            }
        }
    }
}
