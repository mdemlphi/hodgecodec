// M31 Phase 1 — Hodge Fingerprint invariance benchmark
// Usage: hodge-fp-bench <input.wav>
// Applies transforms via sox/ffmpeg, measures similarity vs original.

use std::path::Path;
use std::process::Command;
use hodgecodec::fingerprint_wav;

fn load_wav_mono(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // normalize to 44100 mono WAV first via ffmpeg
    let tmp = format!("/tmp/fp_norm_{}.wav", std::process::id());
    Command::new("ffmpeg").args(["-y","-i",path,"-ar","44100","-ac","1","-f","wav",&tmp])
        .stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null())
        .status()?;
    let mut r = hound::WavReader::open(&tmp)?;
    let spec = r.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().filter_map(|s|s.ok()).collect(),
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample-1)) as f32;
            r.samples::<i32>().filter_map(|s|s.ok()).map(|s| s as f32/max).collect()
        }
    };
    let _ = std::fs::remove_file(&tmp);
    Ok(samples)
}

fn transform_sox(input: &str, args: &[&str]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let tmp = format!("/tmp/fp_sox_{}.wav", std::process::id());
    let mut cmd = Command::new("sox");
    cmd.arg(input);
    cmd.args(["-r","44100","-c","1",&tmp]);
    cmd.args(args);
    cmd.stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null());
    cmd.status()?;
    let result = load_wav_mono(&tmp);
    let _ = std::fs::remove_file(&tmp);
    result
}

fn transform_mp3(input: &str, kbps: u32) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let tmp_mp3 = format!("/tmp/fp_mp3_{}.mp3", std::process::id());
    let tmp_wav = format!("/tmp/fp_mp3_out_{}.wav", std::process::id());
    Command::new("ffmpeg").args(["-y","-i",input,"-b:a",&format!("{}k",kbps),&tmp_mp3])
        .stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).status()?;
    Command::new("ffmpeg").args(["-y","-i",&tmp_mp3,"-ar","44100","-ac","1",&tmp_wav])
        .stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).status()?;
    let result = load_wav_mono(&tmp_wav);
    let _ = std::fs::remove_file(&tmp_mp3);
    let _ = std::fs::remove_file(&tmp_wav);
    result
}

fn sim(orig: &[f32], transformed: &[f32]) -> f32 {
    let fp1 = hodgecodec::fingerprint_pcm(orig, 44100);
    let fp2 = hodgecodec::fingerprint_pcm(transformed, 44100);
    fp1.similarity(&fp2)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: hodge-fp-bench <input.wav>");
        std::process::exit(1);
    }
    let input = &args[1];

    println!("M31 Hodge Fingerprint Invariance Benchmark");
    println!("Input: {}", input);
    println!("{:-<55}", "");

    let orig = match load_wav_mono(input) {
        Ok(s) => s,
        Err(e) => { eprintln!("load error: {e}"); std::process::exit(1); }
    };
    let fp_orig = hodgecodec::fingerprint_pcm(&orig, 44100);
    println!("Original: {:.2}s  frames={}", fp_orig.duration_s, fp_orig.total_frames);
    println!("{:-<55}", "");
    println!("{:<35} {:>8}  {}", "Transform", "Hodge-h", "Pass?");
    println!("{:-<55}", "");

    // baseline self-similarity
    println!("{:<35} {:>8.4}  {}", "Original (self)", sim(&orig, &orig),
        if sim(&orig, &orig) > 0.99 { "✅" } else { "❌" });

    let transforms: &[(&str, Box<dyn Fn() -> Result<Vec<f32>,Box<dyn std::error::Error>>>)] = &[
        ("Pitch +2st (~12%)",  Box::new(|| transform_sox(input, &["pitch","200"]))),
        ("Pitch +5st (~33%)",  Box::new(|| transform_sox(input, &["pitch","500"]))),
        ("Pitch -2st",         Box::new(|| transform_sox(input, &["pitch","-200"]))),
        ("Speed ×1.25",        Box::new(|| transform_sox(input, &["speed","1.25"]))),
        ("Speed ×0.75",        Box::new(|| transform_sox(input, &["speed","0.75"]))),
        ("Tempo ×1.25",        Box::new(|| transform_sox(input, &["tempo","1.25"]))),
        ("Reverb",             Box::new(|| transform_sox(input, &["reverb","50"]))),
        ("Noise +20dB SNR",    Box::new(|| transform_sox(input, &["synth","whitenoise","vol","0.01",":", "mix-power",input]))),
        ("MP3 128kbps",        Box::new(|| transform_mp3(input, 128))),
        ("MP3 64kbps",         Box::new(|| transform_mp3(input, 64))),
        ("MP3 32kbps",         Box::new(|| transform_mp3(input, 32))),
        ("Gain ×0.1",          Box::new(|| { let s: Vec<f32> = orig.iter().map(|x|x*0.1).collect(); Ok(s) })),
    ];

    for (name, f) in transforms {
        match f() {
            Ok(transformed) => {
                let s = sim(&orig, &transformed);
                let pass = if s > 0.75 { "✅" } else { "❌" };
                println!("{:<35} {:>8.4}  {}", name, s, pass);
            }
            Err(e) => println!("{:<35} {:>8}  ⚠ {}", name, "ERR", e),
        }
    }
    println!("{:-<55}", "");
    println!("threshold >0.75 = same track (Hodge-h invariant)");
}
