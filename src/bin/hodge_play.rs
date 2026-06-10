// hodge-play — real-time .hodge playback with ξ terminal visualizer
// hodge-play track.hodge [--attack 2.0] [--tone 0.5] [--soul 1.0]

use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hodgecodec::{decode_hodge_with_stats, DecodeParams, ScClass};
use std::sync::{Arc, Mutex};

#[derive(Parser)]
#[command(name = "hodge-play", about = "Real-time .hodge playback with ξ visualizer")]
struct Args {
    input: String,
    #[arg(long, default_value_t = 1.0)] attack: f32,
    #[arg(long, default_value_t = 1.0)] tone: f32,
    #[arg(long, default_value_t = 1.0)] soul: f32,
    /// Hide ξ visualizer
    #[arg(long)] no_vis: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let params = DecodeParams {
        tone_gain: args.tone,
        attack_gain: args.attack,
        soul_gain: args.soul,
    };

    print!("Decoding {}...", args.input);
    let (pcm, sample_rate, frame_size, frame_stats) = decode_hodge_with_stats(&args.input, params)?;
    let total_samples = pcm.len();
    let total_frames = frame_stats.len();
    let duration_s = total_samples as f64 / sample_rate as f64;
    println!(" {} frames @ {} Hz  ({:.1}s)", total_frames, sample_rate, duration_s);
    println!("  ∇φ×{:.2}  δψ×{:.2}  h×{:.2}", args.tone, args.attack, args.soul);

    let host = cpal::default_host();
    let device = host.default_output_device().ok_or("no audio output device")?;
    println!("Device: {}", device.name()?);

    let desired = cpal::SampleRate(sample_rate);
    let supported = device
        .supported_output_configs()?
        .find(|c| c.min_sample_rate() <= desired && desired <= c.max_sample_rate())
        .ok_or(format!("device does not support {} Hz", sample_rate))?
        .with_sample_rate(desired);

    let channels = supported.channels() as usize;
    let pcm = Arc::new(pcm);
    let frame_stats = Arc::new(frame_stats);
    let pos = Arc::new(Mutex::new(0usize));

    let stream = match supported.sample_format() {
        cpal::SampleFormat::F32 => {
            let (p, c) = (Arc::clone(&pcm), Arc::clone(&pos));
            device.build_output_stream(
                &supported.into(),
                move |data: &mut [f32], _| fill_f32(data, &p, &c, channels),
                |e| eprintln!("cpal error: {e}"),
                None,
            )?
        }
        cpal::SampleFormat::I16 => {
            let (p, c) = (Arc::clone(&pcm), Arc::clone(&pos));
            device.build_output_stream(
                &supported.into(),
                move |data: &mut [i16], _| fill_i16(data, &p, &c, channels),
                |e| eprintln!("cpal error: {e}"),
                None,
            )?
        }
        cpal::SampleFormat::U16 => {
            let (p, c) = (Arc::clone(&pcm), Arc::clone(&pos));
            device.build_output_stream(
                &supported.into(),
                move |data: &mut [u16], _| fill_u16(data, &p, &c, channels),
                |e| eprintln!("cpal error: {e}"),
                None,
            )?
        }
        fmt => return Err(format!("unsupported sample format: {fmt:?}").into()),
    };

    stream.play()?;
    println!("Playing... (Ctrl+C to stop)\n");

    while *pos.lock().unwrap() < total_samples {
        let current_sample = *pos.lock().unwrap();
        let frame_idx = (current_sample / frame_size).min(total_frames.saturating_sub(1));
        let elapsed_s = current_sample as f64 / sample_rate as f64;

        if !args.no_vis && frame_idx < frame_stats.len() {
            let (xi, sc) = frame_stats[frame_idx];
            let bar_fill = (xi * 24.0) as usize;
            let bar: String = format!("{}{}", "█".repeat(bar_fill), "░".repeat(24 - bar_fill));
            let sc_label = match sc {
                ScClass::SC1 => "SC1 ·",
                ScClass::SC2 => "SC2 ~",
                ScClass::SC3 => "SC3 !",
            };
            print!("\r  [{bar}] ξ={xi:.3} {sc_label} | {elapsed_s:.1}s/{duration_s:.1}s  ");
            let _ = std::io::Write::flush(&mut std::io::stdout());
        }

        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    std::thread::sleep(std::time::Duration::from_millis(200));
    println!("\nDone.");
    Ok(())
}

fn fill_f32(data: &mut [f32], pcm: &[f32], pos: &Mutex<usize>, ch: usize) {
    let mut p = pos.lock().unwrap();
    for frame in data.chunks_mut(ch) {
        let s = if *p < pcm.len() { pcm[*p].clamp(-1.0, 1.0) } else { 0.0 };
        for out in frame { *out = s; }
        *p += 1;
    }
}

fn fill_i16(data: &mut [i16], pcm: &[f32], pos: &Mutex<usize>, ch: usize) {
    let mut p = pos.lock().unwrap();
    for frame in data.chunks_mut(ch) {
        let s = if *p < pcm.len() { (pcm[*p].clamp(-1.0, 1.0) * i16::MAX as f32) as i16 } else { 0 };
        for out in frame { *out = s; }
        *p += 1;
    }
}

fn fill_u16(data: &mut [u16], pcm: &[f32], pos: &Mutex<usize>, ch: usize) {
    let mut p = pos.lock().unwrap();
    for frame in data.chunks_mut(ch) {
        let s = if *p < pcm.len() {
            ((pcm[*p].clamp(-1.0, 1.0) + 1.0) * 0.5 * u16::MAX as f32) as u16
        } else { u16::MAX / 2 };
        for out in frame { *out = s; }
        *p += 1;
    }
}
