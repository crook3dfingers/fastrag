//! `fastrag doctor` — check environment for llama-server and report its
//! version, model cache dir, and any issues.

use std::process::Command;

pub fn run() {
    println!("fastrag doctor");
    println!("==============");

    // 1. llama-server in PATH?
    let which = Command::new("which")
        .arg("llama-server")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());

    match &which {
        Some(path) => {
            println!("llama-server: {path}");

            // 2. Version
            let ver = Command::new(path).arg("--version").output().ok();
            match ver {
                Some(o) if o.status.success() => {
                    let stdout = String::from_utf8_lossy(&o.stdout);
                    let line = stdout.lines().next().unwrap_or("(empty)");
                    println!("version:      {line}");
                }
                Some(o) => {
                    let stderr = String::from_utf8_lossy(&o.stderr);
                    println!("version:      ERROR (exit {}): {}", o.status, stderr.trim());
                }
                None => println!("version:      ERROR (failed to execute)"),
            }
        }
        None => {
            println!("llama-server: NOT FOUND");
            println!("  install from https://github.com/ggml-org/llama.cpp/releases");
            println!("  or set LLAMA_SERVER_PATH env var");
        }
    }

    // 3. LLAMA_SERVER_PATH override?
    if let Ok(p) = std::env::var("LLAMA_SERVER_PATH") {
        println!("LLAMA_SERVER_PATH: {p}");
    }

    // 4. FASTRAG_MODEL_DIR / default cache
    if let Ok(p) = std::env::var("FASTRAG_MODEL_DIR") {
        println!("FASTRAG_MODEL_DIR: {p}");
    } else if let Some(cache) = dirs::cache_dir() {
        let model_dir = cache.join("fastrag").join("models");
        println!(
            "model cache:  {} ({})",
            model_dir.display(),
            if model_dir.exists() {
                "exists"
            } else {
                "not created yet"
            }
        );
    }

    // 5. Contextualizer (only when contextual-llama is compiled in)
    #[cfg(feature = "contextual-llama")]
    check_contextualizer();
}

#[cfg(feature = "contextual-llama")]
fn check_contextualizer() {
    use fastrag_embed::llama_cpp::{
        DefaultCompletionPreset, HfHubDownloader, resolve_model_path_default,
    };

    println!();
    println!("contextualizer:");
    println!("  preset:    {}", DefaultCompletionPreset::MODEL_ID);
    println!("  hf_repo:   {}", DefaultCompletionPreset::HF_REPO);
    println!("  gguf_file: {}", DefaultCompletionPreset::GGUF_FILE);
    println!("  ctx_size:  {}", DefaultCompletionPreset::CONTEXT_WINDOW);
    match resolve_model_path_default(&DefaultCompletionPreset::model_source(), &HfHubDownloader) {
        Ok(p) => println!("  resolved:  {}", p.display()),
        Err(e) => println!("  resolved:  ERROR — {e}"),
    }
}
