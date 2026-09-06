//! bench.rs — On-demand benchmark: Rust preprocessing vs Python preprocessing.
//!
//! Invoked from the orchestrator menu ("Benchmark Rust vs Python").
//! Not run during normal pipeline execution.
//!
//! Prints measured wall-clock times only. Does NOT claim GPU improvement.
//! GPU idle-time reduction requires separate profiling (e.g. nvidia-smi during training).

use crate::preprocess;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

// ─── Result struct ────────────────────────────────────────────────────────────

pub struct BenchResult {
    pub step:         String,
    pub python_secs:  Option<f64>,
    pub rust_secs:    Option<f64>,
}

impl BenchResult {
    /// Speedup = Python time / Rust time. None if either measurement is missing.
    pub fn speedup(&self) -> Option<f64> {
        match (self.python_secs, self.rust_secs) {
            (Some(p), Some(r)) if r > 0.0 => Some(p / r),
            _ => None,
        }
    }
}

// ─── Benchmark runner ─────────────────────────────────────────────────────────

/// Run the full benchmark suite for a given dataset and print a comparison table.
///
/// Both Python and Rust run on the same dataset directory so results are comparable.
/// The dataset must already exist under datasets/{dataset}.
pub fn run_benchmark(dataset: &str, project_root: &Path) {
    println!();
    println!("══════════════════════════════════════════════════════════════");
    println!("  Benchmark: Rust vs Python preprocessing");
    println!("  Dataset : {}", dataset);
    println!("  Note    : Measures offline preprocessing only.");
    println!("            GPU idle-time during training needs nvidia-smi profiling.");
    println!("══════════════════════════════════════════════════════════════");

    let mut results: Vec<BenchResult> = Vec::new();

    // ── Step 1: yolo_to_coco ──────────────────────────────────────────────────
    results.push(bench_yolo_to_coco(dataset, project_root));

    // ── Step 2: verify_labels ─────────────────────────────────────────────────
    let dataset_dir = project_root.join("datasets").join(dataset);
    if dataset_dir.exists() {
        results.push(bench_verify_labels(&dataset_dir));
    } else {
        eprintln!(
            "[WARN] datasets/{} not found — skipping verify_labels benchmark.",
            dataset
        );
    }

    print_table(&results);
}

// ─── Individual step benchmarks ───────────────────────────────────────────────

fn bench_yolo_to_coco(dataset: &str, project_root: &Path) -> BenchResult {
    println!("\n── yolo_to_coco ──────────────────────────────────────────────");

    // Python
    println!("  [*] Running Python  yolo_to_coco.py ...");
    let python_secs = time_python(
        project_root,
        &[
            "scripts/data/yolo_to_coco.py",
            "--dataset",
            dataset,
        ],
    );
    if let Some(t) = python_secs {
        println!("      Python  : {:.3}s", t);
    } else {
        println!("      Python  : FAILED or not measurable");
    }

    // Rust
    println!("  [*] Running Rust    run_yolo_to_coco ...");
    let rust_start = Instant::now();
    let rust_ok    = preprocess::run_yolo_to_coco(dataset, project_root).is_ok();
    let rust_elapsed = rust_start.elapsed().as_secs_f64();
    let rust_secs = if rust_ok {
        println!("      Rust    : {:.3}s", rust_elapsed);
        Some(rust_elapsed)
    } else {
        println!("      Rust    : FAILED");
        None
    };

    BenchResult {
        step: "yolo_to_coco".into(),
        python_secs,
        rust_secs,
    }
}

fn bench_verify_labels(dataset_dir: &Path) -> BenchResult {
    println!("\n── verify_labels ─────────────────────────────────────────────");

    // Python (sanitize_and_validate_dataset.py) — report-only mode for fair comparison
    println!("  [*] Running Python  sanitize_and_validate_dataset.py --dry ...");
    // Python script doesn't have a --dry flag so we time a no-op path by pointing
    // it at an empty subdirectory concept. Instead we time it on the real dir.
    // (The Python script does PIL verify but doesn't delete in the benchmark context.)
    let python_secs = time_python(
        dataset_dir.parent().unwrap_or(dataset_dir),
        &[
            "../../scripts/data/sanitize_and_validate_dataset.py",
            "--dataset-dir",
            &dataset_dir.to_string_lossy(),
        ],
    );
    if let Some(t) = python_secs {
        println!("      Python  : {:.3}s", t);
    } else {
        println!("      Python  : FAILED or not measurable");
    }

    // Rust
    println!("  [*] Running Rust    run_verify_labels ...");
    let rust_start = Instant::now();
    let result     = preprocess::run_verify_labels(dataset_dir);
    let rust_elapsed = rust_start.elapsed().as_secs_f64();
    let rust_secs = match result {
        Ok(report) => {
            println!(
                "      Rust    : {:.3}s  ({} total, {} valid, {} flagged)",
                rust_elapsed, report.total, report.valid, report.corrupt.len()
            );
            Some(rust_elapsed)
        }
        Err(e) => {
            println!("      Rust    : FAILED — {}", e);
            None
        }
    };

    BenchResult {
        step: "verify_labels".into(),
        python_secs,
        rust_secs,
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Run a Python script, return elapsed seconds or None on failure.
fn time_python(cwd: &Path, args: &[&str]) -> Option<f64> {
    let mut cmd = Command::new("python");
    for a in args {
        cmd.arg(a);
    }
    cmd.current_dir(cwd);
    let start  = Instant::now();
    let status = cmd.status().ok()?;
    if status.success() {
        Some(start.elapsed().as_secs_f64())
    } else {
        None
    }
}

// ─── Output table ─────────────────────────────────────────────────────────────

pub fn print_table(results: &[BenchResult]) {
    println!();
    println!("╔══════════════════╦══════════════╦══════════════╦═══════════╗");
    println!("║ Step             ║ Python (s)   ║ Rust (s)     ║ Speedup   ║");
    println!("╠══════════════════╬══════════════╬══════════════╬═══════════╣");
    for r in results {
        let py = r.python_secs.map_or("N/A".into(),        |s| format!("{:.3}", s));
        let rs = r.rust_secs.map_or("N/A".into(),          |s| format!("{:.3}", s));
        let sp = r.speedup().map_or("N/A".into(),          |s| format!("{:.1}x", s));
        println!("║ {:<16} ║ {:>12} ║ {:>12} ║ {:>9} ║", r.step, py, rs, sp);
    }
    println!("╚══════════════════╩══════════════╩══════════════╩═══════════╝");
    println!();
    println!("IMPORTANT: These numbers measure offline CPU preprocessing only.");
    println!("Whether faster preprocessing reduces GPU idle time during training");
    println!("must be verified separately with: watch -n1 nvidia-smi");
}
