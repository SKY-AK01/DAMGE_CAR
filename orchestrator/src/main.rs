use chrono::Local;
use rust_xlsxwriter::{Format, Workbook};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Command, Stdio};

mod preprocess;
mod bench;


// ─── Metric structs ───────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug, Clone)]
struct MetricData {
    model_name: String,
    epoch: u32,
    train_stats: HashMap<String, serde_json::Value>,
    val_metrics: Option<HashMap<String, serde_json::Value>>,
    best_mask_map50: f64,
    is_best: bool,
}

// ─── Environment checks ───────────────────────────────────────────────────────

fn check_venv() {
    if let Ok(venv) = env::var("VIRTUAL_ENV") {
        println!("[OK] venv active: {}", venv);
    } else {
        println!("[INFO] No venv active — proceeding with system Python.");
    }
}

fn check_azure_disk() {
    if let Ok(cwd) = env::current_dir() {
        let cwd_str = cwd.to_string_lossy();
        if cwd_str.starts_with("/mnt/") {
            eprintln!("############################################################");
            eprintln!("  WARNING: Running from {}", cwd_str);
            eprintln!("  If this is Azure's temporary resource disk, it will be WIPED");
            eprintln!("  when you Stop (Deallocate) the VM. Move to /home first!");
            eprintln!("############################################################");
            std::thread::sleep(std::time::Duration::from_secs(3));
        }
    }
}

// ─── I/O helpers ──────────────────────────────────────────────────────────────

fn prompt(msg: &str) -> String {
    print!("{}", msg);
    io::stdout().flush().unwrap();
    let mut buf = String::new();
    io::stdin().read_line(&mut buf).unwrap();
    buf.trim().to_string()
}

fn prompt_default(label: &str, default: &str) -> String {
    let val = prompt(&format!("  {} [default: {}]: ", label, default));
    if val.is_empty() { default.to_string() } else { val }
}

fn prompt_u32(label: &str, default: u32) -> u32 {
    let val = prompt_default(label, &default.to_string());
    val.parse().unwrap_or(default)
}

fn prompt_i32(label: &str, default: i32) -> i32 {
    let val = prompt_default(label, &default.to_string());
    val.parse().unwrap_or(default)
}

// ─── Subprocess execution (live logging) ─────────────────────────────────────

/// Spawn a command, stream its stdout/stderr to terminal AND a log file line-by-line.
fn run_and_log(mut cmd: Command, log_path: &str, label: &str) -> bool {
    let mut log_file = OpenOptions::new()
        .create(true).append(true)
        .open(log_path)
        .expect("Failed to open log file");

    writeln!(
        log_file, "=== {} | {} ===",
        label, Local::now().format("%Y-%m-%d %H:%M:%S")
    ).unwrap();

    let mut child = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn process");

    if let Some(stdout) = child.stdout.take() {
        let mut lf = log_file.try_clone().unwrap();
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            println!("[{}] {}", label, line);
            let _ = writeln!(lf, "{}", line);
        }
    }
    if let Some(stderr) = child.stderr.take() {
        let mut lf = log_file.try_clone().unwrap();
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            eprintln!("[{}][ERR] {}", label, line);
            let _ = writeln!(lf, "[ERR] {}", line);
        }
    }

    child.wait().expect("Failed to wait on child").success()
}

/// Same as `run_and_log` but also pipes pre-collected data to the process's stdin.
/// Used for `download_and_prepare_datasets.sh` which reads a menu choice from stdin.
fn run_with_stdin_and_log(mut cmd: Command, stdin_data: &str, log_path: &str, label: &str) -> bool {
    let mut log_file = OpenOptions::new()
        .create(true).append(true)
        .open(log_path)
        .expect("Failed to open log file");

    writeln!(
        log_file, "=== {} | {} ===",
        label, Local::now().format("%Y-%m-%d %H:%M:%S")
    ).unwrap();

    let mut child = cmd
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn process");

    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(stdin_data.as_bytes());
        // stdin dropped here, signalling EOF to the child
    }
    if let Some(stdout) = child.stdout.take() {
        let mut lf = log_file.try_clone().unwrap();
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            println!("[{}] {}", label, line);
            let _ = writeln!(lf, "{}", line);
        }
    }
    if let Some(stderr) = child.stderr.take() {
        let mut lf = log_file.try_clone().unwrap();
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            eprintln!("[{}][ERR] {}", label, line);
            let _ = writeln!(lf, "[ERR] {}", line);
        }
    }

    child.wait().expect("Failed to wait on child").success()
}

// ─── Interactive menus ────────────────────────────────────────────────────────

fn display_main_menu() {
    println!();
    println!("======================================================================");
    println!("              Car-Parts Segmentation — Pipeline Orchestrator          ");
    println!("======================================================================");
    println!("  1) Annotate Images       — Auto-annotate dataset (DINO + SAM2)");
    println!("  2) Train Model           — Prepare dataset & train YOLO / Mask R-CNN / Fast R-CNN");
    println!("  3) Test on Trained Model — Run inference on test images");
    println!("  4) Run Full Pipeline     — Dataset Prep → Train → Evaluate");
    println!("  5) Train on Azure ML    — Auto-upload dataset & submit training job to Azure ML");
    println!("  6) Setup & Clean         — Folder setup and log archive");
    println!("  7) Preprocess (Rust)     — Fast parallel dataset preprocessing");
    println!("  8) Benchmark             — Rust vs Python preprocessing speed");
    println!("  9) Exit");
    println!("======================================================================");
}

struct ModelSelection {
    yolo: bool,
    maskrcnn: bool,
    fastrcnn: bool,
}

fn collect_models() -> ModelSelection {
    println!();
    println!("======================================================================");
    println!(" Select models to train");
    println!("======================================================================");
    println!("  1) YOLO only");
    println!("  2) Mask R-CNN only");
    println!("  3) Fast R-CNN only");
    println!("  4) YOLO + Mask R-CNN");
    println!("  5) YOLO + Fast R-CNN");
    println!("  6) Mask R-CNN + Fast R-CNN");
    println!("  7) All models");
    println!("======================================================================");
    let choice = prompt("Enter choice [1-7]: ");
    match choice.as_str() {
        "1" => ModelSelection { yolo: true,  maskrcnn: false, fastrcnn: false },
        "2" => ModelSelection { yolo: false, maskrcnn: true,  fastrcnn: false },
        "3" => ModelSelection { yolo: false, maskrcnn: false, fastrcnn: true  },
        "4" => ModelSelection { yolo: true,  maskrcnn: true,  fastrcnn: false },
        "5" => ModelSelection { yolo: true,  maskrcnn: false, fastrcnn: true  },
        "6" => ModelSelection { yolo: false, maskrcnn: true,  fastrcnn: true  },
        _   => ModelSelection { yolo: true,  maskrcnn: true,  fastrcnn: true  }, // 7 + default
    }
}

struct Hparams {
    yolo_epochs:  u32, yolo_batch:  i32, yolo_workers:  u32,
    mrcnn_epochs: u32, mrcnn_batch: i32, mrcnn_workers: u32,
    fastrcnn_epochs: u32, fastrcnn_batch: i32, fastrcnn_workers: u32,
}

fn collect_hparams() -> Hparams {
    println!();
    println!("======================================================================");
    println!(" Set training hyperparameters  (press Enter to accept each default)");
    println!("======================================================================");

    println!("\n---- YOLOv11m-seg ----");
    let yolo_epochs  = prompt_u32("Epochs", 50);
    let yolo_batch   = prompt_i32("Batch size (-1 = auto)", -1);
    let yolo_workers = prompt_u32("Dataloader workers", 16);

    println!("\n---- Mask R-CNN ----");
    let mrcnn_epochs  = prompt_u32("Epochs", 20);
    let mrcnn_batch   = prompt_i32("Batch size", 2);
    let mrcnn_workers = prompt_u32("Dataloader workers", 8);

    println!("\n---- Fast R-CNN ----");
    let fastrcnn_epochs  = prompt_u32("Epochs", 20);
    let fastrcnn_batch   = prompt_i32("Batch size", 4);
    let fastrcnn_workers = prompt_u32("Dataloader workers", 8);

    println!();
    println!("---- Summary ----");
    println!("  YOLOv11m-seg : epochs={}  batch={}  workers={}", yolo_epochs, yolo_batch, yolo_workers);
    println!("  Mask R-CNN   : epochs={}  batch={}  workers={}", mrcnn_epochs, mrcnn_batch, mrcnn_workers);
    println!("  Fast R-CNN   : epochs={}  batch={}  workers={}", fastrcnn_epochs, fastrcnn_batch, fastrcnn_workers);
    println!("-----------------");

    Hparams {
        yolo_epochs, yolo_batch, yolo_workers,
        mrcnn_epochs, mrcnn_batch, mrcnn_workers,
        fastrcnn_epochs, fastrcnn_batch, fastrcnn_workers,
    }
}

// ─── Dataset preparation ──────────────────────────────────────────────────────

fn run_dataset_prep(logs_dir: &str, project_root: &str) -> String {
    println!();
    println!("======================================================================");
    println!(" Dataset Source");
    println!("======================================================================");
    println!("  1) Local RAW_DATASET only");
    println!("  2) Download from online only (Ultralytics + DSMLR)");
    println!("  3) Both — download + combine with RAW_DATASET  (recommended)");
    println!("======================================================================");
    let data_choice = prompt("Enter choice [1-3]: ");

    let log_path = format!("{}/01_dataset_prep.log", logs_dir);
    let mut cmd = Command::new("bash");
    cmd.arg(format!("{}/scripts/data/download_and_prepare_datasets.sh", project_root))
        .current_dir(project_root);

    println!("\n[*] Running dataset preparation — this may take a few minutes...");
    let ok = run_with_stdin_and_log(cmd, &format!("{}\n", data_choice), &log_path, "dataset_prep");

    if !ok {
        eprintln!("[FAILED] Dataset preparation failed. Log: {}", log_path);
        std::process::exit(1);
    }
    println!("[OK] Dataset preparation complete.");

    // Read the choice that the bash script persisted
    let saved = fs::read_to_string(format!("{}/datasets/.dataset_choice.txt", project_root))
        .unwrap_or_else(|_| data_choice.clone())
        .trim().to_string();

    match saved.as_str() {
        "1" => "custom_carparts".to_string(),
        "3" => "combined_carparts".to_string(),
        _   => "carparts-seg".to_string(),
    }
}

// ─── Training ─────────────────────────────────────────────────────────────────

fn collect_metrics(out_dir: &str) -> Vec<MetricData> {
    let mut metrics = Vec::new();
    if let Ok(entries) = fs::read_dir(out_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Ok(content) = fs::read_to_string(&path) {
                    if let Ok(data) = serde_json::from_str::<MetricData>(&content) {
                        metrics.push(data);
                    }
                }
            }
        }
    }
    metrics.sort_by_key(|m| m.epoch);
    metrics
}

fn run_training(
    models: &ModelSelection,
    hp: &Hparams,
    dataset: &str,
    base_out_dir: &str,
    logs_dir: &str,
    project_root: &str,
) -> HashMap<String, Vec<MetricData>> {
    let mut all_metrics: HashMap<String, Vec<MetricData>> = HashMap::new();

    // ── YOLO ──────────────────────────────────────────────────────────────────
    if models.yolo {
        let out_dir = format!("{}/yolo11m-seg", base_out_dir);
        fs::create_dir_all(&out_dir).unwrap();
        let log_path = format!("{}/02_train_yolo.log", logs_dir);

        println!("\n========================================");
        println!(" Training: YOLO11m-seg  ({} epochs)", hp.yolo_epochs);
        println!("========================================");

        let mut cmd = Command::new("python");
        cmd.arg(format!("{}/scripts/training/train_yolo_seg.py", project_root))
            .arg("--model").arg("yolo11m-seg")
            .arg("--dataset").arg(dataset)
            .arg("--epochs").arg(hp.yolo_epochs.to_string())
            .arg("--batch").arg(hp.yolo_batch.to_string())
            .arg("--workers").arg(hp.yolo_workers.to_string())
            .arg("--project").arg(&out_dir)
            .current_dir(project_root);

        if run_and_log(cmd, &log_path, "yolo11m-seg") {
            println!("[OK] YOLO training complete. Log: {}", log_path);
        } else {
            eprintln!("[ERROR] YOLO training failed. Log: {}", log_path);
        }
        all_metrics.insert("yolo11m-seg".to_string(), collect_metrics(&out_dir));
    }


    // ── Mask R-CNN ────────────────────────────────────────────────────────────
    if models.maskrcnn {
        let out_dir = format!("{}/maskrcnn", base_out_dir);
        fs::create_dir_all(&out_dir).unwrap();
        let log_path = format!("{}/04_train_maskrcnn.log", logs_dir);

        println!("\n========================================");
        println!(" Training: Mask R-CNN  ({} epochs)", hp.mrcnn_epochs);
        println!("========================================");

        let mut cmd = Command::new("python");
        cmd.arg(format!("{}/scripts/training/train_maskrcnn.py", project_root))
            .arg("--dataset").arg(dataset)
            .arg("--epochs").arg(hp.mrcnn_epochs.to_string())
            .arg("--batch").arg(hp.mrcnn_batch.to_string())
            .arg("--num_workers").arg(hp.mrcnn_workers.to_string())
            .arg("--output_dir").arg(&out_dir)
            .current_dir(project_root);

        if run_and_log(cmd, &log_path, "maskrcnn") {
            println!("[OK] Mask R-CNN training complete. Log: {}", log_path);
        } else {
            eprintln!("[ERROR] Mask R-CNN training failed. Log: {}", log_path);
        }
        all_metrics.insert("maskrcnn".to_string(), collect_metrics(&out_dir));
    }

    // ── Fast R-CNN ────────────────────────────────────────────────────────────
    if models.fastrcnn {
        let out_dir = format!("{}/fastrcnn", base_out_dir);
        fs::create_dir_all(&out_dir).unwrap();
        let log_path = format!("{}/05_train_fastrcnn.log", logs_dir);

        println!("\n========================================");
        println!(" Training: Fast R-CNN  ({} epochs)", hp.fastrcnn_epochs);
        println!("========================================");

        let mut cmd = Command::new("python");
        cmd.arg(format!("{}/scripts/training/train_fastrcnn.py", project_root))
            .arg("--dataset").arg(dataset)
            .arg("--epochs").arg(hp.fastrcnn_epochs.to_string())
            .arg("--batch").arg(hp.fastrcnn_batch.to_string())
            .arg("--num_workers").arg(hp.fastrcnn_workers.to_string())
            .arg("--project").arg(&out_dir)
            .current_dir(project_root);

        if run_and_log(cmd, &log_path, "fastrcnn") {
            println!("[OK] Fast R-CNN training complete. Log: {}", log_path);
        } else {
            eprintln!("[ERROR] Fast R-CNN training failed. Log: {}", log_path);
        }
        all_metrics.insert("fastrcnn".to_string(), collect_metrics(&out_dir));
    }

    all_metrics
}

// ─── Evaluation ───────────────────────────────────────────────────────────────

fn run_evaluation(dataset: &str, base_out_dir: &str, logs_dir: &str, project_root: &str) {
    println!("\n========================================");
    println!(" Running Evaluation");
    println!("========================================");

    let yolo_weights = fs::read_to_string(format!("{}/last_yolo_weights_path.txt", project_root))
        .unwrap_or_else(|_| format!("{}/yolo11m-seg/weights/best.pt", base_out_dir))
        .trim().to_string();

    let mrcnn_weights = format!("{}/maskrcnn/best_model.pt", base_out_dir);
    let fastrcnn_weights = format!("{}/fastrcnn/best_model.pt", base_out_dir);

    let eval_jobs: Vec<(&str, Vec<String>)> = vec![
        ("06a_eval_yolo",        vec!["--model".into(), "yolo".into(),        "--weights".into(), yolo_weights.clone(),  "--dataset".into(), dataset.to_string()]),
        ("06c_eval_maskrcnn",    vec!["--model".into(), "maskrcnn".into(),    "--weights".into(), mrcnn_weights.clone(), "--dataset".into(), dataset.to_string()]),
        ("06d_eval_fastrcnn",    vec!["--model".into(), "fastrcnn".into(),    "--weights".into(), fastrcnn_weights.clone(), "--dataset".into(), dataset.to_string()]),
    ];

    for (label, eval_args) in &eval_jobs {
        let log_path = format!("{}/{}.log", logs_dir, label);
        let mut cmd = Command::new("python");
        cmd.arg(format!("{}/scripts/evaluation/evaluate_confusion_matrix.py", project_root))
            .current_dir(project_root);
        for a in eval_args { cmd.arg(a); }

        println!("[*] {}...", label);
        if run_and_log(cmd, &log_path, label) {
            println!("[OK] {} done.", label);
        } else {
            eprintln!("[WARN] {} failed — check: {}", label, log_path);
        }
    }

    // Compare all models
    let log_path = format!("{}/07_compare_all.log", logs_dir);
    let mut cmd = Command::new("python");
    cmd.arg(format!("{}/scripts/evaluation/compare_all_models.py", project_root))
        .arg("--dataset").arg(dataset)
        .current_dir(project_root);
    println!("[*] Comparing all models...");
    if run_and_log(cmd, &log_path, "compare_all_models") {
        println!("[OK] Comparison done.");
    } else {
        eprintln!("[WARN] Model comparison failed — check: {}", log_path);
    }
}

// ─── Excel report ─────────────────────────────────────────────────────────────

fn generate_excel_report(base_out_dir: &str, all_metrics: &HashMap<String, Vec<MetricData>>) {
    let mut workbook = Workbook::new();
    let bold = Format::new().set_bold();

    // Sorted summary data: best mAP50 descending
    let mut summary: Vec<(String, u32, f64)> = all_metrics.iter().filter_map(|(model, metrics)| {
        metrics.iter().find(|m| m.is_best).or(metrics.last())
            .map(|b| (model.clone(), b.epoch, b.best_mask_map50))
    }).collect();
    summary.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    // Summary sheet
    {
        let ws = workbook.add_worksheet();
        ws.set_name("Summary").unwrap();
        ws.write_string_with_format(0, 0, "Model",           &bold).unwrap();
        ws.write_string_with_format(0, 1, "Best Epoch",      &bold).unwrap();
        ws.write_string_with_format(0, 2, "Best Mask mAP50", &bold).unwrap();
        for (i, (model, epoch, map50)) in summary.into_iter().enumerate() {
            let row = (i + 1) as u32;
            ws.write_string(row, 0, &model).unwrap();
            ws.write_number(row, 1, epoch).unwrap();
            ws.write_number(row, 2, map50).unwrap();
        }
    }

    // Per-model epoch detail sheets
    for (model, metrics) in all_metrics {
        let sheet_name = format!("{}_Epochs",
            model.replace("-", "_").chars().take(24).collect::<String>());
        let ws = workbook.add_worksheet();
        ws.set_name(&sheet_name).unwrap();
        ws.write_string_with_format(0, 0, "Epoch",      &bold).unwrap();
        ws.write_string_with_format(0, 1, "Train Loss", &bold).unwrap();
        ws.write_string_with_format(0, 2, "Val Loss",   &bold).unwrap();
        for (i, ep) in metrics.iter().enumerate() {
            let r = (i + 1) as u32;
            ws.write_number(r, 0, ep.epoch).unwrap();
            if let Some(v) = ep.train_stats.get("train_loss").and_then(|v| v.as_f64()) {
                ws.write_number(r, 1, v).unwrap();
            }
            if let Some(v) = ep.train_stats.get("val_loss").and_then(|v| v.as_f64()) {
                ws.write_number(r, 2, v).unwrap();
            }
        }
    }

    let path = format!("{}/car_part_training_results.xlsx", base_out_dir);
    workbook.save(&path).unwrap();
    println!("\n[OK] Excel report saved to {}", path);
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    check_venv();
    check_azure_disk();

    // Binary lives in orchestrator/target/release/ → project root is 3 levels up
    // But when run via `cargo run` from orchestrator/, cwd IS orchestrator/
    // so project root is simply ".."
    let project_root = Path::new("..")
        .canonicalize()
        .expect("Cannot resolve project root")
        .to_string_lossy()
        .to_string();

    let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();
    let base_out_dir = format!("{}/runs_comparison/run_{}", project_root, timestamp);
    let logs_dir     = format!("{}/logs", base_out_dir);
    fs::create_dir_all(&logs_dir).unwrap();

    {
        let mut f = File::create(format!("{}/orchestrator.log", logs_dir)).unwrap();
        writeln!(f, "Run started at {}", Local::now().format("%Y-%m-%d %H:%M:%S")).unwrap();
        writeln!(f, "Project root: {}", project_root).unwrap();
    }

    display_main_menu();
    let choice = prompt("Enter choice [1-9]: ");

    match choice.as_str() {
        // ── 1. Annotate ───────────────────────────────────────────────────────
        "1" => {
            println!("\n[TASK] Auto-annotation (DINO + SAM2)");
            let input_dir  = prompt_default("Input images directory",  "./RAW_DATASET/IMAGES");
            let output_dir = prompt_default("Output directory",        "./datasets/auto_annotated");
            let log_path   = format!("{}/01_annotation.log", logs_dir);
            let mut cmd = Command::new("python");
            cmd.arg(format!("{}/scripts/inference/auto_annotate_carparts.py", project_root))
                .arg("--input").arg(&input_dir)
                .arg("--output").arg(&output_dir)
                .current_dir(&project_root);
            if run_and_log(cmd, &log_path, "annotation") {
                println!("[OK] Annotation complete.");
            } else {
                eprintln!("[FAILED] Annotation failed. Log: {}", log_path);
            }
        }

        // ── 2. Train only ─────────────────────────────────────────────────────
        "2" => {
            println!("\n[TASK] Dataset Prep & Model Training");
            let models  = collect_models();
            let hparams = collect_hparams();
            let dataset = run_dataset_prep(&logs_dir, &project_root);
            println!("\n[INFO] Using dataset: {}", dataset);
            let all_metrics = run_training(&models, &hparams, &dataset, &base_out_dir, &logs_dir, &project_root);
            println!("\nAll training done. Generating Excel report...");
            generate_excel_report(&base_out_dir, &all_metrics);
        }

        // ── 3. Inference ──────────────────────────────────────────────────────
        "3" => {
            println!("\n[TASK] Inference on Test Images");
            let test_input  = prompt_default("Test images directory", "./test");
            let test_output = prompt_default("Output directory",      "./test_result");
            let log_path    = format!("{}/01_inference.log", logs_dir);
            let mut cmd = Command::new("python");
            cmd.arg(format!("{}/scripts/inference/infer_both_models.py", project_root))
                .arg("--input").arg(&test_input)
                .arg("--output").arg(&test_output)
                .current_dir(&project_root);
            if run_and_log(cmd, &log_path, "inference") {
                println!("[OK] Inference complete.");
            } else {
                eprintln!("[FAILED] Inference failed. Log: {}", log_path);
            }
        }

        // ── 4. Full pipeline ──────────────────────────────────────────────────
        "4" => {
            println!("\n[TASK] Full Pipeline (Dataset Prep → Train → Evaluate)");
            let models  = collect_models();
            let hparams = collect_hparams();
            let dataset = run_dataset_prep(&logs_dir, &project_root);
            println!("\n[INFO] Using dataset: {}", dataset);
            let all_metrics = run_training(&models, &hparams, &dataset, &base_out_dir, &logs_dir, &project_root);
            run_evaluation(&dataset, &base_out_dir, &logs_dir, &project_root);
            println!("\nAll steps done. Generating Excel report...");
            generate_excel_report(&base_out_dir, &all_metrics);
        }

        // ── 5. Azure ML Training ──────────────────────────────────────────────
        "5" => {
            println!("\n[TASK] Automated Azure ML Training");
            let models = collect_models();
            let hparams = collect_hparams();
            let local_dir = prompt_default("Local dataset directory", "./datasets");

            if models.yolo {
                println!("\n[*] Submitting Azure ML job for model: YOLOv11m-seg");
                let log_path = format!("{}/00_azure_train_yolo.log", logs_dir);
                let mut cmd = Command::new("python");
                cmd.arg(format!("{}/scripts/training/azure_train.py", project_root))
                    .arg("--model").arg("yolo11m-seg")
                    .arg("--local_dataset_dir").arg(&local_dir)
                    .arg("--epochs").arg(hparams.yolo_epochs.to_string())
                    .arg("--batch").arg(hparams.yolo_batch.to_string())
                    .arg("--workers").arg(hparams.yolo_workers.to_string())
                    .arg("--auto_upload")
                    .current_dir(&project_root);

                if run_and_log(cmd, &log_path, "azure_train_yolo") {
                    println!("[OK] Azure ML Job submitted for YOLOv11m-seg!");
                } else {
                    eprintln!("[FAILED] Azure ML submission failed for YOLOv11m-seg. Log: {}", log_path);
                }
            }

            if models.maskrcnn {
                println!("\n[*] Submitting Azure ML job for model: Mask R-CNN");
                let log_path = format!("{}/00_azure_train_maskrcnn.log", logs_dir);
                let mut cmd = Command::new("python");
                cmd.arg(format!("{}/scripts/training/azure_train.py", project_root))
                    .arg("--model").arg("maskrcnn")
                    .arg("--local_dataset_dir").arg(&local_dir)
                    .arg("--epochs").arg(hparams.mrcnn_epochs.to_string())
                    .arg("--batch").arg(hparams.mrcnn_batch.to_string())
                    .arg("--workers").arg(hparams.mrcnn_workers.to_string())
                    .arg("--auto_upload")
                    .current_dir(&project_root);

                if run_and_log(cmd, &log_path, "azure_train_maskrcnn") {
                    println!("[OK] Azure ML Job submitted for Mask R-CNN!");
                } else {
                    eprintln!("[FAILED] Azure ML submission failed for Mask R-CNN. Log: {}", log_path);
                }
            }

            if models.fastrcnn {
                println!("\n[*] Submitting Azure ML job for model: Fast R-CNN");
                let log_path = format!("{}/00_azure_train_fastrcnn.log", logs_dir);
                let mut cmd = Command::new("python");
                cmd.arg(format!("{}/scripts/training/azure_train.py", project_root))
                    .arg("--model").arg("fastrcnn")
                    .arg("--local_dataset_dir").arg(&local_dir)
                    .arg("--epochs").arg(hparams.fastrcnn_epochs.to_string())
                    .arg("--batch").arg(hparams.fastrcnn_batch.to_string())
                    .arg("--workers").arg(hparams.fastrcnn_workers.to_string())
                    .arg("--auto_upload")
                    .current_dir(&project_root);

                if run_and_log(cmd, &log_path, "azure_train_fastrcnn") {
                    println!("[OK] Azure ML Job submitted for Fast R-CNN!");
                } else {
                    eprintln!("[FAILED] Azure ML submission failed for Fast R-CNN. Log: {}", log_path);
                }
            }
        }
        
        // ── 6. Setup and Clean ────────────────────────────────────────────────
        "6" => {
            println!("\n[TASK] Setup and Clean");
            let log_path = format!("{}/00_setup_clean.log", logs_dir);
            let mut cmd = Command::new("python");
            cmd.arg(format!("{}/scripts/setup_and_clean.py", project_root))
                .current_dir(&project_root);
            if run_and_log(cmd, &log_path, "setup_clean") {
                println!("[OK] Setup and clean complete.");
            } else {
                eprintln!("[FAILED] Setup and clean failed.");
            }
        }

        // ── 7. Preprocess (Rust) ───────────────────────────────────────────────
        "7" => {
            println!("\n[TASK] Rust Preprocessing");
            println!("  a) yolo_to_coco    — Convert YOLO polygons → COCO JSON (parallel)");
            println!("  b) combine         — Merge multiple YOLO datasets (parallel copy)");
            println!("  c) verify_labels   — Scan for corrupt images (report-only)");
            println!("  d) hash_dataset    — Write blake3 checksums manifest");
            let sub = prompt("Enter sub-option [a-d]: ");
            let root_path = std::path::PathBuf::from(&project_root);
            match sub.as_str() {
                "a" => {
                    println!("  Available datasets: carparts-seg | custom_carparts | combined_carparts");
                    let ds = prompt_default("Dataset", "carparts-seg");
                    if let Err(e) = preprocess::run_yolo_to_coco(&ds, &root_path) {
                        eprintln!("[ERROR] {}", e);
                    }
                }
                "b" => {
                    let dirs_raw = prompt_default(
                        "Dataset dirs (comma-separated)",
                        "datasets/carparts-seg,datasets/custom_carparts",
                    );
                    let dirs: Vec<&str> = dirs_raw.split(',').map(str::trim).collect();
                    let out = prompt_default("Output dir", "datasets/combined_carparts");
                    if let Err(e) = preprocess::run_combine_datasets(&dirs, &out, &root_path) {
                        eprintln!("[ERROR] {}", e);
                    }
                }
                "c" => {
                    let dir_raw = prompt_default("Dataset directory to verify", "datasets");
                    let dir_path = root_path.join(&dir_raw);
                    match preprocess::run_verify_labels(&dir_path) {
                        Ok(report) => {
                            println!();
                            println!("  Total images : {}", report.total);
                            println!("  Valid        : {}", report.valid);
                            println!("  Corrupt      : {}", report.corrupt.len());
                            if !report.corrupt.is_empty() {
                                println!();
                                println!("  Corrupt files:");
                                for p in &report.corrupt {
                                    println!("    - {}", p.display());
                                }
                                println!();
                                println!("  No files were deleted. Remove them manually if needed.");
                            }
                        }
                        Err(e) => eprintln!("[ERROR] {}", e),
                    }
                }
                "d" => {
                    let dir_raw = prompt_default("Directory to hash", "datasets");
                    let dir_path = root_path.join(&dir_raw);
                    let manifest = dir_path.join("checksums.blake3");
                    if let Err(e) = preprocess::run_hash_dataset(&dir_path, &manifest) {
                        eprintln!("[ERROR] {}", e);
                    }
                }
                _ => eprintln!("[ERROR] Unknown sub-option '{}'", sub),
            }
            return; // Skip the TASK COMPLETE banner for sub-menu interactions
        }

        // ── 8. Benchmark ──────────────────────────────────────────────────────
        "8" => {
            println!("\n[TASK] Benchmark: Rust vs Python Preprocessing");
            println!("  The dataset must already exist locally before benchmarking.");
            let ds = prompt_default("Dataset to benchmark", "carparts-seg");
            let root_path = std::path::PathBuf::from(&project_root);
            bench::run_benchmark(&ds, &root_path);
            return;
        }

        // ── 9. Exit ────────────────────────────────────────────────────────────
        "9" => {
            println!("Exiting.");
            return;
        }

        _ => {
            eprintln!("[ERROR] Invalid option.");
            std::process::exit(1);
        }
    }

    println!();
    println!("==============================================");
    println!(" TASK COMPLETE");
    println!(" Outputs : {}", base_out_dir);
    println!(" Logs    : {}", logs_dir);
    println!("==============================================");
}
