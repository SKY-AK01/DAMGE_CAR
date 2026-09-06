//! preprocess.rs — Rust implementations of CPU-heavy dataset preprocessing steps.
//!
//! Replaces (and stays parallel to) the Python scripts:
//!   yolo_to_coco.py                   → run_yolo_to_coco()
//!   combine_datasets.py               → run_combine_datasets()
//!   sanitize_and_validate_dataset.py  → run_verify_labels()
//!   (no Python equivalent)            → run_hash_dataset() (blake3)
//!
//! Design principles
//! -----------------
//! • All Python scripts are kept and remain functional.
//! • Rust implementations are ADDITIVE — the orchestrator calls Rust by default
//!   and can fall back to Python.
//! • verify_labels is REPORT-ONLY. It never deletes or modifies user data.
//! • No performance claims are made here. See bench.rs for measurements.

use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::fs;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

// ─── Shared helpers ───────────────────────────────────────────────────────────

fn make_bar(total: u64, color: &str) -> ProgressBar {
    let pb = ProgressBar::new(total);
    let template = format!(
        "  {{bar:40.{}}} {{pos}}/{{len}}  [{{elapsed_precise}}]",
        color
    );
    pb.set_style(
        ProgressStyle::default_bar()
            .template(&template)
            .unwrap_or_else(|_| ProgressStyle::default_bar()),
    );
    pb
}

// ─── YOLO → COCO conversion ──────────────────────────────────────────────────

/// The 23-class taxonomy used by carparts-seg and custom_carparts.
/// Matches CARPARTS_SEG_CLASSES in yolo_to_coco.py exactly (same order, same names).
pub const CARPARTS_SEG_CLASSES: &[&str] = &[
    "back_bumper",      "back_door",        "back_glass",       "back_left_door",
    "back_left_light",  "back_light",       "back_right_door",  "back_right_light",
    "front_bumper",     "front_door",       "front_glass",      "front_left_door",
    "front_left_light", "front_light",      "front_right_door", "front_right_light",
    "hood",             "left_mirror",      "object",           "right_mirror",
    "tailgate",         "trunk",            "wheel",
];

/// Result of parsing one YOLO polygon annotation line.
struct PolygonResult {
    class_id: usize,
    /// Flat pixel-space coords [x0, y0, x1, y1, ...] — COCO segmentation format.
    points: Vec<f64>,
    /// COCO bbox: [x_min, y_min, width, height]
    bbox: [f64; 4],
    /// Polygon area computed via the Shoelace formula.
    area: f64,
}

/// Convert one YOLO-seg polygon text line to COCO annotation fields.
///
/// Matches Python's yolo_polygon_to_coco() in yolo_to_coco.py exactly,
/// including the Shoelace formula orientation.
///
/// Returns None for empty, malformed, or too-short lines.
fn yolo_polygon_to_coco(line: &str, img_w: f64, img_h: f64) -> Option<PolygonResult> {
    let parts: Vec<&str> = line.trim().split_ascii_whitespace().collect();
    // Minimum: class_id + 3 (x,y) pairs = 7 tokens
    if parts.len() < 7 {
        return None;
    }
    let class_id: usize = parts[0].parse().ok()?;
    let coords: Vec<f64> = parts[1..]
        .iter()
        .filter_map(|s| s.parse::<f64>().ok())
        .collect();
    if coords.len() < 6 || coords.len() % 2 != 0 {
        return None;
    }

    // Denormalise: multiply normalised coords by image dimensions.
    let mut points = Vec::with_capacity(coords.len());
    for chunk in coords.chunks_exact(2) {
        points.push(chunk[0] * img_w); // x
        points.push(chunk[1] * img_h); // y
    }

    let xs: Vec<f64> = points.iter().step_by(2).copied().collect();
    let ys: Vec<f64> = points.iter().skip(1).step_by(2).copied().collect();

    let x_min = xs.iter().cloned().fold(f64::INFINITY,     f64::min);
    let y_min = ys.iter().cloned().fold(f64::INFINITY,     f64::min);
    let x_max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_max = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let n = xs.len();
    // Shoelace formula — matches Python exactly:
    //   0.5 * abs(sum(xs[i] * ys[i-1] - xs[i-1] * ys[i] for i in range(n)))
    // Python's ys[i-1] at i=0 is ys[-1] (last element) → prev = (i + n - 1) % n
    let area = 0.5
        * (0..n)
            .map(|i| {
                let prev = if i == 0 { n - 1 } else { i - 1 };
                xs[i] * ys[prev] - xs[prev] * ys[i]
            })
            .sum::<f64>()
            .abs();

    Some(PolygonResult {
        class_id,
        points,
        bbox: [x_min, y_min, x_max - x_min, y_max - y_min],
        area,
    })
}

/// Per-image result produced in parallel; assembled into COCO JSON sequentially.
struct ImageEntry {
    img_id: usize,
    file_name: String,
    width: u32,
    height: u32,
    polygons: Vec<PolygonResult>,
}

/// Convert one dataset split directory (train / val / test) to a COCO JSON file.
///
/// Image dimensions are read in parallel using rayon + `image::ImageReader`,
/// which reads only the image header (JPEG SOF / PNG IHDR) without decoding pixels.
/// This is the main speedup over Python's sequential PIL.Image.open().
///
/// Output JSON is identical in structure to Python's convert_split().
fn convert_split(
    images_dir: &Path,
    labels_dir: &Path,
    class_names: &[&str],
    out_json_path: &Path,
) -> Result<(usize, usize), String> {
    // Collect and sort image files — matches Python's sorted(images_dir.glob("*"))
    let image_exts = ["jpg", "jpeg", "png"];
    let mut image_files: Vec<PathBuf> = fs::read_dir(images_dir)
        .map_err(|e| format!("Cannot read {}: {}", images_dir.display(), e))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| image_exts.contains(&ext.to_lowercase().as_str()))
                .unwrap_or(false)
        })
        .collect();
    image_files.sort(); // Deterministic order, matches Python

    let pb = make_bar(image_files.len() as u64, "green");

    // ── Parallel pass: read image header + parse label file ──
    let mut entries: Vec<ImageEntry> = image_files
        .par_iter()
        .enumerate()
        .filter_map(|(img_id, img_path)| {
            // Read image dimensions from header only (no pixel decode).
            let (width, height) = image::ImageReader::open(img_path)
                .ok()?
                .with_guessed_format()
                .ok()?
                .into_dimensions()
                .ok()?;

            let file_name = img_path.file_name()?.to_string_lossy().into_owned();
            let stem      = img_path.file_stem()?.to_string_lossy().into_owned();
            let label_path = labels_dir.join(format!("{}.txt", stem));

            let polygons: Vec<PolygonResult> = if label_path.exists() {
                let Ok(file) = fs::File::open(&label_path) else { return None; };
                BufReader::new(file)
                    .lines()
                    .filter_map(|l| l.ok())
                    .filter(|l| !l.trim().is_empty())
                    .filter_map(|l| yolo_polygon_to_coco(&l, width as f64, height as f64))
                    .collect()
            } else {
                vec![]
            };

            pb.inc(1);
            Some(ImageEntry { img_id, file_name, width, height, polygons })
        })
        .collect();

    pb.finish_and_clear();

    // Restore deterministic order — rayon may reorder results.
    entries.sort_by_key(|e| e.img_id);

    // ── Sequential pass: assign annotation IDs, assemble COCO JSON ──
    let categories: Vec<Value> = class_names
        .iter()
        .enumerate()
        .map(|(i, name)| json!({"id": i, "name": name}))
        .collect();

    let mut images_json  = Vec::with_capacity(entries.len());
    let mut annotations_json: Vec<Value> = Vec::new();
    let mut ann_id = 0usize;

    for entry in &entries {
        images_json.push(json!({
            "id":        entry.img_id,
            "file_name": entry.file_name,
            "width":     entry.width,
            "height":    entry.height,
        }));
        for poly in &entry.polygons {
            annotations_json.push(json!({
                "id":          ann_id,
                "image_id":    entry.img_id,
                "category_id": poly.class_id,
                "segmentation": [poly.points],
                "bbox":         poly.bbox,
                "area":         poly.area,
                "iscrowd":      0,
            }));
            ann_id += 1;
        }
    }

    let n_images = images_json.len();
    let n_ann    = annotations_json.len();

    let coco = json!({
        "images":      images_json,
        "annotations": annotations_json,
        "categories":  categories,
    });

    fs::write(
        out_json_path,
        serde_json::to_string(&coco).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("Cannot write {}: {}", out_json_path.display(), e))?;

    Ok((n_images, n_ann))
}

/// Run YOLO→COCO conversion for a named dataset.
///
/// Matches the routing logic in yolo_to_coco.py main().
/// Supported datasets: carparts-seg | custom_carparts | combined_carparts | dsmlr-carparts
pub fn run_yolo_to_coco(dataset: &str, project_root: &Path) -> Result<(), String> {
    match dataset {
        "carparts-seg" | "custom_carparts" | "combined_carparts" => {
            let root = project_root.join("datasets").join(dataset);
            for split in ["train", "val", "test"] {
                let images_dir = root.join("images").join(split);
                let labels_dir = root.join("labels").join(split);
                if !images_dir.exists() {
                    println!("  [skip] {} not found", images_dir.display());
                    continue;
                }
                let out_path = root.join(format!("coco_{}.json", split));
                println!("  Converting {} / {} ...", dataset, split);
                let (n_img, n_ann) =
                    convert_split(&images_dir, &labels_dir, CARPARTS_SEG_CLASSES, &out_path)?;
                println!(
                    "  [OK] {} — {} images, {} annotations",
                    out_path.display(),
                    n_img,
                    n_ann
                );
            }
        }
        "dsmlr-carparts" => {
            println!(
                "[i] dsmlr-carparts is already prepared as COCO JSON by prepare_dsmlr_split.py — skipping."
            );
        }
        _ => {
            return Err(format!(
                "Unknown dataset '{}'. Choices: carparts-seg | custom_carparts | combined_carparts | dsmlr-carparts",
                dataset
            ));
        }
    }
    Ok(())
}

// ─── Combine datasets ─────────────────────────────────────────────────────────

/// Run a parallel copy of all files from multiple YOLO dataset directories
/// into one combined dataset directory.
///
/// Preserves the collision-avoidance prefix from Python:
///   destination filename = "{dataset_dir_name}_{original_filename}"
///
/// This matches combine_datasets.py: combine_yolo_datasets().
pub fn run_combine_datasets(
    dataset_dirs: &[&str],
    out_dir: &str,
    project_root: &Path,
) -> Result<(), String> {
    let out_path = project_root.join(out_dir);

    for split in ["train", "val"] {
        for kind in ["images", "labels"] {
            fs::create_dir_all(out_path.join(kind).join(split))
                .map_err(|e| format!("Cannot create output dir: {}", e))?;
        }
    }

    // Collect all file-copy jobs and the first YAML found
    let mut copy_jobs: Vec<(PathBuf, PathBuf)> = Vec::new();
    let mut yaml_names_block: Option<String> = None;

    for ddir in dataset_dirs {
        let dp = project_root.join(ddir);
        if !dp.exists() {
            println!("  [skip] {} not found", dp.display());
            continue;
        }
        // Use the dataset directory's own name as the file prefix
        let dp_name = dp
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();

        println!("  [*] Queuing {}...", ddir);

        for split in ["train", "val"] {
            let img_dir = dp.join("images").join(split);
            let lbl_dir = dp.join("labels").join(split);

            // Queue image copies
            if img_dir.exists() {
                for entry in fs::read_dir(&img_dir).map_err(|e| e.to_string())?.flatten() {
                    let src = entry.path();
                    if !src.is_file() { continue; }
                    let fname = src.file_name().unwrap().to_string_lossy();
                    let dst_name = format!("{}_{}", dp_name, fname);
                    copy_jobs.push((src, out_path.join("images").join(split).join(dst_name)));
                }
            }

            // Queue label copies
            if lbl_dir.exists() {
                for entry in fs::read_dir(&lbl_dir).map_err(|e| e.to_string())?.flatten() {
                    let src = entry.path();
                    if !src.is_file() { continue; }
                    if src.extension().and_then(|e| e.to_str()) != Some("txt") { continue; }
                    let fname = src.file_name().unwrap().to_string_lossy();
                    let dst_name = format!("{}_{}", dp_name, fname);
                    copy_jobs.push((src, out_path.join("labels").join(split).join(dst_name)));
                }
            }
        }

        // Inherit names block from first dataset that has a YAML
        if yaml_names_block.is_none() {
            let yaml_path = fs::read_dir(&dp)
                .ok()
                .and_then(|d| {
                    d.flatten()
                        .find(|e| {
                            e.path()
                                .extension()
                                .and_then(|ex| ex.to_str())
                                == Some("yaml")
                        })
                        .map(|e| e.path())
                });
            if let Some(yp) = yaml_path {
                if let Ok(content) = fs::read_to_string(&yp) {
                    // Extract everything from the 'names:' line onwards
                    let block: Vec<&str> = content
                        .lines()
                        .skip_while(|l| !l.starts_with("names:"))
                        .collect();
                    if !block.is_empty() {
                        yaml_names_block = Some(block.join("\n"));
                    }
                }
            }
        }
    }

    // ── Parallel copy ──
    println!("  Copying {} files in parallel...", copy_jobs.len());
    let pb = make_bar(copy_jobs.len() as u64, "cyan");

    let errors: Vec<String> = copy_jobs
        .par_iter()
        .filter_map(|(src, dst)| {
            let r = fs::copy(src, dst);
            pb.inc(1);
            r.err()
                .map(|e| format!("{} -> {}: {}", src.display(), dst.display(), e))
        })
        .collect();

    pb.finish_and_clear();

    if !errors.is_empty() {
        eprintln!("[WARN] {} copy error(s):", errors.len());
        for e in &errors[..errors.len().min(10)] {
            eprintln!("  {}", e);
        }
    }

    // Write combined data.yaml
    let mut yaml = format!(
        "path: {}\ntrain: images/train\nval: images/val\ntest:\n\n",
        out_path.display()
    );
    if let Some(names) = yaml_names_block {
        yaml.push_str(&names);
        yaml.push('\n');
    }
    fs::write(out_path.join("data.yaml"), yaml)
        .map_err(|e| format!("Cannot write data.yaml: {}", e))?;

    println!("  [OK] Datasets combined into {}", out_dir);
    Ok(())
}

// ─── Verify labels (report-only) ─────────────────────────────────────────────

/// Summary produced by run_verify_labels().
pub struct VerifyReport {
    pub total: usize,
    pub valid: usize,
    /// Sorted list of paths whose headers could not be read.
    /// Nothing is deleted. The caller decides what to do.
    pub corrupt: Vec<PathBuf>,
}

/// Scan a dataset directory for unreadable or corrupt image files.
///
/// Uses parallel header-only reads (same `image::ImageReader::into_dimensions()`
/// trick as convert_split) for speed on large datasets.
///
/// REPORT-ONLY. Does NOT delete, move, or modify any files.
/// If deletion is needed, add `--delete-corrupt` as an explicit opt-in later.
pub fn run_verify_labels(dataset_dir: &Path) -> Result<VerifyReport, String> {
    let image_exts = ["jpg", "jpeg", "png", "bmp", "webp"];

    let image_files: Vec<PathBuf> = WalkDir::new(dataset_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .filter(|p| {
            p.extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| image_exts.contains(&ext.to_lowercase().as_str()))
                .unwrap_or(false)
        })
        .collect();

    let total = image_files.len();
    let pb    = make_bar(total as u64, "yellow");

    let mut corrupt: Vec<PathBuf> = image_files
        .par_iter()
        .filter_map(|path| {
            let ok = image::ImageReader::open(path)
                .ok()
                .and_then(|r| r.with_guessed_format().ok())
                .and_then(|r| r.into_dimensions().ok())
                .is_some();
            pb.inc(1);
            if ok { None } else { Some(path.clone()) }
        })
        .collect();

    pb.finish_and_clear();
    corrupt.sort(); // Deterministic output order

    let valid = total - corrupt.len();
    Ok(VerifyReport { total, valid, corrupt })
}

// ─── Dataset integrity hashing (blake3) ──────────────────────────────────────

/// Compute blake3 hashes for all files in a directory and write a manifest.
///
/// Output format (one line per file, sorted by path for reproducibility):
///   <64-hex-char-hash>  <relative/path/to/file>
///
/// Useful for verifying dataset integrity before/after Azure upload.
pub fn run_hash_dataset(dataset_dir: &Path, out_manifest: &Path) -> Result<(), String> {
    let mut all_files: Vec<PathBuf> = WalkDir::new(dataset_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .collect();
    all_files.sort();

    let pb = make_bar(all_files.len() as u64, "magenta");

    let mut hash_lines: Vec<(PathBuf, String)> = all_files
        .par_iter()
        .filter_map(|path| {
            let h = hash_file(path);
            pb.inc(1);
            match h {
                Ok(hex) => Some((path.clone(), hex)),
                Err(e) => {
                    eprintln!("[WARN] Cannot hash {}: {}", path.display(), e);
                    None
                }
            }
        })
        .collect();

    pb.finish_and_clear();
    hash_lines.sort_by(|a, b| a.0.cmp(&b.0)); // deterministic manifest order

    let manifest: String = hash_lines
        .iter()
        .map(|(path, hex)| {
            let rel = path
                .strip_prefix(dataset_dir)
                .unwrap_or(path)
                .to_string_lossy();
            format!("{}  {}", hex, rel)
        })
        .collect::<Vec<_>>()
        .join("\n");

    fs::write(out_manifest, format!("{}\n", manifest))
        .map_err(|e| format!("Cannot write manifest {}: {}", out_manifest.display(), e))?;

    println!(
        "  [OK] Hashed {} files → {}",
        hash_lines.len(),
        out_manifest.display()
    );
    Ok(())
}

fn hash_file(path: &Path) -> Result<String, String> {
    let mut file  = fs::File::open(path).map_err(|e| e.to_string())?;
    let mut hasher = blake3::Hasher::new();
    let mut buf   = vec![0u8; 65536]; // 64 KB read buffer
    loop {
        let n = file.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 { break; }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}
