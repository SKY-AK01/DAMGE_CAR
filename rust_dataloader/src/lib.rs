/*!
rust_dataloader/src/lib.rs
==========================
PyO3-powered Rust DataLoader for Car-Parts Instance Segmentation.

Exposes a Python-callable `CarPartsDataset` class that:
  1. Reads a COCO-format JSON annotation file
  2. Decodes JPEG/PNG images in parallel using Rayon
  3. Rasterizes polygon segmentation masks using imageproc
  4. Applies optional augmentations (H-flip, brightness jitter)
  5. Returns (image_bytes, mask_bytes, labels, image_id, n_instances)
     as raw f32 bytes which the Python bridge converts to torch.Tensor.
*/

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ─────────────────────────────────────────────────────────────────────────────
// COCO JSON structures
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
struct CocoImage {
    id: u64,
    file_name: String,
    width: u32,
    height: u32,
}

#[derive(Debug, Clone, Deserialize)]
struct CocoAnnotation {
    image_id: u64,
    category_id: u32,
    segmentation: serde_json::Value,
}

#[derive(Debug, Clone, Deserialize)]
struct CocoCategory {
    id: u32,
    name: String,
}

#[derive(Debug, Clone, Deserialize)]
struct CocoData {
    images: Vec<CocoImage>,
    annotations: Vec<CocoAnnotation>,
    categories: Vec<CocoCategory>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Augmentation helpers (rand 0.9 API)
// ─────────────────────────────────────────────────────────────────────────────

fn hflip_rgb(buf: &mut [u8], width: u32, height: u32) {
    let w = width as usize;
    let h = height as usize;
    let channels = buf.len() / h / w;
    for row in buf.chunks_mut(w * channels) {
        for col in 0..(w / 2) {
            let l = col * channels;
            let r = (w - 1 - col) * channels;
            for c in 0..channels {
                row.swap(l + c, r + c);
            }
        }
    }
}

fn brightness_jitter(buf: &mut [u8], factor: f32, seed: u64) {
    use rand::prelude::*;
    use rand::rngs::SmallRng;
    let mut rng = SmallRng::seed_from_u64(seed);
    let scale: f32 = 1.0 + rng.random_range(-factor..factor);
    for v in buf.iter_mut() {
        *v = ((*v as f32 * scale).clamp(0.0, 255.0)) as u8;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Core: one sample → (img_f32 CHW, masks_f32 NHW, label_ids)
// ─────────────────────────────────────────────────────────────────────────────

fn process_sample(
    image_path: &Path,
    anns: &[CocoAnnotation],
    img_size: u32,
    do_augment: bool,
    aug_seed: u64,
) -> Option<(Vec<f32>, Vec<f32>, Vec<u32>)> {
    use rand::prelude::*;
    use rand::rngs::SmallRng;

    // ── Load & resize ─────────────────────────────────────────────────────────
    let dyn_img = image::open(image_path).ok()?;
    let orig_w = dyn_img.width() as f32;
    let orig_h = dyn_img.height() as f32;
    let resized = dyn_img.resize_exact(img_size, img_size, image::imageops::FilterType::Lanczos3);
    let mut rgb = resized.to_rgb8();

    // ── Augmentations ─────────────────────────────────────────────────────────
    let mut do_flip = false;
    if do_augment {
        let mut rng = SmallRng::seed_from_u64(aug_seed);
        do_flip = rng.random_bool(0.5);
        if do_flip {
            hflip_rgb(rgb.as_mut(), img_size, img_size);
        }
        brightness_jitter(rgb.as_mut(), 0.2, aug_seed.wrapping_add(1));
    }

    // ── Normalise to CHW f32 (ImageNet stats) ────────────────────────────────
    let hw = (img_size * img_size) as usize;
    let mut img_f32 = vec![0f32; 3 * hw];
    let raw = rgb.as_raw();
    let mean = [0.485f32, 0.456, 0.406];
    let std  = [0.229f32, 0.224, 0.225];
    for i in 0..hw {
        for c in 0..3usize {
            let v = raw[i * 3 + c] as f32 / 255.0;
            img_f32[c * hw + i] = (v - mean[c]) / std[c];
        }
    }

    // ── Rasterise polygon masks ───────────────────────────────────────────────
    let scale_x = img_size as f32 / orig_w;
    let scale_y = img_size as f32 / orig_h;
    let mut masks: Vec<Vec<f32>> = Vec::new();
    let mut labels: Vec<u32> = Vec::new();

    for ann in anns {
        let polys: Vec<Vec<f64>> = match &ann.segmentation {
            serde_json::Value::Array(segs) => segs
                .iter()
                .filter_map(|s| {
                    s.as_array().map(|pts| {
                        pts.iter().filter_map(|v| v.as_f64()).collect::<Vec<_>>()
                    })
                })
                .collect(),
            _ => continue,
        };
        if polys.is_empty() { continue; }

        let mut mask_buf = vec![0f32; hw];

        for poly in &polys {
            if poly.len() < 6 { continue; }
            let mut pts: Vec<imageproc::point::Point<i32>> = poly
                .chunks(2)
                .filter_map(|c| {
                    if c.len() == 2 {
                        let x = (c[0] as f32 * scale_x).round() as i32;
                        let y = (c[1] as f32 * scale_y).round() as i32;
                        Some(imageproc::point::Point::new(x, y))
                    } else {
                        None
                    }
                })
                .collect();
            pts.dedup();
            while pts.len() >= 2 && pts.first() == pts.last() {
                pts.pop();
            }
            if pts.len() < 3 { continue; }

            let mut mono = image::GrayImage::new(img_size, img_size);
            imageproc::drawing::draw_polygon_mut(&mut mono, &pts, image::Luma([255u8]));
            if do_flip {
                image::imageops::flip_horizontal_in_place(&mut mono);
            }
            for (px, v) in mask_buf.iter_mut().zip(mono.as_raw().iter()) {
                if *v > 0 { *px = 1.0f32; }
            }
        }

        masks.push(mask_buf);
        labels.push(ann.category_id);
    }

    let masks_flat: Vec<f32> = masks.into_iter().flatten().collect();
    Some((img_f32, masks_flat, labels))
}

// ─────────────────────────────────────────────────────────────────────────────
// PyO3 class
// ─────────────────────────────────────────────────────────────────────────────

#[pyclass]
pub struct CarPartsDataset {
    image_ids:     Vec<u64>,
    images_by_id:  HashMap<u64, CocoImage>,
    anns_by_image: HashMap<u64, Vec<CocoAnnotation>>,
    images_dir:    PathBuf,
    img_size:      u32,
    augment:       bool,
    num_classes:   usize,
    class_names:   Vec<String>,
}

#[pymethods]
impl CarPartsDataset {
    #[new]
    #[pyo3(signature = (json_path, images_dir, img_size=640, augment=false))]
    pub fn new(json_path: &str, images_dir: &str, img_size: u32, augment: bool) -> PyResult<Self> {
        let content = std::fs::read_to_string(json_path).map_err(|e| {
            pyo3::exceptions::PyFileNotFoundError::new_err(format!("Cannot open '{}': {}", json_path, e))
        })?;
        let coco: CocoData = serde_json::from_str(&content).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Bad COCO JSON '{}': {}", json_path, e))
        })?;

        let images_by_id: HashMap<u64, CocoImage> =
            coco.images.into_iter().map(|img| (img.id, img)).collect();
        let mut anns_by_image: HashMap<u64, Vec<CocoAnnotation>> = HashMap::new();
        for ann in coco.annotations {
            anns_by_image.entry(ann.image_id).or_default().push(ann);
        }
        let image_ids: Vec<u64> = images_by_id.keys().cloned().collect();
        let num_classes = coco.categories.len();
        let class_names: Vec<String> = coco.categories.into_iter().map(|c| c.name).collect();

        Ok(Self { image_ids, images_by_id, anns_by_image, images_dir: PathBuf::from(images_dir),
                  img_size, augment, num_classes, class_names })
    }

    pub fn __len__(&self) -> usize { self.image_ids.len() }
    pub fn num_classes(&self) -> usize { self.num_classes }
    pub fn class_names(&self) -> Vec<String> { self.class_names.clone() }

    /// Fetch one sample → (img_bytes, mask_bytes, labels, image_id, n_instances)
    pub fn get_item<'py>(
        &self, py: Python<'py>, idx: usize,
    ) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>, Vec<u32>, u64, usize)> {
        let img_id   = self.image_ids[idx];
        let img_info = &self.images_by_id[&img_id];
        let img_path = self.images_dir.join(&img_info.file_name);
        let empty: Vec<CocoAnnotation> = Vec::new();
        let anns = self.anns_by_image.get(&img_id).unwrap_or(&empty);

        match process_sample(&img_path, anns, self.img_size, self.augment, idx as u64) {
            Some((img_f32, masks_f32, labels)) => {
                let n = labels.len();
                Ok((
                    PyBytes::new(py, f32_as_u8(&img_f32)),
                    PyBytes::new(py, f32_as_u8(&masks_f32)),
                    labels,
                    img_id,
                    n,
                ))
            }
            None => Err(pyo3::exceptions::PyRuntimeError::new_err(
                format!("Failed to load: {:?}", img_path))),
        }
    }

    /// Parallel batch fetch using Rayon (releases GIL during decode).
    pub fn get_batch<'py>(&self, py: Python<'py>, indices: Vec<usize>) -> PyResult<Bound<'py, PyList>> {
        // Collect data in parallel outside the GIL
        let results: Vec<_> = py.allow_threads(|| {
            indices.par_iter().map(|&idx| {
                let img_id   = self.image_ids[idx];
                let img_info = &self.images_by_id[&img_id];
                let img_path = self.images_dir.join(&img_info.file_name);
                let empty: Vec<CocoAnnotation> = Vec::new();
                let anns = self.anns_by_image.get(&img_id).unwrap_or(&empty);
                process_sample(&img_path, anns, self.img_size, self.augment, idx as u64)
                    .map(|(img, masks, labels)| (img, masks, labels, img_id))
            }).collect()
        });

        let list = PyList::empty(py);
        for res in results {
            match res {
                Some((img_f32, masks_f32, labels, img_id)) => {
                    let n = labels.len();
                    let img_bytes  = PyBytes::new(py, f32_as_u8(&img_f32));
                    let mask_bytes = PyBytes::new(py, f32_as_u8(&masks_f32));
                    list.append((img_bytes, mask_bytes, labels, img_id, n))?;
                }
                None => list.append(py.None())?,
            }
        }
        Ok(list)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn f32_as_u8(data: &[f32]) -> &[u8] {
    // SAFETY: f32 is 4 bytes, no padding, valid for any bit pattern.
    unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * 4,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Module
// ─────────────────────────────────────────────────────────────────────────────

#[pymodule]
fn rust_dataloader(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<CarPartsDataset>()?;
    Ok(())
}
