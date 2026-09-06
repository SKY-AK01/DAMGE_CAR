# AI Coding Agent Instructions — Pipeline Migration

Read this file fully before writing or editing any code. Also read `info.md`,
`run_analysis.md`, and `pipeline_migration_plan.md` before starting — those define
the current architecture, the known bug, and the full scope of work. Do not begin
coding from memory of a prior summary; open and read the actual files.

---

## Ground Rules (apply to every task below)

1. **Read before you edit.** Never assume what an argparser, function signature, or
   config file contains — open it and check. The known bug in this repo
   (`--output_dir` vs `--project`) exists precisely because that wasn't done once.
2. **One checklist item at a time**, in order. Don't jump ahead or batch multiple
   items into one change — each item gets its own edit + verification + report.
3. **Re-read a file after editing it** before making a second edit to it — don't
   chain edits from a stale in-memory view of the file.
4. **Don't touch what isn't in scope.** Only edit code related to the current
   checklist item. Leave YOLO / Mask R-CNN / Fast R-CNN working paths alone unless
   the task explicitly says to change them.
5. **Never fabricate a missing piece silently.** If a script, function, or config
   value doesn't exist yet (e.g. no Fast R-CNN training script currently exists),
   say so explicitly and flag it — don't invent plausible-looking logic and present
   it as if it already matched an existing convention.
6. **Verify before declaring done:**
   - Rust changes → run `cargo build` (or `cargo check`), paste the result.
   - Python changes → run a syntax check (`python -m py_compile`) at minimum; run
     the script's `--help` if it's a CLI entry point, to confirm args parse.
   - Shell script changes → run `bash -n script.sh` to catch syntax errors.
7. **No secrets in code.** Any credential, token, subscription ID, or key goes in
   `.env` / config, never hardcoded in a script or committed file.
8. **When ambiguous, stop and ask** — don't guess a file path, folder convention,
   or naming scheme. A wrong guess here (e.g. where raw data lives) breaks the
   thing this whole plan exists to prevent.

---

## Task Checklist (work top to bottom)

1. **Fix** the `--output_dir` → `--project` mismatch in `orchestrator/src/main.rs`.
2. **Audit** every `train_*.py` argparser against its `Command::new("python")` call
   in `main.rs` — report any other mismatches found before fixing them.
3. **Clean**: remove Mask2Former / OneFormer / MaskDINO code, `ModelSelection`
   enum entries, `eval_jobs` entries, shell `case` blocks, env flags
   (`ENABLE_MASKDINO`, `ENABLE_ONEFORMER`), and now-unused dependencies.
4. **Clean**: update `README.md`, `PROJECT_LOG.md`, `info.md` to reflect the
   3-model pipeline (YOLO / Mask R-CNN / Fast R-CNN only).
5. **Add**: Fast R-CNN training script, matching the existing arg convention
   (`--dataset --epochs --batch --project`).
6. **Add**: Azure ML integration — `upload_dataset.py`, `azure_train.py`,
   `.env` template with the required Azure fields.
7. **Add**: Rust-based data transfer layer (multithreaded upload, integrity
   check, timestamped destination folders, junk-dir exclusion).
8. **Add**: `setup_and_clean` script — folder setup, raw-data location guidance,
   archive-on-clean, open-source vs user-data presence check, combined-count
   check, post-merge label conversion check.
9. **Add**: argument-contract safety net (shared schema or `--help`-diff test)
   so the Task 1/2 bug class can't silently reoccur.
10. **Add** (not blocking — pick up once Tasks 1-9 are stable): port CPU/IO-bound
    pipeline steps (label conversion, dataset merge, augmentation, integrity
    hashing) to Rust for speed. Full scope in `pipeline_migration_plan.md`.

---

## Report Back — after EVERY task

Don't move to the next task until you've given this for the current one:

- **Changed:** exact file(s) + line ranges touched.
- **Verified:** the actual command you ran to check it, and its output/result.
- **Incomplete / needs a decision:** anything left undone, or any choice that
  needs the user's input (e.g. "which folder convention did you mean?").
- **Assumptions made:** state them explicitly — don't let one slide in unstated.

---

## Stop Conditions

- Referenced file/folder doesn't exist where expected → stop, ask, don't guess.
- Two plausible ways to implement something with different tradeoffs → stop, ask.
- A "clean" step would delete something not clearly superseded by a "keep" model →
  stop, ask before deleting.
