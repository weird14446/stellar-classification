# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A stellar (astronomical object) classification project using a multilayer neural network, per the README (written in Korean: "multilayer neural network를 이용한 천체 분류 모델"). The repository currently contains only the dataset — no source code, dependency manifest, or build/test tooling exists yet. The `.gitignore` is a standard Python one, so new code is expected to be Python.

## Dataset: `star_classification.csv`

SDSS (Sloan Digital Sky Survey) stellar classification data: 100,000 rows, 17 feature columns plus a `class` target.
Source: [Stellar Classification Dataset - SDSS17 (fedesoriano, Kaggle)](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17) — cite this when referencing the dataset.

- Target `class` has three values: GALAXY (59,445), STAR (21,594), QSO (18,961) — imbalanced.
- Photometric features: `u`, `g`, `r`, `i`, `z` (filter magnitudes), plus `redshift`.
- Positional features: `alpha` (right ascension), `delta` (declination).
- Identifier/metadata columns that are not physically meaningful features: `obj_ID`, `run_ID`, `rerun_ID`, `cam_col`, `field_ID`, `spec_obj_ID`, `plate`, `MJD`, `fiber_ID`.

The CSV is ~16 MB and already committed; do not duplicate or transform it in place.

## Research Notes: `memory/`

The `memory/` directory holds the user's research notes for this project. Notes are named by date: `memory/YYYY-MM-DD.md` (e.g. `memory/2026-07-07.md`). Consult them for context on past experiments and decisions, and record new research notes in this format when asked.

## Figures: `figures/`

Save all generated figures (plots, charts, visualizations) as image files in the `figures/` directory.

## Environment Note

The working directory lives inside Google Drive (`CloudStorage/GoogleDrive-...`), so file paths contain spaces and an `@` — always quote paths in shell commands.
