# Configuring the 3D binding-pose viewer

The AlchemFold **Docking & virtual screening** page renders the predicted complex
(receptor cartoon + ligand sticks) in an embedded 3D viewer, next to the Boltz-2
affinity / confidence scores. The viewer uses **3Dmol.js** running inside the
desktop's Chromium (QtWebEngine), which needs two things:

1. the **3Dmol.js** library available locally (no internet needed at view time), and
2. **WebGL** support in the embedded browser.

On a laptop/workstation both usually work out of the box. On an HPC login/compute
node (headless, often offline) you normally need the one-time setup below.

---

## 1. Vendor the 3Dmol.js library (offline)

The viewer looks for `3Dmol-min.js` in this order: `CHEMFLOW_3DMOL_PATH` →
`~/.cache/chemflow/3dmol/3Dmol-min.js` → `<repo>/.chemflow/vendor/3dmol/3Dmol-min.js`.
If none exists and there's no internet, the 3D panel shows
*"3Dmol.js not found. Run `chemflow vendor-3dmol` …"* instead of a structure.

Fetch it once on a node that **has internet** (on Biowulf, a login or
data-transfer node):

```bash
conda run -n chemflow chemflow vendor-3dmol
# writes ~/.cache/chemflow/3dmol/3Dmol-min.js
```

Options:

- `chemflow vendor-3dmol --project` — vendor inside the repo
  (`<repo>/.chemflow/vendor/3dmol/3Dmol-min.js`) instead of the user cache.
- `chemflow vendor-3dmol --dest /path/3Dmol-min.js` — explicit path.
- Manual: download `https://3Dmol.org/build/3Dmol-min.js` by any means to one of
  the lookup paths, or point `CHEMFLOW_3DMOL_PATH` at an existing copy:
  ```bash
  export CHEMFLOW_3DMOL_PATH=/path/to/3Dmol-min.js
  ```

This only needs to be done once per machine/account; the file persists.

---

## 2. Enable WebGL (headless / HPC rendering)

3Dmol.js creates a **WebGL** context. On a remote/headless Linux session the
embedded Chromium can't use the hardware GPU, and recent Chromium blocks WebGL in
plain software mode — the viewer then fails with
`TypeError: Cannot read properties of null (reading 'getParameter')`.

ChemFlow handles this automatically: on Linux (unless you opt into a real GPU) the
desktop sets Chromium to use **SwiftShader** (in-process software GL, which
supports WebGL) before starting:

```
QTWEBENGINE_CHROMIUM_FLAGS=--use-gl=angle --use-angle=swiftshader
  --enable-unsafe-swiftshader --disable-gpu-compositing --ignore-gpu-blocklist --no-sandbox
QT_OPENGL=software
```

Notes:

- This is applied via `os.environ.setdefault`, so if you **export
  `QTWEBENGINE_CHROMIUM_FLAGS` yourself** the app won't override it — `unset` it to
  let ChemFlow configure WebGL, then relaunch.
- On a machine with a working GPU, opt into hardware GL for faster rendering:
  ```bash
  export CHEMFLOW_DESKTOP_GPU=1
  ```
- Any flag change requires a **full restart** of `chemflow-desktop` (flags are read
  once at startup).
- The `Failed to connect to the bus … system_bus_socket` message in the log is
  harmless (no D-Bus on the node) and unrelated to rendering.

---

## 3. View a predicted pose

1. Run a docking job (localhost or HPC). Boltz-2 writes the structure a few levels
   deep under your output directory, e.g.
   `docking_output/boltz_results_boltz_input/predictions/<id>/<id>_model_0.pdb`
   (Chai-1 writes mmCIF). Find it with:
   ```bash
   find <workdir>/docking_output -name '*_model_0.pdb' -o -name '*_model_0.cif'
   ```
2. In the **Predicted complex** field, Browse to that file (or paste the path).
3. Click **Show binding pose**. The receptor renders as a cartoon and the ligand as
   sticks, zoomed to the binding site.

### Binding data (Boltz-2)

The **Binding data** panel scrapes the JSON files written next to the structure
(`confidence_*.json`, `affinity_*.json`) and shows:

- **Confidence score**, **pTM**, **ipTM (interface)**, **(complex) pLDDT**
- **Predicted affinity** (`affinity_pred_value`) and the derived **pIC50** and
  **IC50**, when affinity prediction was enabled.

Affinity only appears if the run had it on — tick **Predict affinity** on the
docking page (Boltz-2 only), or set `predict_affinity: true` and
`affinity_binder: <ligand id>` in the config. Boltz-2 reports
`affinity_pred_value = log10(IC50 in µM)`, so **pIC50 = 6 − affinity_pred_value**
(treat it as relative ranking within a congeneric series, not an absolute potency).

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| Panel shows *"3Dmol.js not found…"* | No local library, no internet | Run `chemflow vendor-3dmol` on a node with internet (§1) |
| Panel stays blank / `getParameter` is null | WebGL unavailable (software mode) | Ensure the SwiftShader flags apply (unset your own `QTWEBENGINE_CHROMIUM_FLAGS`), restart (§2) |
| Structure shows but no scores | Affinity off, or wrong path | Enable **Predict affinity**; point at the real `predictions/<id>/…_model_0.pdb` (the JSONs live beside it) |
| Nothing loads at all | WebGL truly unsupported on this node | Use `CHEMFLOW_DESKTOP_GPU=1` on a GPU node, or open the `*_model_0.pdb` in PyMOL/ChimeraX/Mol* |
