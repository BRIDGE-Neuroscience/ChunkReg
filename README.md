> [!NOTE]
> **ChunkReg is just a scaling wrapper, not a new registration method.** The
> registration itself is
> [anatomix by neel-dey](https://github.com/neel-dey/anatomix/tree/main/anatomix/registration):
> its learned feature channels, its MIND-SSC descriptors, and its FireANTs
> backend running the moments, rigid, affine and greedy stages. What ChunkReg
> adds is the chunking, the resolution pyramid, configs and cluster scripts for
> running at scale (mainly built for HiP-CT X-ray microscopy datasets)


<p align="center">
  <img src="docs/chunkreg-logo.png"
       alt="ChunkReg — chunked metadata wrapper for registration in massive volumes"
       width="440">
</p>

Managing registration for 3D volumes that are too large for one GPU. ChunkReg
cuts each volume into fixed-size chunks, registers every chunk on a GPU with
anatomix features and the FireANTs backend, and blends the results into one
displacement field per volume. It works on two volumes (a fixed and a moving one)
or on a cohort (an unbiased group template plus one field per subject). Volumes
are OME-Zarr and are never held whole in memory.

**Documentation:** start at [docs/index.md](docs/index.md). It is built with
MkDocs (`pip install -r docs/requirements.txt && mkdocs serve`).

## Install

```bash
pip install -e ".[store,io]"        # core, zarr, TIFF/NIfTI readers
pip install -e ".[gpu]"             # GPU path: torch, MONAI, SimpleITK
pip install git+https://github.com/neel-dey/anatomix.git
bash anatomix/registration/registration_backend/install_fireants.sh
```

The CPU reference engine needs only the first line. On Windows ARM64 there is no
`numcodecs` wheel, so use an x64 Python for zarr.

## Register a pair of volumes

No config file. Neither volume has to fit in memory.

```python
import chunkreg

result = chunkreg.register(
    fixed="fixed.ome.zarr", moving="moving.ome.zarr", root="work/", progress=print
)
result.export_moving("moving_in_fixed.nii.gz")   # on the fixed volume's own lattice
```

`result.field` is the displacement in millimetres, mapping fixed coordinates to
moving ones. Call `register` again with the same arguments to resume a run that
stopped. Two subjects of a cohort: `chunkreg pair run.json --fixed s01 --moving s02`.

## Build a cohort template

```bash
cp configs/example.json run.json     # set root and runner, and list your subjects
chunkreg setup run.json              # ingest, check, measure, print the plan
chunkreg run run.json                # run the levels; run again to resume
```

Outputs are in `<root>`: the template at `levels/L0/template.zarr`, one field per
subject at `fields/<id>.zarr`, and a QC sheet per pass under `levels/L<k>/qc/`.
Warp a subject with `chunkreg apply subjects/s01.zarr fields/s01.zarr out.zarr` and
write a NIfTI with `chunkreg export out.zarr out.nii.gz`.

Levels are numbered as OME-Zarr numbers its datasets: level 0 is full resolution,
and the highest level is the coarsest, the one where the whole volume fits one
chunk. The loop runs from the coarsest down to 0.

## What makes it scale

| Idea | Effect |
|---|---|
| One chunk geometry at every level (core 256, halo 48) | One memory footprint and one resource request from the coarsest level to the finest. |
| A resolution pyramid that ends in a single chunk | The global alignment is estimated where the whole volume fits; each finer level refines a seed. |
| The halo sets the displacement clamp | Chunks cannot disagree by more than the halo can absorb; overlaps are blended with a partition of unity. |
| Owner-computes shards and resumable passes | No locks, and a killed job resumes at the first unfinished pass. The same code runs on one GPU, a multi-GPU node, SGE or SLURM. |
| Copy nothing that can be read in place | Linked scans and pyramids, `warm_start` (a scan sampled through an affine, never resampled), linked templates. |
| Bounded transient data | Task arrays are deleted as soon as they are blended, can live on node-local `scratch`, and the residual QC maps have a size cap. |

## Credit

ChunkReg is a scaling wrapper around [anatomix](https://github.com/neel-dey/anatomix)
by Neel Dey and the [FireANTs](https://github.com/rohitrango/FireANTs) registration
backend. If you use it, cite both; the references are in their repositories.
