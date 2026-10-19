# ChunkReg

<p align="center">
  <img src="chunkreg-logo.png"
       alt="ChunkReg: chunked metadata wrapper for registration in massive volumes"
       width="440">
</p>

ChunkReg registers 3D volumes that are too large for one GPU. It cuts each
volume into fixed-size chunks, registers every chunk on a GPU, and blends the
chunk results into one smooth displacement field. Volumes live on disk as
OME-Zarr and are never held whole in memory.

!!! note "ChunkReg is a scaling wrapper, not a new registration method"
    The registration itself is
    [anatomix](https://github.com/neel-dey/anatomix/tree/main/anatomix/registration)
    by Neel Dey: its learned feature channels, its MIND-SSC descriptors, and its
    [FireANTs](https://github.com/rohitrango/FireANTs) backend running the
    moments, rigid, affine and greedy stages. ChunkReg adds the chunking, the
    resolution pyramid, configs and cluster scripts for running at scale
    (mainly built for HiP-CT X-ray microscopy). If you use it, cite anatomix and
    FireANTs; the references are in their repositories.

## Two ways in

| | You give it | You get |
|---|---|---|
| **Registration** | a fixed and a moving volume | the moving volume on the fixed volume's lattice, and the displacement field |
| **Coregistration** | a cohort of volumes | an unbiased group template, and one displacement field per subject |

Both are the same pipeline. Registration holds the fixed volume as the template
instead of estimating one. See [Running](running.md).

## When to use it

- A volume or cohort is too big to register on one GPU at the resolution you want.
  HiP-CT scans at 20 to 50 µm are the design target.
- The volumes are the same kind of object but not necessarily the same modality.
  The similarity is computed on anatomix feature channels.
- You want a run that resumes after a killed job, on one node or a cluster.

If both volumes fit on one GPU at your target resolution, run anatomix's own
registration directly. ChunkReg adds nothing there: the whole volume is one
chunk.

!!! info "Figure placeholder"
    Overview: a volume cut into cores with halos at three pyramid levels, one
    chunk's task array, and the blended field. To be drawn.

## Quickstart

```bash
git clone https://github.com/BRIDGE-Neuroscience/ChunkReg.git && cd ChunkReg
pip install -e ".[store,io,gpu]"
pip install git+https://github.com/neel-dey/anatomix.git
bash anatomix/registration/registration_backend/install_fireants.sh   # run from an anatomix checkout
cp configs/example.json run.json     # set root, runner, and list your subjects
chunkreg setup run.json              # ingest, check, measure, print the plan
chunkreg run run.json                # build the template; run again to resume
```

For two volumes, no config file is needed:

```python
import chunkreg
result = chunkreg.register(fixed="fixed.ome.zarr", moving="moving.ome.zarr", root="work/")
result.export_moving("moving_in_fixed.nii.gz")
```

## Where to go next

| Page | Read it for |
|---|---|
| [Concepts](concepts.md) | the run grid, level numbering, chunks and halos, the loop, stopping rules |
| [Configuration](configuration.md) | every config key, with defaults |
| [Large scans](large-scans.md) | what makes a 20 µm run feasible, and how to budget disk and time |
| [Running](running.md) | one GPU, several GPUs, SGE, SLURM, resuming, stopping early |
| [Outputs and QC](outputs-qc.md) | the run root, QC sheets, residual maps, applying and exporting fields |
| [Command line](cli.md), [Python API](api.md) | terse references |
