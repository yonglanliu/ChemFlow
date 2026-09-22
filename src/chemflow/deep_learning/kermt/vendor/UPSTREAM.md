# Vendored KERMT source

This directory contains source adapted from
[`NVIDIA-BioNeMo/KERMT`](https://github.com/NVIDIA-BioNeMo/KERMT) at commit
`8828743036675d1b6d5f4586ef7bcfea70233d59`.

The source was namespaced under `chemflow.deep_learning.kermt.vendor`, and
`cuik_molmaker` loading was made optional so the original RDKit featurization
path can run on platforms where the NVIDIA accelerator is unavailable.

Redistribution terms and third-party notices are retained in `LICENSE` and in
the headers of the corresponding source files.
