# Feature input contract

## Purpose

The annotation pipeline operates on aligned LC-MS features, not individual
raw MS2 scans. MS-DIAL performs peak detection, deconvolution, alignment and
representative-spectrum selection. Its GNPS export supplies the two runtime
inputs:

1. one alignment quantification table (`.txt`, `.tsv`, or `.csv`);
2. one representative MS/MS spectral summary (`.mgf`).

One accepted table row linked to one MGF spectrum is one query and one network
node.

## Required feature fields

Header matching is case- and punctuation-insensitive. Common MS-DIAL aliases
are accepted.

| Logical field | Examples |
| --- | --- |
| feature ID | `Alignment ID`, `row ID`, `FEATURE_ID` |
| precursor m/z | `Average Mz`, `row m/z`, `Precursor m/z` |
| RT in minutes | `Average Rt(min)`, `row retention time`, `RT (min)` |

Ion mode, adduct and charge are optional table metadata. All original columns,
including sample abundance columns, are preserved in the match pickle as JSON
metadata; they are not identity gates.

## MGF linkage

MGF blocks require `PEPMASS`, peaks, and a feature identifier in one of
`FEATURE_ID`, `ALIGNMENT_ID`, `ROW_ID`, or `SCANS`. A feature/alignment ID in
`TITLE` is accepted as a fallback. Linkage is exact after only trivial numeric
normalization such as `12.0` to `12`.

The adapter reports:

- table features without MS2;
- MGF IDs absent from the table;
- duplicate MGF spectra per feature;
- polarity conflicts;
- table-to-MGF precursor and RT conflicts.

If multiple MGF blocks map to one feature, the representative with the highest
total ion current is selected deterministically and the multiplicity is
audited. MS-DIAL GNPS export normally provides one representative spectrum.

## Polarity and RT

Positive and negative alignments must be run separately. Explicit table or MGF
polarity that conflicts with `--ion_mode` is rejected. If export metadata omits
polarity, the explicitly requested single-mode run supplies it.

RT differences are audited as an input-quality warning but are not rejected:
an alignment-average RT and the representative scan RT can legitimately
differ. RT does not establish compound identity.

## Near-tied isomeric library seeds

Library competition is performed after replicate spectra have been collapsed
to the best spectrum per cleaned chemical name. When the best entity-level
match passes the normal score, fragment-count, q-value, precursor-mass and
polarity gates, every other passing entity within the configurable absolute
cosine margin (`--seed_competitor_margin`, default `0.01`) becomes a seed
hypothesis for that feature.

Such a feature is exported as `ambiguous_library_seed`; it is not reported as
an exact known identity. Each candidate entity may independently enumerate its
outgoing literature reaction rules. Downstream targets still have to pass the
normal target mass and fragment-evidence gates. This policy is class-agnostic
and is intended for isomer-rich families such as saponins as well as betalains.

## Command

```powershell
python dl_annotator/v5/run_match_network_v5.py `
  --experiment_manifest experiments/betalain/experiment.json `
  --ion_mode positive `
  --stage all
```

Paths can be overridden without editing the manifest:

```powershell
python dl_annotator/v5/run_match_network_v5.py `
  --experiment_manifest experiments/betalain/experiment.json `
  --ion_mode positive `
  --feature_table path/to/quantification_table.txt `
  --spectra_mgf path/to/representative_spectra.mgf `
  --stage all
```
