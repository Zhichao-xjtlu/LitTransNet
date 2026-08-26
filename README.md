# LitTransNet

LitTransNet converts local natural-product literature into an auditable entity
registry and executable reaction network, then uses the network to extend
LC-MS/MS annotation from a sparse spectral library.

This repository is the minimal runnable package. It contains the production
pipeline only; manuscript files, figures, tests, benchmarks, historical code,
external-tool outputs, literature PDFs, spectral libraries, and experimental
data are not included.

## Included workflow

1. Convert local PDF/TXT/CSV/XLSX literature into chunks.
2. Build a BM25S index using the scientific tokenizer.
3. Generate class-level queries and extract evidence claims through an
   OpenAI-compatible API.
4. Compile the evidence into a schema-4.0 entity registry and five executable
   rule tables.
5. Match an MS-DIAL alignment table and GNPS-compatible MGF against an MSP
   library.
6. Expand resolved library anchors through polarity-compatible, chemically
   validated reaction relations.

## Installation

Python 3.11 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Set the OpenAI-compatible endpoint only for query planning and literature
mining:

```powershell
$env:OPENAI_BASE_URL="https://your-endpoint.example/v1"
$env:OPENAI_API_KEY="your-api-key"
$env:OPENAI_MODEL="your-model-name"
```

## Literature-to-rules

```powershell
python rag/scripts/rag_prepare_corpus.py `
  --literature_dir literature/your_class `
  --output_jsonl work/your_class/corpus/chunks.jsonl

python rag/scripts/rag_build_bm25_index.py `
  --corpus_jsonl work/your_class/corpus/chunks.jsonl `
  --index_path work/your_class/index/retrieval_index

python rag/scripts/rag_query_planner.py `
  --compound_class your_class `
  --output_json work/your_class/query_plan.json

python rag/scripts/rag_run_agentic_pipeline.py `
  --compound_class your_class `
  --query_plan work/your_class/query_plan.json `
  --corpus_jsonl work/your_class/corpus/chunks.jsonl `
  --index_path work/your_class/index/retrieval_index `
  --output_root work/your_class/run
```

## Match and network

Supply one MS-DIAL alignment table, one GNPS-compatible MGF, and an MSP
library for each ion-mode run. The following example uses the included frozen
betalain rule bundle:

```powershell
python dl_annotator/v5/run_match_network_v5.py `
  --experiment_manifest experiments/betalain/experiment.json `
  --ion_mode positive `
  --rules_dir examples/rule_bundles/betalain_schema4 `
  --feature_table path/to/alignment_table.txt `
  --spectra_mgf path/to/representative_spectra.mgf `
  --stage all
```

For ginsenosides, use
`examples/rule_bundles/ginsenoside_schema4` and the corresponding experiment
manifest. The manifests define the expected data and library directories;
these files must be supplied locally.

## Package layout

- `rag/core`: production retrieval, evidence, registry, chemistry, and compiler
  modules required by the canonical pipeline.
- `rag/scripts`: five production command-line entry points.
- `dl_annotator/v5`: production feature matching and reaction-network runtime.
- `examples/rule_bundles`: two complete, hash-validated schema-4.0 bundles.
- `experiments`: betalain and ginsenoside runtime manifests.
- `docs`: feature-input and polarity contracts.

The package excludes API credentials. Copy `.env.example` only as a local
configuration reference; do not commit secrets.

No software license has been selected. Add the intended license before making
the repository public.

