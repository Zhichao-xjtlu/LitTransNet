# Polarity contract for literature-derived rules and Network V5

## Scope

This contract applies to every compound class. It is especially important for
ginsenosides, where positive and negative ionization can produce different
adducts, fragments, neutral losses, and diagnostic behavior.

## Acquisition metadata

1. A network run must receive polarity from either an explicit single-mode run
   manifest or per-spectrum `query_ion_mode_arr` metadata.
2. Filename-based polarity inference is prohibited.
3. Mixed positive/negative input requires per-spectrum metadata.
4. Positive and negative experiments should normally be processed as separate
   runs and compared only at the entity-audit layer.

## Rule semantics

- Neutral compound formula, exact mass, and ordinary chemical transformation
  relations are mode-agnostic unless the literature explicitly restricts
  them.
- Precursor adduct observations are mode-specific.
- Compound-specific observed or theoretical fragment evidence is
  mode-specific when the source reports polarity.
- Diagnostic-fragment and neutral-loss rules are mode-specific when the source
  reports polarity.
- A rule explicitly marked `negative` must never support a positive-spectrum
  identity hypothesis, and vice versa.
- For the ginsenoside experiment, blank fragment/diagnostic/neutral-loss
  `ion_mode` means `unknown`, not `both`. Such evidence may remain explanatory
  but must not independently satisfy a target-identity gate.

## Collision energy

Collision energy should be preserved as audit metadata when available, but is
not a required identity gate in the initial ginsenoside migration. Exact CE
matching would be too brittle across instruments and acquisition methods.
Future scoring may use broad low/medium/high CE bins or energy-aware spectral
models after standard-spectrum calibration.

## Implementation status

The generic V5 match stage now:

- reads query polarity per MS2 spectrum from mzML metadata;
- partitions MSP entries by explicit polarity or adduct;
- prohibits cross-polarity target and decoy matches;
- preserves query/library collision energy as audit metadata;
- writes `query_ion_mode_arr` and a known-match polarity-consistency flag;
- uses the bounded `ppm AND absolute Da` precursor criterion.

Network V5 rejects polarity-inconsistent known seeds and propagates edges only
between nodes of the same ion mode.

## Remaining rule-evidence gap

The current betalain publication bundle predates this contract:

- its historical match pickle has no `query_ion_mode_arr`;
- the historical regression explicitly assigns all spectra to positive mode;
- compound, diagnostic-fragment, and neutral-loss rule `ion_mode` fields are
  blank;
- blank rule modes are currently treated as applicable to either polarity;
- collision energy is not yet used in rule scoring (by design for V5.0).

These historical behaviors are retained only for betalain regression
compatibility. For ginsenosides, Agent 2/3 still must preserve mode-specific
compound fragments in the five-table evidence contract. Blank GS fragment,
diagnostic, or neutral-loss mode cannot independently satisfy target identity.
