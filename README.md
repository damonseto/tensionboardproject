# Tension Board Climb Generator

A small GPT that generates boulder problems for the Tension Board 2, conditioned
on board angle and grade. Built on ~15.5k community climbs pulled from the
Aurora sync API.

Inspired by [KilterGPT](https://climbest.app/blog/kiltergpt)

---

## What it does

Ask for a grade and an angle, get a climb:

```python
climb = generate(model, angle=40, grade=20.0)   # ~V5 at 40 degrees
plot_climb(climb, holds)
```

![V3](examples/example_v3_40) ![V5](examples/example_v5_40) ![V7](examples/example_v7_40)
Board rules are enforced during decoding rather than left to the model:
holds and roles must alternate, no hold repeats, at most two starts, one finish.

---

## Data

Pulled with [BoardLib](https://github.com/lemeryfertitta/BoardLib):

**Filters applied:**

| Filter | Reason | Rows remaining |
|---|---|---|
| `layout_id = 10` (TB2 Mirror) | Placement IDs are layout-scoped; mixing layouts silently corrupts data | 54,000 climbs |
| `ascensionist_count >= 5` | Grades are community averages — 2 ascents means 2 opinions | 15,853 pairs |
| `n_holds <= 20` | Longer entries are circuits, not boulder problems (~2%) | 15,543 pairs |

**The training unit is a (climb, angle) pair, not a climb.** `climb_stats` holds
a separate row per angle, each with its own difficulty and ascent count.

---

## Representation

A climb is stored as `p466r8p477r6...` — placement ID and role ID per hold.
Role IDs are per-product; for TB2 (product 5): 5 = start, 6 = hand, 7 = finish,
8 = foot.

```
[BOS, p466, r8, p477, r6, ..., EOS]
```

Vocabulary is 697: 3 specials + 4 roles + 690 placements. Interleaving keeps the
vocabulary small and dense (every hold is seen in all four roles) and makes
constrained decoding trivial — odd positions must be roles, even must be holds.

Placement IDs are sparse (690 IDs scattered across 304–1491), so the tokenizer
maps them to contiguous indices. Encoding is round-trip tested on all 15,543
climbs.

---

## Grade predictor

I built this before the generator, partly as a warm-up and partly to check the pipeline worked. It also ended up useful as a filter, since asking the GPT for a V5 doesn't guarantee you get one.

Same model throughout (`HistGradientBoostingRegressor`, 300 iterations), only
the features change. Split grouped by climb uuid so no climb appears in both
train and validation.

| Features | Count | MAE | Within 1 |
|---|---|---|---|
| Baseline — always guess training mean | 0 | 3.90 | — |
| Binary hold vector + angle | 691 | 1.56 | 39.8% |
| Hold × role one-hot + angle + hold count | 2,762 | 1.46 | 43.5% |
| Geometry only — spans, reach gaps, hand/foot counts | 8 | 2.21 | 29.2% |
| Combined | 2,770 | **1.34** | **43.9%** |

**Hold identity dominates.** Geometry alone (2.21) loses badly to hold identity
alone (1.56), despite being the only spatial information. On TB2 the holds vary
enough in difficulty that knowing *which* ones are lit beats knowing how they
are arranged.

**Geometry still adds ~8% on top** (1.46 → 1.34), but within-1 barely moves — so
it shrinks large errors rather than sharpening near-misses.

Not directly comparable to published Moonboard results (~46% exact, ~85% within
one grade); those are discrete V-grades where rounding helps, this is the
continuous Aurora scale.

---s

## Generator

Decoder-only transformer, written from scratch:

- 978k parameters — 4 blocks, 4 heads, d_model 128
- Angle and grade normalized to 0–1 and prepended as a conditioning prefix
- Causal masking, learned position embeddings
- Trained on CPU

**Training:** validation loss bottomed at **epoch 8 (2.03)**, then rose steadily
to 2.41 by epoch 19 while training loss kept falling to 0.99 — textbook
overfitting. Checkpointing on validation improvement keeps the epoch-8 model.

Perplexity at the checkpoint is ~7.6, meaning the model has narrowed 697
possible tokens to roughly 8 plausible ones per position.

---

## What didn't work

**Fine-tuning on high-quality climbs.** KilterGPT fine-tunes its base model on a
curated high-quality subset. Tried the same: 1,937 climbs at `quality_average >=
2.90` and `>= 20` ascents, lr 3e-5, 5 epochs.

Training loss barely moved (1.537 → 1.492), validation drifted up slightly, and
generated climbs were **indistinguishable from base in a blind comparison**.

*Diagnosis:* Tension's quality ratings are too compressed to be a useful filter.
On a 1–3 scale the mean is 2.81 with an interquartile range of 2.74–2.93 —
almost everything is rated good, so a "high quality" subset is close to a random
subset.

*Would work if* a better quality signal existed. Ascent count spans a much wider
range and is probably the stronger proxy.

---

## Known limitations

**Angle coverage is heavily skewed.** 40° holds 41% of rows, 45° holds 18%.
Below 20° and above 55° there are only hundreds or dozens. Expect good results
at 40–45, unreliable at the extremes.

**Grades cluster in the middle.** Mean difficulty 19.1, IQR 15.8–22.8 — roughly
V2 to V7. Conditioning degrades at both ends.

**No hold type or orientation data.** `placements` has only
`id, layout_id, hole_id, set_id, default_placement_role_id` — no crimp/jug/sloper
label, no orientation. The model learns hold difficulty purely from
co-occurrence. A generated climb can be unclimbable because a hold faces the
wrong way, and nothing in the data would catch it.

**Occasional hands above the finish hold**, which is not climbable. Currently
unhandled — needs either a decoding constraint or post-hoc rejection.

**Novelty is unmeasured.** No Jaccard similarity check against the training set
yet, so it is not yet established how often the model regurgitates real climbs.

