# Changelog

## 0.9.0 (unreleased)

Biological names throughout, brain regions, and a generic brain.

- `Records` (`cadence.records`): the records cortex. A reading of `inputs` units loses each
  unit's running mean (the plain average of the witnessed readings until one over their
  count falls to `habituation`, then a running mean at that rate); a fixed random expansion
  keeps the `active` most driven of `cells` cells at unit length; one delta-rule table per
  predicted field is read through the active cells and written by the error, at `rate` for
  consequence fields and at `valued_rate` for `valued` fields, which read a second code
  whose input `pathways` are divided by their running norms. `known` masks unobserved target
  entries and `to_dict` rebuilds the fixed cells from the seed. `tests/test_records.py`
  covers the codes, the write identity, a correlated walk, habituation, the valued code,
  the masks and the settling mean.
- `Records(backend="torch", device=..., precision=...)`: the records cortex on a torch device.
  The projection, the offsets, the running mean, the pathway norms and every record table stay
  there, and the drive product, the `active` winners, the reads and the delta-rule writes run
  there. The signatures do not change: readings arrive and codes and reads leave as NumPy
  float64 arrays, and `mean`, `pathway_norm` and `tables` read back from the device, so a
  checkpoint saves and restores the same numbers. CUDA defaults to float64; MPS needs
  `precision="float32"`. The winners come from `topk`, which selects the same cells as
  `argpartition` unless the drive at the boundary is tied, and the code is scattered into a
  zero row, so the order inside the set never matters. Reduction order is the device library's:
  over 300 witnessed readings and writes of a 957-input, 4,000-cell cortex on an A10G in
  float64 the largest deviation from the NumPy path was 3.6e-16 in a code, 4.4e-16 in a read
  and 2.9e-16 in a record, and every winner set matched. `tests/test_records_torch.py` compares
  the codes, the reads, 200 writes, the pathway norms, the task sets, the fan-in, the unvalued
  code and the state accessors against the NumPy path.
- `Mulberry32`: the 32-bit generator of `brain_scan.js`, with `batch` and Box-Muller
  `normals`. `Records` draws its projection and offsets from it, so a page rebuilds the same
  cells from the seed.
- The evidence trees `experiments/` and `benchmarks/` leave the repository together with the
  tests bound to them, `docs/comparisons.md`, whose rows cited them, and every documentation
  reference to them. `docs/assets/cadence-logo-prompt.md` is removed.
- Documentation: `docs/cortex.md` (regions, the catalogue, projections, ports, the synapses
  a head owns, when to settle and when to record), `docs/brain.md` (development, checked
  settling, one experience step by hand with records beside a policy head, `GenericBrain`,
  checkpoints, the browser page) and `docs/evolution.md` (mutation, selection over short
  lives, parallel lives, equilibrium detuning). Their snippets run in the documentation
  tests, together with those of `docs/concepts.md` and `docs/certificate.md`.
- Documentation: the record principle in the README, the concepts, the records section of
  `docs/memory.md`, and the learning, reward, experience, task, continuous-interaction,
  rehearsal, content-memory and sequence pages. The README links the two examples, their
  receipts and the gallery.
- Documentation corrections: the certificate example uses a certified brain and states that
  `steps_for` raises for an uncertified one. The API reference adds `Records`,
  `Mulberry32`, `PatternSeparator`, the certificate, the atlas, `Lineage`,
  `FastSynapses(separator=)`, `Learner(synapse_rate=)`, `Learner.apply`,
  `Learner.contrast_rows`, `save(..., compressed=)` and the keyword-only arguments of
  `record_settlements`, `SequenceCache` and `BoundedTrace`. The learning knob table gives
  the defaults of `eta_bias` and `layered(density=)` and adds `synapse_rate`; the viewer
  page gives `particleBudget` as 300,000, `subsample_edges(limit, seed=None)` and
  `page(note=, inputs=)`. The pages drop contrastive and status wording.

- The README and quickstart center one ongoing experience loop without a
  training/inference mode switch. Retired example links and the bundled demo catalog
  are removed; their behavioral checks remain in the test suite. Minimal-install CI
  runs the actual quickstart from the built wheel.
- Action caches refresh after parameter changes, including reward updates and checkpoint
  resume. Memory and trace queries preserve live streams when the query batch differs.
- Complete-brain loading validates continuation arrays, counters and pending actions.
  Overflowing fast-memory writes preserve previous records and separator state.
- `ContentMemory` provides competitive prototype recall; `ReservoirReplay`
  retains a bounded uniform sample of real observations for rehearsal. Both
  expose their memory budgets and keep observation separate from pure reads.
- `cadence.sequence` adds causal content readback and bounded activity traces,
  with fixed coordinate frames per read and stable extreme-value arithmetic.
- Persistent memory now honors expanded separator keys, and `GenericBrain`
  checkpoints preserve the actual separator projection and running mean.
- Float64 Torch residual checks can stay on device; float32 keeps the independent
  host reference. Transport avoids unnecessary accelerator synchronizations.
  `ep_structure` checks effective free/free symmetry under declared fixed inputs.
- Host, compiled and accelerator paths factor small phase contrasts before products
  to reduce cancellation, including per-stream reward traces and restored phases.
- `layered(skip=True, skip_init=0.0)` adds a learnable direct sensory-to-motor
  route with unchanged initial predictions; the existing initializer is the default.
- Discrete reward eligibility now differentiates the sampled softmax policy even
  when imitation uses quadratic nudges. Native categorical motor slots can have
  unequal sizes and never sample padding. Numerical derivative and CPU/Torch
  regression tests cover both contracts.
- `ActorCriticConfig(critic_signal="td")` lets the critic fit return in reward
  units independently of actor clipping/centring. The default `"modulated"`
  preserves the original bounded critic update. Reports distinguish raw TD error
  from the actor's modulation; the reward guide explains calibration and tuning.

- `cadence.certificate(brain)`: the settling certificate. From the largest absolute incoming
  effective weight sum (`row_mass`) and the neuron model's slope bound (`lipschitz_constant`)
  it reports the contraction rate `1 - dt (1 - L rho)`, whether the free phase is certified,
  the a posteriori error bound from the last step's movement, the a priori bound, and the
  warm-start step budget after a stimulus change. Under `learning_neuron_model()` a brain is
  certified below a row mass of 2. Documented in `docs/certificate.md`.
- `PatternSeparator` and `FastSynapses(separator=...)`: keys pass through a fixed random
  expansion and a k-winners-take-all before the record sees them, with an optional running
  mean removed first. Expansion can reduce overlap between correlated keys;
  exact noninterference requires disjoint supports and is not guaranteed by expansion.
  Documented in `docs/memory.md`.

- `cadence.atlas_of(brain)` / `build_atlas(connectome)`: the brain atlas, one integrated
  layout of the whole connectome. Regions (the populations) are placed by a force layout of
  the region graph, neurons inside a region by a whitened spectral embedding of their
  synapses, sheets by declared shapes, or by supplied coordinates. `atlas.frames` quantises
  recorded settlings, `atlas.page` writes a self-contained page, and `brain_scan_script()`
  returns the shipped WebGL2 renderer (`brain_scan.js`): tissue in region colours whose
  brightness is the activation, a hot scan-coloured glow where neurons change, synapses that
  light up as their presynaptic neuron changes, messages travelling along synapses, and an
  EEG-style montage of every region. Documented in `docs/pages.md`.
- `GenericBrain.step` coordinates ongoing perception, previous-action feedback, optional
  current demonstrations and the next action, without a training/inference mode switch.
- `SynapticMemory` adds shared persistent synapses and fading per-stream residuals.
  Repetition and salience consolidate only actually observed value components. Persistent
  weights survive episode/batch resets and can be revised or explicitly cleared. New
  `episodic=True` brains use this rule; older checkpoints keep their original fast memory.
- Complete brain checkpoints now include pending actions and their nudged states, so
  interaction can resume before reward arrives. Generic checkpoint format 2 loads format 1.
  `parameters()` includes consolidated synaptic storage. The continuous-learning guide
  documents the clocks, equations, biological analogy and measured retention checks.
- The imitation optimizer counts only its own contrasts for bias correction; interleaved
  reward/direct updates no longer advance an optimizer whose moments did not change.
  That count is checkpointed; older checkpoints retain their saved count as the fallback.
- Tests pin the default batch row of `fraction_active`, integer stimulus indices, the
  60-step `settle` default, and a host warm state that the float64 torch kernel shares
  memory with and must not write. The NumPy and torch settling loops update fresh arrays
  in place (#1, Jonathan Hill).

- Names: `Connectome` (was `Wiring`), `Brain` (was `Settlement`), `BrainState`,
  `NeuronModel` (was `GradedRule`), `learning_neuron_model`, neurons and synapses (were
  owners, overlaps and seams), `populations` (were `sets`), `synapses` (was `edges`),
  `stimulus` (was `clamp`), `efficacy` (was `edge_scale`), `plastic_synapses` and
  `plastic_neurons` (were `trainable_overlaps` and `trainable_owners`), `reciprocal` (was
  `symmetric`), `activity_change` (was `repair`), `FastSynapses` (was `FastSeams`),
  `Genome` and `develop` (were `Constitution` and `grow`), `cadence.circuits` with
  `assemble` and `reflex_arc` (were `cadence.brains`, `couple` and `sensor_motor`), and the
  modules `cadence.connectome`, `cadence.neuron`, `cadence.brain` and `cadence.genome`.
- `cadence.legacy` keeps every 0.8 name, module path, keyword, attribute and method as a
  deprecated alias for one release; each use warns with its replacement.
- Checkpoints are written in format 2 with the new entry names; format 1 loads.
- Dictionaries use the new names: `Connectome.summary()["populations"]`,
  `Brain.to_dict()["neuron_model"]` and `["efficacy_changed"]`, `Learner.to_dict()["brain"]`,
  and the conformance ledger's `declared_synapses`, `transmissions` and
  `undeclared_transmissions`. Error messages use the new vocabulary.
- `Region`: a blank group of neurons or a designed region with its own circuit, inputs and
  outputs. A `Genome` mixes both; `develop` lays designed circuits into the connectome and
  draws projections between region populations (`region/population`), and `mutate` keeps
  designed regions' sizes. Blank genomes develop to the same connectome as in 0.8.
- `cadence.regions`: `visual_cortex` (retinotopic input and feature maps with local
  receptive fields), `cortex`, `motor_cortex` and `prefrontal_cortex`.
- `GenericBrain`: a sensory region or visual cortex, an association cortex and a motor cortex
  in one connectome, with an external reward helper (`ActorCritic`), optional prefrontal working
  memory (a `Trace` of the association cortex) and an optional hippocampus (`SynapticMemory`)
  for one-trial records. It learns from labels (`fit`) and reward (`act`, `learn`), and its
  genome can be evolved.
- `ActorCritic.state`: the free phase of the latest moment.
- `Brain.equilibrate` checks the whole circuit's equation residual under an exact step
  budget and returns per-row convergence information, including for zero-budget checks.
- `GenericBrain.save/load` preserves the standard composition: critic,
  both optimizers, random state, eligibility, working memory and episodic records.
  Checkpoint replacement is atomic; malformed learner optimizer state is rejected.
- Reward decisions now honor changed observations, own cached inputs, clear stale greedy
  eligibility and reset ended rows before one next-state phase. Batch neighbours no longer
  receive extra settling when another stream ends. Eligibility survives host/device transitions.
- Generic learning validates labels and transitions before mutation, owns observation/action
  buffers, exposes the current learned brain and accepts truncation bootstrap values.
- Fixed renamed device-cache keys, dense transport dropping parallel synapses, frozen
  weights changing under clipping, and overlapping reciprocal/explicit ties. Large tie IDs
  no longer allocate by label magnitude. Complete parameter assignment refreshes each backend.
  Sparse learning uses a bounded-memory host contrast instead of an unsupported device
  block operation or a potentially full dense Gram allocation; bias-only reports stay finite.
- Fused nudges honor fractional quadratic masks and excluded softmax groups. Signed
  traces produce nonnegative bounded salience; population-code probabilities normalize per axis.
- Added validation for reward settings, masks, nudges, neuron indices, region ports and sizes,
  genome constraints and optimizer state. Zero-density layered projections are supported.
- `imagine` accepts optional adversarial pruning and an isolated `clone` callback, supports
  array-valued action sequences, and checks its budget before invoking another transition.
  `ActivityMonitor.reset` starts an independent monitoring episode.
- Documentation now includes a runnable README quickstart, a complete brain-design guide,
  biological-object/function mapping with support boundaries, and tested introductory snippets
  and links. Standard cortex builders remain optional functional analogues.
- Timing tolerates unavailable Unix statistics. Minimal wheel CI also runs on Windows.

## Historical changes before 0.9.0

The entries below describe earlier interfaces and documentation. Use the current
README and quickstart for the ongoing experience API.

- Class-label and output-port validation prevents negative indices, duplicate outputs
  and accidental batch broadcasting. Slotted accuracy averages over every row and slot.
- Single-state softmax nudges and empty nudge masks work consistently. Zero-step
  accelerator trajectories return an empty trajectory, and invalid step counts fail
  with a clear error before execution.
- Wiring validates endpoints before merging edges and preserves tiny signed weights.
  Shuffled controls preserve degree counts even when endpoint swaps overlap.
- Gain selection breaks score ties toward the smallest admissible gain and rejects a
  grid with no admissible candidate. Checkpoints preserve explicit precision, and
  traces retain an owned snapshot when callers reuse state storage.
- Beginner documentation and minimal-dependency installation checks were added.
- Sparse CPU settlement uses optional SciPy CSR transport, avoiding the batch-by-edge
  message array. NumPy-only installations retain segmented sums; local dynamics and
  parameter semantics are unchanged.
- Blocked PyTorch learners keep adaptive optimizer history and updates on the device.
  Host reads, edits and checkpoints remain supported. Parameter rebuilds preserve an
  explicit torch device, and in-place trainability-mask edits invalidate cached masks.
- `FastSeams.observe`/`recall`: direct key/value ports, with opt-in `rule="delta"`
  residual writes and per-row resets. Existing Hebbian modes retain their semantics.
  Independent LMS, interference, legacy, reset and validation tests cover the change.
- `Settlement.residual`: remaining potential/adaptation equation discrepancy, so
  saturation and tiny steps cannot masquerade as equilibrium.
- Fixed fused source freezing on activation saturation, fractional warm-mask handling,
  per-row batch masks, and `GradedRule.replace` losing the `Adaptation` object.
- Warm starts rebuild published activity under the current mask on CPU, PyTorch and
  MLX, including after changing or removing a fractional mask.
- Receipt verification messages distinguish checks actually performed. Documentation
  separates records, transient dynamics and equilibrium, and states gradient scaling.

## 0.8.1 (2026-09-14)

- A checkpoint saved by an earlier release loads. `checkpoint.load` keeps the fields of the
  saved configuration that the running `LearnerConfig` has and drops the rest; 0.8.0 raised
  on the retired `consolidate` and `restore` of a 0.7 checkpoint. The test saves a
  checkpoint with the retired knobs in its metadata and loads it.
- The valence's and the population code's returns are typed as arrays, for the mypy of the
  CI runner.

## 0.8.0 (2026-09-14): the condensation

The library condensed into its elements (`docs/concepts.md`): owners and seams, the
settlement, the contrast, the trace, the valence, and one step. Dropped from the top level,
each with what replaces it. The gates it passed: the core suite with the child's
experiments; the four example rungs' receipts verifying under it; the NES player's
one-frame afterimage brain retrained on it reading the held-out presses at the same
0.862 / 0.325 / 0.579 bits as on 0.7.1.


- `Rehearsal`, `RehearsalConfig` (the clipped rehearsal learner): the trace and the valence
  of `ActorCritic`; E4 of the NES player measured the rehearsal a failure.
- `ValueNet`, `ValueConfig` (a separate value net as the critic): the critic is the
  settlement's own value readout, `ActorCritic(learner, critic=<owners>)`.
- `Population` (the continuous population code with Gaussian exploration): `Bins`.
- `DreamActorCritic`, `DreamConfig`, `DiscreteCode`, `actor_critic_wiring` (`dream.py`) and
  `Seams`, `SleepConfig` (`structure.py`, fast and slow strengths, pruning and sprouting):
  removed; the consolidation of `LearnerConfig` is the slow strength that stays.
- `ActorCriticConfig`: `critic_init` (the critic starts at zero), `lam_critic` (the critic's
  trace decays as the actor's), `normalize_floor` (a constant) and `center_per_stream` (the
  valence is per stream, always: one brain playing several games keeps a level for each)
  are gone; sixteen fields become twelve.
- `LearnerConfig.scale_floor`, `scale_cap`, `target_level`, `off_level`: never set by any
  script in five repositories; the cap is the constant `cadence.learning.SCALE_CAP` (8), the
  nudge's targets one and zero, and a seam may cross zero.
- `LearnerConfig.consolidate` and `restore` (a slow copy of the seams the fast ones are
  pulled back toward): removed; the NES player's E4 measured it no better than without at
  the settings tried, and the trace and the valence carry what it was for.
- `Ledger`, `settle_owner_by_owner`, `canonical_sha256`, `source_manifest`, `mutate`: still in
  `cadence.reference`, `cadence.receipts` and `cadence.constitution`, no longer top level.

## 0.7.2 (not released; shipped in 0.8.0)

- `Trace`: one class for the memory of the moment before, per stream, clamped into the next
  settlement: the trace of a range's activation, weighted by each owner's movement since the
  last moment when `focus` is above zero (`source` and `target` name the ranges). `Echo` is
  the `Trace` at focus 0 into the `context` range and `Afterglow` the focused one into the
  `afterglow` range, both unchanged in use; `ringing` moved to the `Trace`.
- `Valence`: the reward less its expectation as its own element (the running level per
  stream, the reward's own units or its scale, the floor, the cap); `ActorCritic.valence` is
  the agent's, built from its config, and `delta_mean`/`delta_var` read through to it.
- The first two steps of the condensation planned in `docs/concepts.md`; the API doc and the
  README carry the new names.

## 0.7.1 (2026-09-14)

- `Afterglow`: an owned state like `Echo`, a fading picture of the preceding moments that
  is brightest where they changed: the trace of the hidden equilibria weighted by each
  owner's movement since the last moment (its share of the mean movement, to the power
  `focus`), entering as a clamp on the `afterglow` range. What the settlement just had to
  repair stays lit; what stood still fades. With `focus` 0 it is the Echo. `source` names the
  range traced: the hidden owners (the interpretation), or the input owners for an afterimage
  of the picture itself, which is the one that works: predicting the symbol seen one moment
  ago against twelve owners of always-on background, the echo reads chance (0.25), the
  afterglow of the interpretation 0.30, the unfocused afterimage 0.58 and the focused
  afterimage 1.00 (`tests/test_child.py`).
- `ActorCritic.salience` and `Afterglow.ringing`: the eligibility of every seam weighted by
  its pre owner's salience, set before `learn`; the afterglow's ringing (each source owner's
  trace over the row's mean, plus a floor) is one such salience, so that what is still
  ringing is what a signal writes through. Off unless set. In the small experiment (a cue
  paid three moments later against twelve owners of always-on background) the plain trace
  learns the cue too, since the background's eligibility is noise that averages out, and the
  weighting is a faster start on some seeds and no difference on others: a mechanism, not
  yet a gain.
- `tests/test_child.py`: the child's capabilities as small experiments, each from simpler
  components already here: the eligibility trace credits a press paid three moments later
  (hit rate 1.00 with the trace, 0.47 without); the dopamine that is quiet for the usual
  reward and speaks for a missing or a larger one, in proportion; the afterimage above; and
  the settlement as a memory (a capped repair from the previous equilibrium still holds the
  moment before, a full repair forgets it).
- `ActorCriticConfig.center_per_stream`: the centred dopamine keeps one running mean and
  scale per stream instead of one over the batch, for streams on different tasks (one brain
  playing several games, whose rewards differ in size); the batch-wide centre is unchanged
  and stays the default.
- `ActorCriticConfig.dopamine_floor`: the centred dopamine within that many scales of its
  mean is nothing, so the broadcast is quiet while the reward is what it usually is and
  speaks only for a surprise; with the floor at 0 (the default) nothing changes.
- `ActorCriticConfig.center_scale`: whether the centred dopamine is divided by its running
  scale (the default, a unit signal whatever the reward's size) or left in the reward's own
  units, so a stage cleared is ten coins and not one.
- `LearnerConfig.momentum` and `normalize` are the adaptive local step proper: Adam's order
  (the running average of each seam's own contrast, divided by the RMS of its raw contrast)
  with both corrected for their short history, as `ActorCritic` already did. Before, the
  first steps were inflated and the two were composed the other way round, which is why they
  hurt on Pong; the corrected step lived in that rung's script only.
- `Learner(slots=(9, 2, 2))`: slot sizes instead of a count, for a controller whose
  choices differ in size (a move of nine, a grip of two); `slot_sizes`, `slot_offsets`;
  the checkpoint keeps the slots.
- `mutate(..., tied=((leader, follower), ...))`: region pairs whose sizes move together (a context range with one owner per hidden owner); `evolve` passes it through.
- `LearnerConfig(consolidate=, restore=)`: a slow copy of the seams and biases that follows them by `consolidate` per update and toward which every update pulls them back by `restore`, on the host and on the device: what is learned and kept settles into the slow copy, what is learned and then unlearned drifts back to it.
- `ActorCritic`: streams settled on the torch kernel keep their eligibility traces and the step on the device (`contrast_rows`: each stream's own contrast (the contrast through `contrast_tensors`, the step through the learner's device-side apply); the host arrays are never made. A reward-learning decision on a 7.2-million-seam brain: 121 ms on the Mac's GPU against 320 ms on the host.
- `Settlement.with_parameters` on the host engine: a copy whose derived arrays are remade on first use, not a rebuilt engine (60 ms a decision at 7.2 million seams).
- `Constitution.from_dict`: a constitution read back from a lineage's record, the inverse of `to_dict`.
- `evolve(..., report=callback)`: the lineage so far after every generation, so a long run is written out as it goes.
- `evolve(..., mapper=map)`: a generation's lives run through `mapper`; pass a pool's `map` to run them side by side (the fitness must then be a module-level function).
- `FastSeams`: fast Hebbian seams between two ranges of a batch of streams, as owned state
  (an outer-product write per step, a read as a drive, a decay); `columns` (slices for
  contiguous ranges) exported from `cadence.stream`.
- The torch kernel: the block weights scattered on the device through an edge index kept
  there; a settled state fetches its host arrays only when something reads them; no host
  copy of a warm state; the per-overlap arrays uploaded only on the gather-scatter path;
  no reference cycle in the device handle. The seam update addresses the paired and the
  tied overlaps through index arrays made once. A wiring keeps its segment boundaries.
  At 7,300 owners and 10.8 million overlaps, batch 2,048 on an A10G: one learning update
  4.4 s to 0.73 s.
- The learner's update stays on the torch device when both phases rest there: the contrast,
  the masks, the pairing, the tying, the decay and the bounds as device tensors, the
  kernel's blocks updated in place, and `Settlement.edge_scale` and `bias` fetched only
  when something reads them (a checkpoint, a receipt, a life of the seams). A test checks
  it against the host update to 1e-9. The seam update at the shape above: 265 ms to 40 ms.
  `normalize` and `momentum` take the host path.
- `Echo` reads and writes its ranges through slices.

## 0.7.0 (2026-09-11)

The accelerators, in the core.

- `backend="mlx"`: the settlement on Apple silicon through MLX (`pip install
  "cadence-net[apple]"`): the block transport as device matrix products in float32 on the
  unified memory, nudges, adaptation, masks, tolerance, trajectories and `repair` as on the
  other backends; checked against the CPU engine to 1e-4 with every feature.
- A device handle on `SettledState` (`state.device`): a settlement that continues from a
  state the same engine produced starts from the device copy instead of uploading the host
  arrays, and `Settlement.contrast_on_device` gives the learning rule its per-overlap
  contrast as block Gram products on the device, so only one number per overlap comes back.
  `Learner.contrast` uses it when both phases carry the handle; on torch/CUDA float64 the
  learned scales agree with the CPU engine to 2e-16, on MPS and MLX float32 to 2e-7.
- `available_backends()` lists `mlx`; the backends doc has a hardware guide (Apple silicon,
  Intel and AMD CPUs, NVIDIA) with measured updates at language-model shapes.

## 0.6.0 (2026-09-11)

The cost of a settlement step is now the cost of the owners that move.

- `cadence.blocks`: the block transport. The overlap matrix is stored as dense blocks
  between the contiguous ranges the wiring's named sets cut, and the product of a range
  whose activation did not change since the previous step is reused. The clamped inputs of
  a layered net are multiplied once per settlement instead of once per step; the inbox is
  the same sum in a different association order, and the conformance check against the
  owner-by-owner reference still holds to rounding (2e-16). `Settlement(..., layout=)` and
  `Settlement.layout`; `to_dict()` reports the layout. The `dense_limit` now bounds the
  block entries, so a layered net of several thousand owners settles on blocks.
- The fused kernel skips owners whose potential did not move and freezes ranges that hear
  nothing once they are still (exact: the update of such an owner is a fixed function of
  its own state), and no longer recomputes the activation of a state it continues from.
- The learning rule's contrast is one small Gram product per block, and one product
  instead of two for a range that is the same in both phases.
- On the MNIST shape (784 inputs, 256 hidden, batch 256) one learning update went from
  63 ms to 18 ms on one M4 core, and from 39 ms to 7 ms at 32 hidden owners; before, the
  cost of an update barely depended on the hidden width because the input block dominated.
- `cadence.timing`: `latency(decide)` times one decision many times and reports the median,
  the tails, and the scheduler's context switches from `getrusage`; `environment()` records
  threads, pinning (Linux), load and library versions for a receipt.
- `Settlement(precision=)` on the torch backend: float32 on CUDA when speed matters more than
  the receipt (read out on the cpu backend), float64 where the device has it.
- `cadence.stream`: owned state as a clamp. `stateful(...)` builds a windowed net with a range
  of context owners, one per hidden owner, wired densely into the hidden owners; `Echo` keeps
  a leaky trace of the hidden owners' equilibria across a batch of streams and writes it into
  the context clamp, so each state reverberates and fades. A test learns to name the symbol
  seen one input ago from a window of one, which no window can.
- `SettledState.repair`: the total movement of the published activations during a settlement,
  one number per row, on every backend. It is the work the net did to get from where it was
  to rest: the surprise of an input, measured rather than inferred from the step count.
- `Learner(slots=)`: the output owners as equal groups, each its own softmax choice, all nudged
  together; `targets` takes one label per slot and `predict` returns one choice per slot. A
  whole utterance settles at once. Each slot's nudge carries `beta / slots`, so the contrast
  is the gradient of the mean loss over the slots and `eta` means the same at any count
  (a span of four at full beta drove the hidden owners into saturation).
- `cadence.constitution`: where a wiring comes from. `Constitution` (regions and projections as
  a few numbers each), `grow` (development into a wiring with named sets, deterministic in
  the seed), `mutate`, `evolve` (selection over constitutions under a fitness the caller
  supplies). The phase before learning; not a new primitive.
- CI is green again: the 0.5.0 modules are formatted and typed.
- Tests: 58.

## 0.5.0 (2026-09-10)

Learning from reward and a life for the seams, built for the paper "You don't need attention
after all" and measured on its gates.

- `ActorCritic` (`cadence.plasticity`): the three-factor rule. An eligibility trace at every
  seam of the free/nudged contrast for the action taken, a linear critic on named owners
  with its own trace, a dopamine owner broadcasting the temporal-difference error; the
  adaptive local step (`momentum`, `normalize`, bias-corrected); `dopamine_cap`,
  `dopamine_center`, `critic_normalize`; `bootstrap=` for time limits. `Population` for
  continuous actions as a bump code. Cart-pole: 500 on every seed with the threshold at 40k
  to 60k steps, 1,716 parameters, two warm settlement steps per decision at deployment.
- `DreamActorCritic` and `actor_critic_wiring` (`cadence.dream`): an actor and a critic in
  one net with a memory; imagine-and-feel action selection (candidates felt in the critic),
  the critic's dream toward a target read with the slow strengths, the actor imitating the
  action taken. `DiscreteCode` for discrete actions.
- `Seams` and `SleepConfig` (`cadence.structure`): fast and slow strengths, tags, sleep
  (consolidate, downscale, prune, sprout within a budget), conserved incoming strength.
- `cadence.fused`: a compiled dense settlement kernel and a fused three-factor step on the
  CPU backend when numba is installed (`pip install "cadence-net[fast]"`); identical
  arithmetic, checked against the NumPy loop to 2e-16; `CADENCE_FUSED=0` forces the loop.
- `Learner.apply` and `Learner.contrast_rows`; the contrast as a Gram matrix product read
  at the overlaps (forty times faster than the gather on dense wirings).
- Docs: `reward.md`, `life.md`; the capability table carries the gates.
- Tests: 50.

## 0.4.1 (2026-09-09)

- Docs only. The grey parrot rung was withdrawn from cadence-examples (its imitations did not
  reach the bar); `embodied.md` now works through cart-pole and the sign writer, `tasks.md`
  keeps the several-learners-in-one-net recipe without the parrot, and the examples table
  lists the nine rungs.

## 0.4.0 (2026-09-09)

- `Learner.save` / `Learner.load` (`cadence.save`, `cadence.load`): one-file checkpoints of a
  trained learner, wiring and parameters included, that load on any backend and keep learning.
- `LearnerConfig.decay`: a leak on the seams, so a net that never stops learning stays plastic.
- `LearnerConfig.momentum`: each seam steps on a running average of its own contrast.
- `Learner(trainable_owners=...)` beside `trainable_overlaps`: two learners can share one net,
  each moving and decaying only its own seams and owners; a frozen overlap never moves, not
  even through the seam tying.
- `Learner(tie_groups=...)` and `embedded(...)`: a shared embedding across window positions.
- `py.typed`: the package is typed; `mypy --strict` clean.
- Tests: 42 across every module, 96% line coverage; the torch kernel is checked against the
  CPU engine with nudges, weights, adaptation and trajectories.
- Docs: task recipes (tabular place codes, regression as a pattern, streams, few labels, several
  learners in one net), `embodied.md` for deployment.

## 0.3.0

- The free/nudged learning rule, layered wirings, the torch backend, receipts, the examples ladder.
