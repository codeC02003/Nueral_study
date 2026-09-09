# Start here — the whole project in plain words

Written for someone meeting all of this for the first time. Every technical word is
explained the first time it appears. No prior knowledge assumed beyond "computers
can learn things from data."

---

## 1. The one-sentence version

We built two neural networks. The first one learned to write Python code. The second
one's only job was to **find the parts of the first one that actually matter** — and it had to figure
it out by itself, with nobody telling it what to look for.

We call them **A** (the subject) and **B** (the anatomist).

> **neural network** — a big pile of numbers ("weights") arranged in layers, which
> you adjust bit by bit until the whole pile does something useful. Nobody programs
> the behaviour directly; it emerges from adjusting the numbers.

---

## 2. Why anyone would want this

Here is the uncomfortable fact about modern AI: **we build these systems but we
cannot read them.**

When A writes `def connection(self):`, no human — including the person who trained
it — can point at which numbers inside it produced that and explain why. It's not
that the code is hidden. All 820,000 numbers are right there. It's that a pile of
numbers doing something clever is not the same as an explanation.

That's not a curiosity. If you can't tell why a model does what it does, you can't
tell when it's about to do something bad, and you can't fix it properly when it
does.

The field that tries to solve this is called **mechanistic interpretability**.

> **mechanistic interpretability** — figuring out the actual internal machinery of a
> trained network: not "it's 94% accurate" but "*this* part does *this* job, and
> here's the proof."

Almost all of that work is done by humans, slowly, one model at a time. **This
project asks whether a second network can do it instead** — and, crucially, whether
it can do it *efficiently*, choosing what to investigate rather than checking
everything.

---

## 3. Network A — the thing being studied

A is a small **transformer** trained on 9.2 million characters of real Python code
from your computer's own Python installation.

> **transformer** — the architecture behind ChatGPT, Claude, and essentially every
> modern language model. Its key trick is **attention**: when processing one piece
> of text, it can look back at earlier pieces and pull in what it needs.

> **attention head** — one "looker." A transformer has many, and each learns to look
> for something different. One might look at the previous word; another might look
> for a matching bracket. Ours has 16 (4 layers × 4 heads).

A reads **one character at a time** and predicts the next one. That's it. That's the
whole training objective.

After 20 hours it went from this:

```
step    250   def y areewaretin u)   dele    suob.e()
```

to this:

```
step 56,500   def connection(self):    return self._internal_connec
```

Every identifier spelled correctly. Quotes and parentheses closed. A method body
indented, then dedented. It is still nonsense *semantically* — it doesn't know what
a connection is — but it learned the **shape** of Python from nothing but "guess the
next character."

**A is deliberately small.** 820,000 numbers, versus hundreds of billions in a
frontier model. That's not a limitation we tolerated — it's the point. A model small
enough to test exhaustively is a model where you can *check* whether B is right.

---

## 4. The discovery that decided everything about B

The obvious plan is: A has neurons, so B should read A's neurons and say what each
one means.

> **neuron** — one number inside the network that gets computed and passed along.
> Think of it as a single dial that goes up and down as the network processes text.

**That plan is broken**, and we proved it before building B. We made a tiny test:
20 pieces of information forced through a bottleneck of only 5 neurons. Information
theory says you can't fit 20 independent things into 5. But the network did it
anyway — by storing them as *overlapping directions*, accepting a bit of blur as the
price.

```
neuron 3 responded to items 0,1,2,5,6,7,8,9,10,11,12,14,19  — thirteen unrelated things
```

> **superposition** — a network storing more concepts than it has neurons, by
> letting them overlap. Real consequence: a single neuron in a real model fires for
> legal language, the letter Q, *and* DNA sequences. Asking "what does neuron 3
> mean?" has no clean answer.

> **direction** — instead of "neuron 3," the meaningful unit is a *combination* of
> neurons, like "0.3 of neuron 1 plus 0.7 of neuron 4." There are far more useful
> combinations than there are neurons.

So B could not be a neuron-reader. It had to be a **direction-finder**.

---

## 5. Network B — the anatomist

B is a **sparse autoencoder**.

> **autoencoder** — a network that takes something in, squeezes it, and tries to
> rebuild the original. If it rebuilds well, the squeezed version captured what
> mattered.

> **sparse** — only a few parts are allowed to be active at once. Ours has 1,024
> "features" but only about 15 may fire for any given character.

The trick: give B *more* slots than A has neurons (1,024 vs 128), but force it to
use only a handful at a time. Since superposition is compression *into* a small
space, forcing a large sparse space runs that compression backwards.

**B's entire training signal is:** rebuild A's internals, using as few features as
possible. That's all. No labels, no human hints, no "look for the code-comment
detector."

And it found things. Unprompted, B discovered features that fire on:

- exception class names mid-word (`AttributeE`, `TypeE`)
- the position right before the word `None`
- assignment operators
- `self` as a function's first parameter
- runs of repeated characters

Nobody wrote any of that down anywhere.

---

## 6. The trap that ruins most work like this

B finding a feature that lights up near `None` sounds impressive. **It proves
nothing.**

> **correlation vs causation** — the difference between "these two things happen
> together" and "one causes the other." B's finding is the first kind. A model can
> carry information around and never actually *use* it.

This is the single most common way interpretability results turn out to be wrong.
So we tested it properly.

**The test:** delete a feature from A and see if A breaks.

> **ablation** — deliberately damaging one part of a system to find out what it was
> for. Borrowed from neuroscience, where it's called a lesion study. Vesalius didn't
> learn anatomy by looking. He learned it by cutting.

**The control, which is the entire result:** deleting *anything* hurts a network. So
we also deleted **random directions of the same size**, on the same text, and
compared.

```
B's features         3.50x more damage than matched random directions
damage concentration 67% of a feature's damage on ONE character
                     (random directions: 8%)
```

> **control** — a comparison that tells you whether your result means anything. Without
> it, "I removed it and the model got worse" is not evidence, because that's true of
> almost any change.

And the specific findings held up:

| feature | what breaks when you remove it |
|---|---|
| f515 | A can no longer predict the `r` in `return` |
| f384 | A can no longer finish the word `None` |
| f336 | A can no longer predict the `.` after `self` |

That is **causal**, not correlational. Not "this information is present" but "remove
it and A stops working."

---

## 7. From "B found parts" to "B understands parts"

Finding a part isn't understanding it. So we raised the bar: **B has to predict, in
advance, how much damage removing something will cause.**

> **calibration** — whether your confidence matches reality. If B says "removing this
> costs 0.010" and reality says 0.011, B is calibrated. If B says 0.002 and reality
> says 0.011, B ranks things correctly but doesn't actually understand them.

This is much harder than ranking. A shortcut can guess *which* parts matter.
Predicting *how much* — including correctly predicting **zero** for something A
carries but never uses — means generalising precise counterfactual numbers to
features and contexts never tested.

> **counterfactual** — what *would* have happened under a change you didn't make.
> "How much worse would A get if this part were removed?"

One honest caution: doing this well is strong evidence that B has learned something
real about A. It is **not** proof that B represents A's algorithm — a predictor can
exploit statistical regularities without any correspondence to the underlying
machinery. Telling those apart is exactly what section 11 describes us failing to
test.

**Our first attempt failed**, and the failure was informative. B was asked to predict
one number per feature — its average importance. But we measured this:

```
96.3% of a feature's effect happens in 1% of the places it appears
its effect where it fires is ~3,000x its effect elsewhere
```

Importance isn't a property of a feature. It's a property of a **feature in a
context**. `f515` matters enormously when the text is heading toward `return`, and
not at all inside a comment. Asking B for the average was asking the wrong question.

Once we fixed the question, B's predictions became essentially unbiased on features
**it had never been tested on**.

---

## 8. The actual new idea: B chooses what to investigate

Here's the part that makes this more than a standard interpretability study.

A scientist doesn't test everything. A scientist **decides what to test next**. So we
gave B a budget of experiments and let it pick.

> **active learning** — letting a model choose which examples it wants labelled,
> instead of feeding it random ones. Standard in machine learning; unusual in
> interpretability.

Then we raced it against a control that picked **at random**, same budget, same
scoring.

```
                 causal mass found after 2,400 experiments
random selection                  0.68%
B choosing                       10.99%     <- 16x more
```

To match what B found in 2,400 experiments, random search would need roughly
**40,000**.

> **causal mass** — add up how much damage all the discoverable mechanisms cause.
> "Finding 11% of the mass" means B's chosen experiments captured 11% of everything
> there was to find.

We also compared against a **perfect oracle** — a cheater that knows all the answers
in advance and picks the very best experiments. B reached **30–40% of what the
cheater achieved**, while random managed 2.4%.

---

## 9. The contradiction, and what fixing it revealed

At first, making B good at *finding* important things made it bad at *predicting*
accurately. It became a prospector instead of a surveyor.

The obvious fix — mix in some random experiments — barely helped. Then we measured
why:

```
the random experiments were 20% of the count
but only 0.57% of the training signal
```

Because the important experiments have huge effects, they dominated the learning
even when outnumbered. **Mixing by count can't rebalance something weighted by
magnitude.**

The real fix was to change how B measures error — compress the huge values so no
single experiment can drown out the rest. That restored accuracy while keeping
almost all of the discovery advantage.

It was never a tradeoff between exploring and exploiting. It was one badly weighted
number.

---

## 10. Making sure we weren't fooling ourselves

The easiest way to get an impressive result is to keep adjusting until you get one.
So:

**We tested it on a second, independently trained model.** Same everything, different
random starting point. And we wrote down the pass/fail line **before** that model
existed, so we couldn't move the goalposts.

> **preregistration** — committing in writing to what counts as success, before you
> look at the answer. Standard in medicine. Rare in machine learning.

It passed. Numbers landed inside the band we'd committed to.

**And it falsified a prediction we'd made on the record** — we predicted a specific
weakness would reappear in the second model. It didn't. So that "finding" was a
quirk of one model, not a real property. We reported it as a failed prediction.

**We also put honest error bars on everything.**

> **confidence interval** — a range that says "the true value is probably in here."
> A result of "16×" means little; "16×, and we're 95% confident it's between 10× and
> 25×" means something.

Doing this properly required a **cluster bootstrap** — because measurements from the
same feature aren't independent, and treating them as if they were would make the
error bars far too small, which flatters the result.

The error bars forced us to **retract two of our own claims**, both now recorded in
the project history alongside the results.

---

## 11. What we could not prove

Here's the honest boundary.

We proved B finds structure that A genuinely **uses**. We did **not** prove that B's
description matches the algorithm A actually runs.

To test that you need a model whose true inner workings are already known — then you
check whether B rediscovers them. We built one. It failed for a reason worth
recording: the model **solved the task correctly but built a different circuit than
the textbook one**. Since the test depended on the textbook circuit being present,
there was nothing valid to score B against.

We had committed in advance to three attempts before abandoning. We used three, and
abandoned. **B was never run against it**, so the question is *untested*, not
*answered*.

That's frozen as its own milestone, labelled "instrument invalid," so nobody later
mistakes it for a result.

---

## 12. What's genuinely new here — honestly assessed

You should be suspicious of novelty claims, including this one. So here's the split.

**Not new:**

- Mechanistic interpretability is an established field with many researchers.
- Sparse autoencoders for finding features: established, notably at Anthropic.
- Ablation with controls: standard practice.
- Active learning: decades old.
- Using one network to study another: also not new (probing classifiers, distillation).

**Unusual, and the reason this project is interesting:**

1. **B chooses its own experiments.** Most interpretability is human-directed ("let's
   look at head 4") or exhaustive ("test everything"). Letting the analysing network
   allocate its own experimental budget, and measuring the speed-up, is not the
   standard setup.

2. **The efficiency question is quantified.** Not "we found interesting features" but
   "**16× faster than random, 95% CI 10–25×, replicated on a second model**." Most
   interpretability results are qualitative. This one has a number and an error bar.

3. **Zero human labels, end to end.** No point in the pipeline where a person says
   what to look for. The only training signal is A's own loss. That constraint was
   fixed at the start and never broken.

4. **The methodological standard.** Preregistration, a cluster bootstrap, published
   retractions, an abandoned benchmark reported as abandoned. This is normal in
   clinical trials and rare in ML. It's arguably the most transferable thing here.

**What I can't claim:** that nobody has done this. Verifying that needs a proper
literature search, which we did not do. What's defensible is that the *combination* —
self-directed experiment selection, a quantified efficiency result, no labels, and
this level of methodological hygiene — is unusual.

**And what's plainly not proven:** whether B understands A's actual algorithm.
That's the interesting question, and it's still open.

---

## 13. The honest summary

**What holds up:**

> B, given no human labels, finds causally important internal features and
> interventions roughly 10–25× more efficiently than random search, and predicts the
> effect of interventions it has never tried. Replicated on a second independently
> trained model.

We say **features and interventions**, not **mechanisms** — on purpose. A mechanism
would mean B's description matches the algorithm A actually runs, and that is the
thing section 11 says we could not test.

**What doesn't:**

> Whether B's picture of A corresponds to what A actually computes. Untested.

**And what the project is really an argument for:** that you can study these
questions rigorously — with controls, preregistration, error bars, and public
retractions — on a laptop, with a model small enough to actually check.

---

## Glossary

| word | plain meaning |
|---|---|
| **ablation** | breaking one part on purpose to see what it was for |
| **active learning** | letting the model choose what to investigate next |
| **attention** | a transformer looking back at earlier text to pull in what it needs |
| **attention head** | one "looker"; a model has many, each looking for something different |
| **autoencoder** | squeeze something down, rebuild it, see if anything was lost |
| **calibration** | does the model's confidence match reality |
| **causal** | proven by intervention, not just observed together |
| **cluster bootstrap** | error bars that account for measurements not being independent |
| **confidence interval** | the range the true answer is probably in |
| **control** | the comparison that tells you your result isn't trivial |
| **direction** | a combination of neurons; the real unit of meaning, not one neuron |
| **feature** | one thing a network has learned to detect |
| **loss** | how wrong the model is; training = making this smaller |
| **mechanistic interpretability** | working out a network's actual internal machinery |
| **neuron** | a single number inside the network |
| **oracle** | a hypothetical cheater who knows all the answers; used as a ceiling |
| **preregistration** | writing down what counts as success before you look |
| **sparse** | only a few things active at once |
| **superposition** | storing more concepts than there are neurons, by overlapping them |
| **transformer** | the architecture behind modern language models |

---

Next: [`01_OVERVIEW.md`](01_OVERVIEW.md) for the technical version, or
[`02_PROJECT_HISTORY.md`](02_PROJECT_HISTORY.md) for everything that happened,
failures included.
