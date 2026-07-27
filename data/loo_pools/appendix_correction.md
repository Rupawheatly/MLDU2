# Appendix Correction — LOO Protocol (around line 1098 of `mldu_arxiv.tex`)

The current appendix passage describes pools $\mathcal{C}$ and $\mathcal{B}$
in a way that does not match the implementation in
`notebooks/02_pythia.ipynb`, `notebooks/03_gpt-2-medium.ipynb`, and
`notebooks/04_mistral-7b.ipynb`. Specifically:

1. **Pool $\mathcal{C}$ is not "same prefix, randomly resampled completion".**
   In code, `POOL_C[k]` is a hand-crafted *register-matched paraphrase* of the
   memorized text `POOL_M[k]`: it preserves register, length, and surface form
   but substitutes distinctive terms so that `log P/tok` drops below the
   non-memorization threshold (e.g. `POOL_C['mit_license']` begins
   "Permission is explicitly denied, under all conditions, …" rather than
   "Permission is hereby granted, …").
2. **The pure-distinguishability baseline uses a third per-sequence pool,
   not "a second set of sequences with matched prefix decoys".** In code this
   third pool is called `POOL_B` and is a *second*, more aggressively
   distinctive-term-swapped paraphrase per memorized key (e.g.
   `POOL_B['mit_license']` begins "Authorization is hereby withheld,
   universally and permanently, …"). Below we refer to it in the paper as
   $\mathcal{D}$ to avoid colliding with the existing symbol $\mathcal{B}$,
   which already denotes the 24-element neutral context pool elsewhere in
   the manuscript (e.g. lines 1164, 1298).
3. **The pure-distinguishability LOO is $\mathcal{C}$ vs $\mathcal{D}$
   across held-out sequences,** not "a second set of sequences that do not
   pass screening with matched prefix decoys". The two are similar in spirit
   — both compare two register-matched non-memorized text classes — but the
   prose as currently written misdescribes the construction.

Below is a drop-in replacement preserving section labels, LaTeX environments,
existing meanings of $\mathcal{M}$ and $\mathcal{B}$, and downstream
cross-references. Bibliography keys are unchanged. Only the
\textbf{Procedure} and \textbf{Null baselines} paragraphs are rewritten;
the surrounding text is preserved.

---

## Proposed replacement text

```latex
\paragraph{Procedure.}
For each model we construct two per-sequence pools of memorized and
non-memorized text, plus a neutral context pool used to inject within-class
variance, all indexed by the same set of string identifiers
(e.g.\ \texttt{mit\_license}, \texttt{apache\_license}, \dots):

\begin{itemize}
 \item \textbf{Memorized pool} $\mathcal{M}$: $N$ candidate sequences that
 pass per-model log-probability screening
 ($\log P/\text{tok} > \theta_M$). The set of identifiers passing
 screening defines $\mathcal{K}_\text{valid}$.
 \item \textbf{Control pool} $\mathcal{C}$: for each $k \in \mathcal{K}_\text{valid}$, a
 register-matched paraphrase $\mathcal{C}[k]$ of comparable length to
 $\mathcal{M}[k]$, constructed by substituting distinctive terms so that
 $\log P/\text{tok}(\mathcal{C}[k]) < \theta_C$ while preserving register
 and surface form.
 \item \textbf{Second control pool} $\mathcal{D}$: for each
 $k \in \mathcal{K}_\text{valid}$, a further-paraphrased variant $\mathcal{D}[k]$,
 again screened to satisfy $\log P/\text{tok}(\mathcal{D}[k]) < \theta_C$.
 $\mathcal{D}$ is disjoint from $\mathcal{C}$ in surface content and serves
 the pure-distinguishability null below. (In the released code this pool
 is named \texttt{POOL\_B}.)
 \item \textbf{Neutral context pool} $\mathcal{B}$: 24 short neutral
 prefixes (e.g.\ ``Here is a passage:'') which supply within-class variance
 by being prepended to each pool entry at extraction time.
\end{itemize}

For each $k \in \mathcal{K}_\text{valid}$, each $b \in \mathcal{B}$, and each
class $X \in \{\mathcal{M}, \mathcal{C}, \mathcal{D}\}$ we extract the
last-token residual-stream activation of the concatenation $b \,\Vert\, X[k]$
at every depth $d$. This produces tensors of shape
\[
(|\mathcal{K}_\text{valid}|, |\mathcal{B}|, D, d_\text{model})
\]
for each of the three classes.

\paragraph{LOO classifier.}
For each depth $d$ and each held-out identifier
$k_j \in \mathcal{K}_\text{valid}$, we fit a logistic regression probe on all
$(k, b)$ activations for $k \in \mathcal{K}_\text{valid} \setminus \{k_j\}$
(label $1$ for the $\mathcal{M}$ activations and label $0$ for the
$\mathcal{C}$ activations), then evaluate on the $(k_j, b)$ activations for
$b \in \mathcal{B}$. We report per-sequence accuracy and the mean across
held-out identifiers as the \emph{true LOO} accuracy at depth $d$.

\paragraph{Null baselines.}
\begin{itemize}
 \item \textbf{Shuffled LOO.} Identical procedure, but training labels are
 randomly permuted within each fold. Expected value $0.5$ under the null
 hypothesis of no linear separability; we report the empirical mean over
 five random seeds.
 \item \textbf{Pure-distinguishability LOO.} Identical procedure, but with
 $\mathcal{C}$ as the positive class and $\mathcal{D}$ as the negative class
 (both screened as non-memorized). This measures how much of the true LOO
 score is attributable to cross-sequence string-separability structure
 shared among register-matched non-memorized variants, in the absence of
 memorization.
\end{itemize}

The memorization-specific gap is defined as
\[
\Delta_\text{mem}(d) = \text{true LOO}(d) - \text{pure LOO}(d).
\]
A positive gap indicates cross-sequence structure specific to memorization
rather than to string features shared across register-matched non-memorized
variants.

Per-model thresholds used in the paper are
$(\theta_M, \theta_C) = (-2.0, -3.0)$ for Pythia-70M,
$(-1.5, -2.5)$ for GPT-2 Medium, and the Mistral-7B pool is pre-screened
from \texttt{E1\_results.json} with $\mathcal{D}$ additionally re-screened
against $\log P/\text{tok} < -2.5$. See the released
\texttt{loo\_pools/} folder for the verbatim text of each pool.
```

---

## Why this rewrite is faithful

- The headline numbers in the paper (`true LOO`, `pure LOO`, `\Delta_mem`)
  are unchanged — only the description of how the non-memorized pools are
  constructed and how the pure baseline is computed is corrected.
- The new wording matches the actual code: pool extraction in
  `02_pythia.ipynb` cell 133 (and the equivalent extraction cell in
  `03_gpt-2-medium.ipynb`), the C-vs-D / `ctl`-vs-`B` substitution in
  `02_pythia.ipynb` cell 134, and the Mistral implementation in
  `04_mistral-7b.ipynb` cells 16 and 21.
- The symbol $\mathcal{B}$ retains its existing meaning (24-element neutral
  context pool), so cross-references at lines 1164 and 1298 of
  `mldu_arxiv.tex` continue to resolve correctly.
- The terminology `valid_keys` (in code) appears as $\mathcal{K}_\text{valid}$
  in the LaTeX and matches the contents of
  `loo_pools/<model>/valid_keys.json`.
- The replacement does not change any LaTeX label, ref, citation key,
  or environment.

## Optional companion sentence for the main text

If the paper body currently summarizes the controls in passing — for
example as "a matched-prefix control and a neutral-prefix baseline" — that
phrase can be tightened to "two register-matched non-memorized controls per
sequence ($\mathcal{C}$ and $\mathcal{D}$), with the pure-distinguishability
baseline computed between them; see Appendix~\ref{app:loo_protocol} for the
full construction."
