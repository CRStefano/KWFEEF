# The Mind as an Information Filter

## The idea

It all started with a research project I undertook between my first and second years of studying philosophy at the University of Turin. I had become fascinated by the philosophy of mind and had started reading a lot about it.
Now, years later, I wanted to take it seriously. So I’d say let’s start from a fact everyone half-knows but rarely takes seriously: **information costs energy.** A brain is metabolically expensive. Every bit a nervous system acquires, holds, and refreshes is paid for in ATP, and that energy could have gone to staying alive instead.

So here is the question this project is about. If knowing the world costs energy, and that energy is subtracted from survival, **how much of the world should a living thing actually bother to represent?**

Contrary to what you might think (and many people do think this), the answer is *not* "as much as possible." That is what information theory would tell you if information were free. It isn't. Period. The right answer is a trade-off, and this repo works it out, formally, then in code, then in plain language.

The punchline, which I did not expect when I started: the optimal agent deliberately **throws information away**, sitting *below* the point where it would have everything it needs. Ignorance turns out to be a thermodynamic achievement, not a cognitive deficit.

---

## What's in here

Three things, for three kinds of reader:

- **`Program/`** — the model itself: a small foraging agent, the exhaustive optimiser, and an interactive dashboard you can open in a browser. Everything is reproducible from the scripts. I like things to be simple and intuitive, so you can launch the entire program simply by opening a .bat file (called “avvia_programma.bat”) - Everything is very neat; I hope you like it. **(Give me a star, give me a star, give me a star)** -

---

## The model in one breath

A creature forages on a 3×3 grid. It cannot perceive the world in full detail for free; instead it picks a **perceptual policy**,  a way of grouping the world into coarse categories (e.g. "food on the left" instead of "food in cell (0,2)"). There are exactly 203 such policies, and the program tries all of them. For each, it measures two quantities in the *same physical unit* (bits of free energy): how much **information** the policy keeps, and how much **viability** (distance from the death-equilibrium) it buys. The S-FEEF rule then picks the policy that maximises viability *net of the metabolic cost of information.*

---

## What it finds

Four results, stated honestly.

**1. Less is more.** The optimal agent keeps ~42% of the available information and still secures ~93% of the achievable viability. Past a certain resolution, the *Semantic Saturation Threshold*, more detail simply doesn't pay.

**2. More information can hurt, but only for a bounded agent.** Some coarse-grainings are *actively misleading*: merge two opposite places into one category, head for the average, and you walk into the empty middle. So viability is *non-monotone* in information. I want to be precise about this, because it's the most easily over-sold result: it survives a myopically Bayes-optimal action rule (so it is **not** a trivial artefact of going to the centroid), but it **disappears** if the agent commits to a concrete hypothesis or hedges. The honest reading is that "more can hurt" is a property of *bounded perception–action coupling* — the regime real organisms live in — not a universal theorem about information.

**3. The dark room is a sharp threshold, not a paradox.** Raise the per-bit cost α and the optimum jumps discontinuously from "engage with the world" to "process nothing" (the dark room) at a computable critical value. No extra drives or preferences required — and because viability and information now share units, that threshold sits a fixed multiple above Landauer's physical floor.

**4. The body gates the value of the mind.** Vary only the agent's reach and three regimes appear: too weak to use information (it gives up), an intermediate band where information is precious and selective, and *so capable that information becomes harmful*, a creature whose body solves the problem has no use for a mind that represents it. Rational inattention is bounded by morphology, non-monotonically.

The same marginal-return logic then extends, in the paper, to evolutionary life-history: semelparity (the salmon that breeds once and dies), the *emergence* of senescence as an optimum, and bet-hedging under environmental noise.

---

## The move I'm proudest of

Early versions postulated viability with a hand-tuned penalty parameter. The fix that made everything click is to **define viability as the non-equilibrium free energy** of the agent's distributio, its Kullback–Leibler distance from the Boltzmann (death) equilibrium. This is not cosmetic:

- the free parameter disappears, the "cost of death" is now just the free-energy gap, already fixed by the substrate;
- viability and information end up in the *same unit*, so the metabolic multiplier α becomes a genuine exchange rate with a **Landauer floor** (α ≥ 1);
- and it lets me **derive** the S-FEEF criterion from Millidge's *Free Energy of the Future* under four explicitly stated approximations (Appendix B), turning a suggestive analogy with Friston's Free Energy Principle into a reduction.

It also commits the framework to a stance I'm happy to defend: **information is physical** (Some people might be shocked, while others will say, “That was obvious,” but I don't think it's a given. I discuss this in more detail in the paper (please read the paper, thank you))

---

Is everything perfect? I wish. I know full well that  It's a **toy model**, 108 states. It demonstrates mechanisms; it does not prove universal laws at scale. Several headline results are **contingent on the action rule**, as said above. I treat that as a finding, not as something to hide. here is **no empirical validation yet** (the key word here is “yet”). The paper proposes a concrete paradigm (titrate metabolic load, measure perceptual grain); nobody has run it. The link to the Free Energy Principle is a derivation **under stated approximations**, not yet ((again: the key word is “yet”)) a parameter-free identity.

If those bother you, good, they bother me too, and they are the to-do list....If you'd like to give me a hand...

---

## Running the program

You need **Python 3.11+**.

**The easy way (interactive dashboard):** open `Program/` and double-click `Run_program.bat`. It installs its own dependencies, starts a local server, and opens your browser at `http://localhost:5000`. Drag the sliders (metabolic cost α, horizon τ, reach, …) and watch the optimum move. (I know, some things are still written in Italian. Unfortunately, I still can't think in English. Most of it, though, has been translated, and I solemnly promise that I'll translate everything.)

**Reproducing the paper's numbers** (from inside `Programma/`):

```bash
python run_experiments.py          # core results          -> results_v2.txt
python life_history_analysis.py    # iteroparity/semelparity transition
python age_structured_analysis.py  # emergent senescence
python eco_analysis.py             # density dependence + bet-hedging
python parameter_sweep.py          # robustness sweep
python robustness_analysis.py      # SST robustness + local search for the optimum
```

Pre-generated outputs are in `Programma/risultati/` for reference.

---

## Repository layout

```
.
├── README.md                        # this file
└── Program/
    ├── Run_program.bat              # one-click launcher
    ├── main.py                      # interactive dashboard
    ├── run_experiments.py
    ├── core/                        # the model
    │   ├── model.py                 #   S-FEEF foraging model (viability = free energy)
    │   ├── information.py           #   entropy, mutual information, SST
    │   ├── lifehistory.py           #   reproductive allocation
    │   ├── age_structured.py        #   age structure & senescence
    │   └── eco_dynamics.py          #   density dependence & bet-hedging
    ├── *.py                         # analysis scripts (reproduce figures/tables)
    └── results/                     # pre-generated outputs
```

---

## Standing on

This work is a synthesis, and it owes its bones to others: Kolchinsky & Wolpert on semantic information and the thermodynamics of agency; Friston's Free Energy Principle and Millidge, Tschantz & Buckley on the Free Energy of the Future; Landauer on the physicality of information; Rovelli on meaning as information plus evolution; and the long tradition of bounded and ecological rationality (Simon, Gigerenzer, Wheeler). Full references are in the paper (I know, sorry, I have a soft spot for Kolchinsky; I mention it in every project I work on.).

---


## Contact

[Stefano Coelati Rama](https://stefanocoelatirama.com)


## License

- Code (`Programma/`): **MIT** — see [`Programma/LICENSE`](Programma/LICENSE).
- Paper and text (`Paper/`, this README, the explainer): **CC BY 4.0** — see [`Paper/LICENSE`](Paper/LICENSE).

In short: use, modify, and build on the code freely; reuse the writing freely too, just give credit.


~ be so good that they can't ignore you.
