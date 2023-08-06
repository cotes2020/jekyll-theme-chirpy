---
title: "A Closer Look at Quantum Measurements (Slides)"
description: "And how the Born rule can be generalized and derived from some principles of measurement"
categories: [quantum mechanics]
tags: [measurement problem, density operators, slides]
iframe:
    asset: true
    path: "pdf/posts/2023-08-03-closer-look-quantum-measurements/main.pdf"
    options: "view=FitH"
---

> “*What is particularly curious about quantum theory is that there can be actual physical effects arising from what philosophers refer to as counterfactuals - that is, things that might have happened, although they did not in fact happen.*”
> — Roger Penrose

## The measurement problem

Quantum measurement has remained a disputed aspect of quantum mechanics for about a century. One of the first models for quantum measurement was projection-valued measures (PVM), later generalized to positive operator-valued measures (POVM). These further fall under the study of quantum operations, which describe the kind of transitions quantum systems can undergo. For example, POVM is generalized by introducing Kraus operators, which further describe how quantum systems change under measurement.

In general, modelling quantum measurement warrants a study of how quantum systems interact with the measurement apparatus and an environment. This is the subject of quantum thermodynamics, which has revealed deep connections between the measurement problem and how open quantum systems (which are quantum systems interacting with an external quantum system which is typically a heat bath) work. In particular, open quantum systems with the Markovian property, together with their environment, form a completely unitary quantum system with no wavefunction collapse. The apparent wavefunction collapse of the open system, seemingly induced by its interaction with the environment, simply turns out to be an artefact of looking at a small part of a larger, unitary system.

There are even more general models in quantum thermodynamics, such as that of open quantum systems without the Markovian property, or those with non-unitary evolution (for example, the GKSL formalism). 

It can be safely said that the research arena for the problem of measurement in quantum mechanics is *vast*.

## A toy model

To begin playing around with the nature of quantum measurement, here's a study of a toy model with lots of simplifications compared to the kind of models described above. In particular, we consider mixed states and interpret wavefunction collapse to do the following:

- collapse the states comprising the mixed states to new mixed states resembling ensembles of eigenstates of the measurable observed

- not change the density operator of the mixed state in the above transition!

It turns out that such notions automatically give rise to the Born rule! As an alternative approach, we also model transition of mixed states to mixed states and construct a simplified picture of quantum operation. We can then apply this to transitions from mixed states to eigenstates of observables, to derive the standard Born rule.

Here are some slides elaborating on this topic:

{% include iframe.html %}

## Caveats

The above model, being simplified, has some caveats:

- Upon closer inspection, we find that the probability that pure states transition to themselves is $1$, and yet, there is a non-zero probability of a pure state to transition to some other state. 

    This is an artefact of not fully considering the space of states the transition pertains to. It turns out that the notion of transition we construct is fairly general in that now, we should consider all possible ways a transition can occur (in the Hilbert space of states) as opposed to via the evolution of some particular states arbitrarily chosen for mixed states. In fact, given the initial and final states between a transition, we should sum over *all* states contributing to such a transition. Consequently, a natural formalization of the toy model's approach is the path integral formulation of quantum mechanics (interestingly, computing probabilities of transition between mixed states involves traces of products of their densities, which can be represented by tensor networks rudimentarily resembling Feynman diagrams; this will likely be explored in a future post).

    When dealing with notions such as the above, we should also be careful about how we interpret probabilities, typically using measure-theoretic notions. It turns out that the solution to the problem of transition probabilities adding up to more than $1$ (since a state can transition to other states and seemingly *must* transition to itself too) is to reconsider the meaning of the probability function used — it is, in fact, a probability *density* to be integrated with an appropriate measure.

    Therefore, in the slides, while we dealt with traces and such linear algebraic constructs accurately, the probabilities they were assigned to were not given much attention.

- The density operator for a mixed state involves classical uncertainity with respect to the ensemble of states. On the other hand, the density operator for a mixed state after measurement involves an ensemble of eigenstates, and the kind of uncertainty here is purely quantum and has no classical analogue.

    Therefore, equating the density operator before and after measurement, as we did, is generally not valid (for example, in Stern-Gerlach like experiments, where mixed states have discrete, but classical probability distributions of outcomes; while states in quantum superposition always end up in the same projected states determined by the orientation of the measurement apparatus — and the probabilities associated with these events do not look classical at all).

## Conclusion

There are likely many more caveats to the simplified model studied. What, then, *does* it describe? I'd say it provides insight on some important aspects of quantum mechanics that we would encounter in some more general situations, such as processes involving transition amplitudes, transition symmetry, models of wavefunction collapse for parts of unitary systems which yield the Born rule and so forth. After all, we do derive transition amplitudes, their symmetry, the Born rule from a simple form of wavefunction collapse, etc. 

This means that these things, usually involving more general and consistent approaches than in the slides, do have analogues in the simplified world of Copenhagen-type interpretations. And, we must remember that these analogues don't all fit into a consistent system, which isn't surprising as we are not using a constructive approach but instead, making some interesting observations that individually parallel complicated measurement-related phenomena.

Above all, the purpose of this little project was to have fun playing around with measurement and hopefully, you will too! :)