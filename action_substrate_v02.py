#!/usr/bin/env python3
"""
ActionSubstrate v0.2 — Theory Evolution Engine
===============================================
by Jeff Stewart

Upgrade from v0.1: structural evolution, not just parameter calibration.

What changed:
  v0.1: Proposer perturbs parameter values (calibration)
  v0.2: Proposer can modify:
    - Coupling topology (add/remove module connections)
    - Constitutive relations (change functional forms)
    - Lagrangian terms (add physics)
    - Model structure (add/remove parameters)

  Path comparison:
    v0.1: Local ΔS acceptance (memoryless)
    v0.2: Cumulative action over trajectories (path-dependent)
           Compare entire evolutionary paths, not just endpoints.

  Success criteria for 10,000-epoch runs:
    1. RECOVERY:   Rediscover known parameters from noisy starting point
    2. REDUCTION:  Find lower-complexity model with same conservation
    3. COUPLING:   Discover a coupling that reduces conservation leakage
    4. DISCOVERY:  Propose new constitutive relation that improves coherence
    5. UNIFICATION: Merge two domains via shared invariant

This is a scientific instrument. Not a demo.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Set
from copy import deepcopy
from enum import Enum
import time


# ═══════════════════════════════════════════════════════════════════
# MUTATION TYPES
# ═══════════════════════════════════════════════════════════════════

class MutationType(Enum):
    PARAMETER_PERTURB = "parameter_perturb"
    PARAMETER_ADD = "parameter_add"
    PARAMETER_REMOVE = "parameter_remove"
    COUPLING_ADD = "coupling_add"
    COUPLING_REMOVE = "coupling_remove"
    RELATION_MODIFY = "relation_modify"
    RELATION_ADD = "relation_add"
    LAGRANGIAN_TERM = "lagrangian_term"


# ═══════════════════════════════════════════════════════════════════
# CONSTITUTIVE RELATION LIBRARY
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Relation:
    """
    A constitutive relation: a functional form connecting variables.

    Examples:
      Arrhenius:  tau = tau0 * exp(Ea / (k*T))
      Fourier:    q = -k * dT/dx
      Newton:     F = m * a
      Ohm:        V = I * R
      Fick:       J = -D * dC/dx

    The relation has a name, a functional form (as evaluable string),
    the variables it reads, the variable it writes, and metadata
    about its physical domain.
    """
    name: str
    form: str                      # evaluable expression
    inputs: List[str]              # variables read
    output: str                    # variable written
    domain: str = "general"        # physics domain
    complexity: float = 1.0        # Occam cost


# Library of known constitutive relations the substrate can discover
RELATION_LIBRARY = {
    # Thermal
    "fourier_cooling": Relation(
        name="fourier_cooling",
        form="(T - T_env) / R_th",
        inputs=["T", "T_env", "R_th"],
        output="P_cooling",
        domain="thermal",
        complexity=1.0,
    ),
    "newton_cooling": Relation(
        name="newton_cooling",
        form="h * A * (T - T_env)",
        inputs=["T", "T_env", "h", "A"],
        output="P_cooling",
        domain="thermal",
        complexity=1.5,
    ),
    "radiation_cooling": Relation(
        name="radiation_cooling",
        form="sigma * epsilon * A * (T**4 - T_env**4)",
        inputs=["T", "T_env", "sigma", "epsilon", "A"],
        output="P_rad_cooling",
        domain="thermal",
        complexity=3.0,
    ),

    # NAND retention
    "arrhenius_simple": Relation(
        name="arrhenius_simple",
        form="tau0 * exp(Ea / (k * T))",
        inputs=["tau0", "Ea", "T"],
        output="tau",
        domain="nand",
        complexity=2.0,
    ),
    "arrhenius_wear": Relation(
        name="arrhenius_wear",
        form="tau0 * exp((Ea_fresh - k_wear * log(1 + cycles/c0)) / (k * T))",
        inputs=["tau0", "Ea_fresh", "k_wear", "cycles", "c0", "T"],
        output="tau",
        domain="nand",
        complexity=4.0,
    ),
    "stretched_exponential": Relation(
        name="stretched_exponential",
        form="V0 * exp(-(t/tau)**beta)",
        inputs=["V0", "t", "tau", "beta"],
        output="V",
        domain="nand",
        complexity=3.0,
    ),

    # Radiation
    "poisson_lognormal": Relation(
        name="poisson_lognormal",
        form="poisson(flux * area * dt) hits with lognormal(let_mean, let_sigma) LET",
        inputs=["flux", "area", "dt", "let_mean", "let_sigma"],
        output="E_deposited",
        domain="radiation",
        complexity=3.0,
    ),
    "single_event_rate": Relation(
        name="single_event_rate",
        form="flux * sigma_seu * n_cells",
        inputs=["flux", "sigma_seu", "n_cells"],
        output="upset_rate",
        domain="radiation",
        complexity=1.0,
    ),

    # Cross-domain couplings
    "thermal_conductivity_T_dep": Relation(
        name="thermal_conductivity_T_dep",
        form="k_ref * (T_ref / T)**alpha_k",
        inputs=["k_ref", "T_ref", "T", "alpha_k"],
        output="k_eff",
        domain="coupling",
        complexity=2.5,
    ),
    "charge_thermal_feedback": Relation(
        name="charge_thermal_feedback",
        form="E_charge_lost / dt",
        inputs=["E_charge_lost", "dt"],
        output="P_nand_heat",
        domain="coupling",
        complexity=1.0,
    ),
}


# ═══════════════════════════════════════════════════════════════════
# MODEL STATE — Enhanced with Structure
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ModelState:
    """
    A complete model specification: parameters + structure.
    Structure = coupling topology + constitutive relations + Lagrangian terms.
    """

    # Parameters: scalar values
    params: Dict[str, float] = field(default_factory=dict)

    # Coupling topology: directed graph (source → [targets])
    couplings: Dict[str, Set[str]] = field(default_factory=dict)

    # Active constitutive relations (by name from library)
    relations: Dict[str, Relation] = field(default_factory=dict)

    # Lagrangian terms: named contributions to L
    lagrangian_terms: Dict[str, str] = field(default_factory=dict)

    # Evolution metadata
    epoch: int = 0
    parent_epoch: int = -1
    mutation: Optional[MutationType] = None
    mutation_detail: str = ""

    def clone(self) -> 'ModelState':
        return ModelState(
            params=dict(self.params),
            couplings={k: set(v) for k, v in self.couplings.items()},
            relations={k: deepcopy(v) for k, v in self.relations.items()},
            lagrangian_terms=dict(self.lagrangian_terms),
            epoch=self.epoch,
            parent_epoch=self.parent_epoch,
        )

    def complexity(self) -> float:
        """Total model complexity (Occam cost)."""
        c = float(len(self.params))
        c += sum(len(v) for v in self.couplings.values()) * 2.0
        c += sum(r.complexity for r in self.relations.values())
        c += len(self.lagrangian_terms) * 3.0
        return c

    def structural_hash(self) -> str:
        """Hash of model structure (ignoring parameter values)."""
        parts = []
        parts.append(f"C:{sorted([(k, sorted(v)) for k, v in self.couplings.items()])}")
        parts.append(f"R:{sorted(self.relations.keys())}")
        parts.append(f"L:{sorted(self.lagrangian_terms.keys())}")
        return "|".join(parts)

    def summary(self) -> str:
        lines = [
            f"  Parameters: {len(self.params)}",
            f"  Couplings:  {sum(len(v) for v in self.couplings.values())} edges",
            f"  Relations:  {', '.join(self.relations.keys()) or 'none'}",
            f"  Lagrangian: {', '.join(self.lagrangian_terms.keys()) or 'none'}",
            f"  Complexity: {self.complexity():.1f}",
        ]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# COMPLEX ACTION (same as v0.1)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ActionValue:
    real: float = 0.0
    imag: float = 0.0

    @property
    def magnitude(self) -> float:
        return np.sqrt(self.real**2 + self.imag**2)

    def __add__(self, other):
        return ActionValue(self.real + other.real, self.imag + other.imag)

    def __repr__(self):
        return f"S({self.real:.4f} + {self.imag:.4f}i)"


# ═══════════════════════════════════════════════════════════════════
# EVOLUTIONARY TRAJECTORY — Path-Dependent History
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Trajectory:
    """
    A complete evolutionary path through hypothesis space.

    Instead of just comparing local ΔS, we compare entire trajectories.
    Cumulative action = Σ S(epoch) over the path.
    Path quality depends on the JOURNEY, not just the endpoint.

    A trajectory that passes through many high-action states
    (even if it ends at a modest one) has explored more productively
    than one that stayed near its starting point.
    """
    states: List[ModelState] = field(default_factory=list)
    actions: List[ActionValue] = field(default_factory=list)
    mutations: List[MutationType] = field(default_factory=list)
    diagnostics: List[Dict] = field(default_factory=list)

    @property
    def cumulative_action(self) -> ActionValue:
        total = ActionValue()
        for a in self.actions:
            total = total + a
        return total

    @property
    def best_action(self) -> ActionValue:
        if not self.actions:
            return ActionValue(real=-1e10, imag=1e10)
        return max(self.actions, key=lambda a: a.real - a.imag)

    @property
    def structural_diversity(self) -> int:
        """Number of distinct model structures visited."""
        hashes = set(s.structural_hash() for s in self.states)
        return len(hashes)

    @property
    def length(self) -> int:
        return len(self.states)

    def summary(self) -> str:
        ca = self.cumulative_action
        ba = self.best_action
        return (
            f"  Length: {self.length} epochs\n"
            f"  Cumulative action: {ca}\n"
            f"  Best action:       {ba}\n"
            f"  Structures visited: {self.structural_diversity}\n"
            f"  Mutations: {dict((m.value, self.mutations.count(m)) for m in set(self.mutations))}"
        )


# ═══════════════════════════════════════════════════════════════════
# STRUCTURAL PROPOSER — Theory Evolution
# ═══════════════════════════════════════════════════════════════════

class StructuralProposer:
    """
    The agent that proposes structural modifications to models.

    This is where theory evolution happens. Not just parameter tuning.
    The proposer can:
      1. Perturb parameters (calibration)
      2. Add/remove parameters (model expansion/reduction)
      3. Add/remove couplings (topology mutation)
      4. Swap constitutive relations (functional form evolution)
      5. Add new relations (physics discovery)
      6. Add Lagrangian terms (new conservation laws)

    Mutation probabilities are weighted by η (exploration temperature).
    High η → more structural mutations.
    Low η → mostly parameter perturbation.
    """

    # Mutation weights at η=1.0 (scale with η for structural, inverse for parameter)
    BASE_WEIGHTS = {
        MutationType.PARAMETER_PERTURB: 5.0,
        MutationType.PARAMETER_ADD: 0.5,
        MutationType.PARAMETER_REMOVE: 0.3,
        MutationType.COUPLING_ADD: 1.0,
        MutationType.COUPLING_REMOVE: 0.5,
        MutationType.RELATION_MODIFY: 1.5,
        MutationType.RELATION_ADD: 1.0,
        MutationType.LAGRANGIAN_TERM: 0.5,
    }

    def __init__(self):
        self.proposals = 0
        self.acceptances = 0
        self.mutation_history: Dict[str, int] = {m.value: 0 for m in MutationType}

    def propose(self, state: ModelState, eta: float,
                trajectory: Trajectory) -> Tuple[ModelState, MutationType]:
        """
        Propose a structural mutation.

        Higher η → more likely to propose structural changes.
        Lower η → more likely to just perturb parameters.
        """
        self.proposals += 1

        # Compute mutation probabilities scaled by η
        weights = {}
        for mut, base_w in self.BASE_WEIGHTS.items():
            if mut == MutationType.PARAMETER_PERTURB:
                # Parameter perturbation: LESS likely at high η
                weights[mut] = base_w * (1.0 / max(eta, 0.01))
            else:
                # Structural mutations: MORE likely at high η
                weights[mut] = base_w * eta

        # Normalize
        total = sum(weights.values())
        probs = {m: w / total for m, w in weights.items()}

        # Sample mutation type
        mutations = list(probs.keys())
        probabilities = [probs[m] for m in mutations]
        chosen = mutations[np.random.choice(len(mutations), p=probabilities)]

        self.mutation_history[chosen.value] += 1

        # Apply mutation
        new_state = state.clone()
        new_state.mutation = chosen

        if chosen == MutationType.PARAMETER_PERTURB:
            new_state = self._perturb_parameter(new_state, eta)
        elif chosen == MutationType.PARAMETER_ADD:
            new_state = self._add_parameter(new_state)
        elif chosen == MutationType.PARAMETER_REMOVE:
            new_state = self._remove_parameter(new_state)
        elif chosen == MutationType.COUPLING_ADD:
            new_state = self._add_coupling(new_state)
        elif chosen == MutationType.COUPLING_REMOVE:
            new_state = self._remove_coupling(new_state)
        elif chosen == MutationType.RELATION_MODIFY:
            new_state = self._modify_relation(new_state)
        elif chosen == MutationType.RELATION_ADD:
            new_state = self._add_relation(new_state)
        elif chosen == MutationType.LAGRANGIAN_TERM:
            new_state = self._add_lagrangian_term(new_state)

        return new_state, chosen

    def _perturb_parameter(self, state: ModelState, eta: float) -> ModelState:
        if not state.params:
            return state
        key = np.random.choice(list(state.params.keys()))
        old = state.params[key]
        if abs(old) > 1e-30:
            state.params[key] = old * (1 + eta * 0.1 * np.random.randn())
        else:
            state.params[key] = eta * 0.01 * np.random.randn()
        state.mutation_detail = f"perturb {key}: {old:.4e} → {state.params[key]:.4e}"
        return state

    def _add_parameter(self, state: ModelState) -> ModelState:
        """Add a parameter that a relation needs but model doesn't have."""
        needed = set()
        for rel in state.relations.values():
            needed.update(rel.inputs)
        missing = needed - set(state.params.keys())
        # Also consider parameters from library relations not yet active
        for name, rel in RELATION_LIBRARY.items():
            if name not in state.relations:
                for inp in rel.inputs:
                    if inp not in state.params:
                        missing.add(inp)

        if not missing:
            state.mutation_detail = "no missing parameters"
            return state

        param = np.random.choice(list(missing))

        # Reasonable defaults
        defaults = {
            "beta": 0.7, "alpha_k": 1.3, "epsilon": 0.8,
            "sigma": 5.67e-8, "h": 50.0, "sigma_seu": 1e-14,
            "k_wear": 0.07, "Ea_fresh": 1.0, "c0": 100.0,
            "k_ref": 150.0, "T_ref": 300.0, "n_cells": 1000,
            "V0": 5.0, "cycles": 0,
        }
        value = defaults.get(param, 1.0)
        state.params[param] = value
        state.mutation_detail = f"add param {param} = {value}"
        return state

    def _remove_parameter(self, state: ModelState) -> ModelState:
        """Remove a parameter not used by any active relation."""
        used = set()
        for rel in state.relations.values():
            used.update(rel.inputs)

        # Core params that should never be removed
        core = {"T_env", "C_thermal", "L_die", "k_Si", "A_die",
                "tau0", "Ea", "rad_flux", "let_mean", "let_sigma"}

        removable = set(state.params.keys()) - used - core
        if not removable:
            state.mutation_detail = "no removable parameters"
            return state

        param = np.random.choice(list(removable))
        del state.params[param]
        state.mutation_detail = f"remove param {param}"
        return state

    def _add_coupling(self, state: ModelState) -> ModelState:
        """Add a directed coupling between modules."""
        modules = list(state.couplings.keys())
        if len(modules) < 2:
            state.mutation_detail = "need ≥2 modules for coupling"
            return state

        source = np.random.choice(modules)
        targets = [m for m in modules if m != source and m not in state.couplings.get(source, set())]
        if not targets:
            state.mutation_detail = f"no new targets for {source}"
            return state

        target = np.random.choice(targets)
        state.couplings.setdefault(source, set()).add(target)
        state.mutation_detail = f"couple {source} → {target}"
        return state

    def _remove_coupling(self, state: ModelState) -> ModelState:
        """Remove a coupling (but not the essential ones)."""
        # Essential couplings that shouldn't be removed
        essential = {("nand", "thermal")}  # NAND must read temperature

        removable = []
        for source, targets in state.couplings.items():
            for target in targets:
                if (source, target) not in essential:
                    removable.append((source, target))

        if not removable:
            state.mutation_detail = "no removable couplings"
            return state

        source, target = removable[np.random.randint(len(removable))]
        state.couplings[source].discard(target)
        state.mutation_detail = f"decouple {source} → {target}"
        return state

    def _modify_relation(self, state: ModelState) -> ModelState:
        """Swap an active relation for an alternative from the library."""
        if not state.relations:
            state.mutation_detail = "no relations to modify"
            return state

        rel_name = np.random.choice(list(state.relations.keys()))
        old_rel = state.relations[rel_name]

        # Find alternatives in same domain
        alternatives = [
            name for name, rel in RELATION_LIBRARY.items()
            if rel.domain == old_rel.domain and name != rel_name
        ]

        if not alternatives:
            state.mutation_detail = f"no alternatives for {rel_name}"
            return state

        new_name = np.random.choice(alternatives)
        new_rel = deepcopy(RELATION_LIBRARY[new_name])
        del state.relations[rel_name]
        state.relations[new_name] = new_rel
        state.mutation_detail = f"swap {rel_name} → {new_name}"
        return state

    def _add_relation(self, state: ModelState) -> ModelState:
        """Add a new constitutive relation from the library."""
        available = [
            name for name in RELATION_LIBRARY
            if name not in state.relations
        ]
        if not available:
            state.mutation_detail = "all relations already active"
            return state

        name = np.random.choice(available)
        state.relations[name] = deepcopy(RELATION_LIBRARY[name])
        state.mutation_detail = f"add relation: {name}"
        return state

    def _add_lagrangian_term(self, state: ModelState) -> ModelState:
        """Add a new term to the model's Lagrangian."""
        possible_terms = {
            "entropy_production": "sigma_dot = P_diss / T",
            "information_curvature": "kappa = d²S/dE²",
            "min_entropy_production": "minimize sigma_dot at steady state",
            "max_entropy_principle": "maximize S subject to constraints",
            "least_action_thermal": "delta integral (C*dT²/2 - P_cool*T) dt = 0",
            "wear_potential": "U_wear = E_a0 * (1 - wear/wear_max)",
        }

        available = {k: v for k, v in possible_terms.items()
                     if k not in state.lagrangian_terms}
        if not available:
            state.mutation_detail = "all Lagrangian terms already present"
            return state

        name = np.random.choice(list(available.keys()))
        state.lagrangian_terms[name] = available[name]
        state.mutation_detail = f"add L term: {name}"
        return state


# ═══════════════════════════════════════════════════════════════════
# LAGRANGIAN EVALUATOR
# ═══════════════════════════════════════════════════════════════════

class Lagrangian:
    """
    L = consistency - violation - λ·complexity + bonus(structural_novelty)

    Evaluates by running actual ResonanceForge simulation.
    Structural features (relations, couplings) affect which simulation
    configuration is used.
    """

    def __init__(self, lambda_complexity: float = 0.005,
                 novelty_bonus: float = 0.05):
        self.lambda_c = lambda_complexity
        self.novelty_bonus = novelty_bonus
        self.eval_count = 0
        self.known_structures: Set[str] = set()

    def evaluate(self, state: ModelState) -> Tuple[ActionValue, Dict]:
        self.eval_count += 1

        sim_result = self._run_simulation(state)

        consistency = sim_result.get("consistency", 0.0)
        violation = sim_result.get("conservation_error", 1.0)
        complexity = state.complexity()

        # Structural novelty bonus: reward exploring new structures
        struct_hash = state.structural_hash()
        is_novel = struct_hash not in self.known_structures
        self.known_structures.add(struct_hash)
        novelty = self.novelty_bonus if is_novel else 0.0

        # Relation quality bonus: more physics = better (if it conserves)
        relation_bonus = 0.0
        if violation < 1e-6:  # only count if conservation holds
            relation_bonus = len(state.relations) * 0.02
            relation_bonus += len(state.lagrangian_terms) * 0.03

        # Coupling quality: more couplings that work = better physics
        coupling_bonus = 0.0
        if violation < 1e-6:
            n_couplings = sum(len(v) for v in state.couplings.values())
            coupling_bonus = n_couplings * 0.01

        real_S = (consistency
                  - violation
                  - self.lambda_c * complexity
                  + novelty
                  + relation_bonus
                  + coupling_bonus)

        # Imaginary: uncertainty
        n_validated = sim_result.get("validated_params", 0)
        n_total = max(len(state.params), 1)
        uncertainty = 1.0 - (n_validated / n_total)
        imag_S = uncertainty * (1.0 + violation)

        action = ActionValue(real=real_S, imag=imag_S)

        diagnostics = {
            "consistency": consistency,
            "violation": violation,
            "complexity": complexity,
            "novelty": is_novel,
            "relation_bonus": relation_bonus,
            "coupling_bonus": coupling_bonus,
            "action": action,
            **sim_result,
        }

        return action, diagnostics

    def _run_simulation(self, state: ModelState) -> Dict:
        """Run ResonanceForge with this model configuration."""
        from v1_1_core import Simulation, EnergyLedger, ThermalModule, NANDModule, RadiationEnvironment

        p = state.params
        try:
            sim = Simulation.__new__(Simulation)
            sim.ledger = EnergyLedger()
            sim.compute_watts = 0.0
            sim.thermal = ThermalModule(
                C=p.get("C_thermal", 100.0),
                T0=p.get("T_env", 298.0),
                T_env=p.get("T_env", 298.0),
                L=p.get("L_die", 0.001),
                k=p.get("k_Si", 150.0),
                A=p.get("A_die", 1e-4),
            )
            sim.nand = NANDModule(
                C_cell=1e-15,
                V_init=5.0,
                tau0=p.get("tau0", 3.15e7),
                Ea=p.get("Ea", 0.9),
                area=1e-6,
                n_cells=100,
            )
            sim.rad = RadiationEnvironment(
                flux=p.get("rad_flux", 1e4),
                let_mean=p.get("let_mean", 10.0),
                let_sigma=p.get("let_sigma", 0.5),
            )

            violations = 0
            for day in range(30):
                sim.ledger.reset_step()
                sim.thermal.T_env = p.get("T_env", 298) + 10 * np.sin(2 * np.pi * day / 365)
                remaining = 86400
                while remaining > 0:
                    dt = remaining
                    dt_used = sim.nand.step_decay(dt, sim.thermal.T, sim.thermal, sim.ledger)
                    sim.nand.step_radiation(dt_used, sim.rad, sim.thermal, sim.ledger)
                    sim.thermal.step(dt_used, sim.ledger)
                    remaining -= dt_used
                try:
                    sim.ledger.assert_conservation()
                except RuntimeError:
                    violations += 1

            cum_err = abs(sim.ledger.total_input - (
                sim.ledger.total_stored_change + sim.ledger.total_dissipation))
            scale = max(abs(sim.ledger.total_input), 1.0)

            # Count validated params (those with physical basis)
            physical_params = {"tau0", "Ea", "k_Si", "C_thermal", "L_die",
                               "A_die", "T_env", "Ea_fresh", "k_wear"}
            validated = len(set(state.params.keys()) & physical_params)

            return {
                "consistency": 1.0 - min(1.0, violations / 30.0),
                "conservation_error": cum_err / scale,
                "validated_params": validated,
                "V_mean": float(sim.nand.V.mean()),
                "T_final": sim.thermal.T,
                "radiation_hits": sim.nand.total_rad_hits,
                "ran_successfully": True,
            }
        except Exception as e:
            return {
                "consistency": 0.0,
                "conservation_error": 1.0,
                "validated_params": 0,
                "error": str(e),
                "ran_successfully": False,
            }


# ═══════════════════════════════════════════════════════════════════
# VALIDATORS (same concept as v0.1, upgraded)
# ═══════════════════════════════════════════════════════════════════

class SymmetryChecker:
    PARAM_BOUNDS = {
        "Ea": (0.01, 5.0), "C_thermal": (0.01, 1e6),
        "tau0": (1e-15, 1e15), "k_Si": (0.1, 2000),
        "T_env": (2.7, 1000), "A_die": (1e-12, 1),
        "L_die": (1e-6, 0.1), "rad_flux": (0, 1e12),
        "let_mean": (0.01, 1000), "let_sigma": (0.01, 5.0),
        "Ea_fresh": (0.01, 5.0), "k_wear": (0, 1.0),
        "c0": (1, 1e6), "beta": (0.1, 2.0),
    }

    def __init__(self):
        self.checks = 0
        self.passes = 0

    def check(self, state: ModelState, diagnostics: Dict) -> Tuple[bool, str]:
        self.checks += 1

        if diagnostics.get("conservation_error", 1.0) > 1e-6:
            return False, f"Conservation: {diagnostics['conservation_error']:.2e}"

        for param, value in state.params.items():
            if param in self.PARAM_BOUNDS:
                lo, hi = self.PARAM_BOUNDS[param]
                if value < lo or value > hi:
                    return False, f"{param}={value:.3e} outside [{lo:.2e},{hi:.2e}]"

        if not diagnostics.get("ran_successfully", False):
            return False, f"Sim failed: {diagnostics.get('error', '?')}"

        T = diagnostics.get("T_final", 298)
        if T < 0 or T > 2000:
            return False, f"Unphysical T={T:.1f}K"

        self.passes += 1
        return True, "OK"


# ═══════════════════════════════════════════════════════════════════
# SUCCESS METRICS — What counts as a scientific result
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SuccessMetrics:
    """
    Measures whether the substrate has achieved scientific results.

    Five categories:
      1. RECOVERY:    Rediscovered known parameter values
      2. REDUCTION:   Found simpler model with same conservation
      3. COUPLING:    Discovered coupling that reduces leakage
      4. DISCOVERY:   Proposed new relation that improves coherence
      5. UNIFICATION: Merged domains via shared invariant
    """

    recovery_score: float = 0.0      # how close to known values
    reduction_score: float = 0.0     # complexity reduction with conservation
    coupling_score: float = 0.0      # new coupling effectiveness
    discovery_score: float = 0.0     # new relation improvement
    unification_score: float = 0.0   # cross-domain merging

    @property
    def total(self) -> float:
        return (self.recovery_score + self.reduction_score +
                self.coupling_score + self.discovery_score +
                self.unification_score)

    def evaluate(self, trajectory: Trajectory, known_params: Dict[str, float]) -> None:
        """Evaluate all success metrics for a trajectory."""
        if not trajectory.states:
            return

        initial = trajectory.states[0]
        final = trajectory.states[-1]

        # 1. RECOVERY: how close are final params to known values?
        recovery_errors = []
        for k, known_val in known_params.items():
            if k in final.params and known_val != 0:
                rel_err = abs(final.params[k] - known_val) / abs(known_val)
                recovery_errors.append(rel_err)
        if recovery_errors:
            self.recovery_score = max(0, 1.0 - np.mean(recovery_errors))

        # 2. REDUCTION: complexity went down while conservation held?
        if final.complexity() < initial.complexity():
            last_diag = trajectory.diagnostics[-1] if trajectory.diagnostics else {}
            if last_diag.get("conservation_error", 1.0) < 1e-6:
                reduction = (initial.complexity() - final.complexity()) / initial.complexity()
                self.reduction_score = reduction

        # 3. COUPLING: did new couplings reduce conservation error?
        if len(trajectory.diagnostics) > 1:
            initial_err = trajectory.diagnostics[0].get("conservation_error", 1.0)
            final_err = trajectory.diagnostics[-1].get("conservation_error", 1.0)
            initial_couplings = sum(len(v) for v in initial.couplings.values())
            final_couplings = sum(len(v) for v in final.couplings.values())
            if final_couplings > initial_couplings and final_err < initial_err:
                self.coupling_score = 1.0 - (final_err / max(initial_err, 1e-30))

        # 4. DISCOVERY: did new relations improve consistency?
        if len(trajectory.diagnostics) > 1:
            initial_consist = trajectory.diagnostics[0].get("consistency", 0)
            final_consist = trajectory.diagnostics[-1].get("consistency", 0)
            initial_rels = len(initial.relations)
            final_rels = len(final.relations)
            if final_rels > initial_rels and final_consist > initial_consist:
                self.discovery_score = final_consist - initial_consist

        # 5. UNIFICATION: relations span multiple domains?
        domains = set(r.domain for r in final.relations.values())
        if len(domains) >= 3:
            self.unification_score = len(domains) / 5.0  # normalize to ~1

    def report(self) -> str:
        lines = [
            "  ── SUCCESS METRICS ──",
            "",
            f"    Recovery    (known params):     {self.recovery_score:.3f}",
            f"    Reduction   (Occam):            {self.reduction_score:.3f}",
            f"    Coupling    (new connections):   {self.coupling_score:.3f}",
            f"    Discovery   (new relations):     {self.discovery_score:.3f}",
            f"    Unification (cross-domain):      {self.unification_score:.3f}",
            f"    ─────────────────────────────────",
            f"    TOTAL SCORE:                     {self.total:.3f}",
        ]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# ACTION SUBSTRATE v0.2 — Theory Evolution Engine
# ═══════════════════════════════════════════════════════════════════

class ActionSubstrate:
    """
    The core substrate. Evolves model STRUCTURE through hypothesis space.

    Path-dependent: compares cumulative action of entire trajectories.
    Multi-trajectory: maintains a population of competing paths.
    Conservation-enforced: δΦ = 0 at every evaluation.
    """

    def __init__(
        self,
        initial_state: ModelState,
        lagrangian: Lagrangian,
        eta_initial: float = 1.0,
        eta_min: float = 0.01,
        eta_decay: float = 0.97,
        n_trajectories: int = 3,
    ):
        self.lagrangian = lagrangian
        self.eta = eta_initial
        self.eta_min = eta_min
        self.eta_decay = eta_decay

        self.proposer = StructuralProposer()
        self.symmetry = SymmetryChecker()

        # Multiple trajectories (path competition)
        self.trajectories: List[Trajectory] = []
        self.current_states: List[ModelState] = []
        for i in range(n_trajectories):
            traj = Trajectory()
            state = initial_state.clone()
            # Evaluate initial state
            action, diag = self.lagrangian.evaluate(state)
            traj.states.append(state)
            traj.actions.append(action)
            traj.diagnostics.append(diag)
            self.trajectories.append(traj)
            self.current_states.append(state.clone())

        self.epoch = 0

    def step(self, verbose: bool = True) -> Dict:
        """One epoch: evolve all trajectories, compare paths."""
        self.epoch += 1
        results = []

        for i, (traj, state) in enumerate(zip(self.trajectories, self.current_states)):
            # Propose structural mutation
            candidate, mutation = self.proposer.propose(state, self.eta, traj)
            candidate.epoch = self.epoch
            candidate.parent_epoch = state.epoch

            # Evaluate
            action, diag = self.lagrangian.evaluate(candidate)

            # Symmetry check
            sym_ok, sym_reason = self.symmetry.check(candidate, diag)

            if not sym_ok:
                status = "rejected"
                reason = sym_reason
            else:
                # Path-dependent acceptance: compare CUMULATIVE action
                # of trajectory WITH this step vs WITHOUT
                current_cum = traj.cumulative_action
                proposed_cum = ActionValue(
                    real=current_cum.real + action.real,
                    imag=current_cum.imag + action.imag,
                )

                # Accept if cumulative action improves, or stochastically
                delta_real = action.real - (traj.actions[-1].real if traj.actions else 0)
                accept = False

                if delta_real > 0:
                    accept = True
                    reason = "improved"
                elif self.eta > 1e-10:
                    prob = np.exp(min(delta_real / self.eta, 10))
                    if np.random.random() < prob:
                        accept = True
                        reason = f"stochastic (p={prob:.3f})"

                if accept:
                    self.current_states[i] = candidate
                    traj.states.append(candidate)
                    traj.actions.append(action)
                    traj.mutations.append(mutation)
                    traj.diagnostics.append(diag)
                    self.proposer.acceptances += 1
                    status = "accepted"
                else:
                    status = "rejected"
                    reason = f"action (Δ={delta_real:.4f})"

            result = {
                "trajectory": i,
                "epoch": self.epoch,
                "mutation": mutation.value,
                "detail": candidate.mutation_detail,
                "status": status,
                "action": action if sym_ok else None,
            }
            results.append(result)

            if verbose:
                marker = "✓" if status == "accepted" else "✗"
                action_str = f"S={action}" if sym_ok else "INVALID"
                print(f"  E{self.epoch:4d} T{i} | {marker} {mutation.value:20s} | "
                      f"{candidate.mutation_detail:45s} | {action_str}")

        # Anneal
        self.eta = max(self.eta_min, self.eta * self.eta_decay)

        return {"epoch": self.epoch, "results": results, "eta": self.eta}

    def evolve(self, n_epochs: int, verbose: bool = True,
               known_params: Dict[str, float] = None) -> SuccessMetrics:
        """Run full evolution with trajectory comparison."""

        if verbose:
            print("=" * 78)
            print("  ActionSubstrate v0.2 — Theory Evolution Engine")
            print("  by Jeff Stewart")
            print("=" * 78)
            print()
            print("  Primitive: ACTION  |  Acceptance: δΦ = 0")
            print(f"  Trajectories: {len(self.trajectories)}  |  η: {self.eta:.3f} → {self.eta_min}")
            print(f"  Mutation types: {len(MutationType)} (parameter + structural)")
            print(f"  Relation library: {len(RELATION_LIBRARY)} constitutive relations")
            print()
            print(f"  Initial model:")
            print(self.current_states[0].summary())
            print()

        for epoch in range(n_epochs):
            self.step(verbose=verbose)

        # Compare trajectories
        if verbose:
            print()
            print("  ── TRAJECTORY COMPARISON (path-dependent) ──")
            print()
            for i, traj in enumerate(self.trajectories):
                print(f"  Trajectory {i}:")
                print(traj.summary())
                print()

            # Best trajectory by cumulative action
            best_idx = max(range(len(self.trajectories)),
                           key=lambda i: self.trajectories[i].cumulative_action.real)
            print(f"  BEST TRAJECTORY: {best_idx} "
                  f"(cumulative Re(S) = {self.trajectories[best_idx].cumulative_action.real:.4f})")

            # Final model of best trajectory
            best_state = self.current_states[best_idx]
            print()
            print("  ── BEST MODEL ──")
            print(best_state.summary())
            print()
            print("  Parameters:")
            for k, v in sorted(best_state.params.items()):
                print(f"    {k:25s} = {v:.6e}")
            print()
            print("  Relations:")
            for name, rel in best_state.relations.items():
                print(f"    {name:30s} [{rel.domain}] {rel.form}")
            print()
            if best_state.lagrangian_terms:
                print("  Lagrangian terms:")
                for name, form in best_state.lagrangian_terms.items():
                    print(f"    {name:30s}: {form}")
                print()

            # Mutation statistics
            print("  ── MUTATION STATISTICS ──")
            print()
            for mut, count in sorted(self.proposer.mutation_history.items()):
                print(f"    {mut:25s}: {count}")
            print(f"    {'acceptance rate':25s}: "
                  f"{self.proposer.acceptances}/{self.proposer.proposals} "
                  f"({self.proposer.acceptances/max(1,self.proposer.proposals)*100:.0f}%)")
            print(f"    {'Lagrangian evals':25s}: {self.lagrangian.eval_count}")

        # Success metrics
        metrics = SuccessMetrics()
        best_traj = self.trajectories[best_idx]
        if known_params:
            metrics.evaluate(best_traj, known_params)

        if verbose:
            print()
            print(metrics.report())
            print()
            print("=" * 78)

        return metrics


# ═══════════════════════════════════════════════════════════════════
# DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════

def demo():
    """
    Theory evolution: start with a basic NAND model,
    let the substrate discover structure.
    """

    # Initial model: minimal, correct, but incomplete
    initial = ModelState(
        params={
            "T_env": 298.0,
            "C_thermal": 100.0,
            "L_die": 0.001,
            "k_Si": 150.0,
            "A_die": 1e-4,
            "tau0": 3.15e7,
            "Ea": 0.9,
            "rad_flux": 1e4,
            "let_mean": 10.0,
            "let_sigma": 0.5,
        },
        couplings={
            "nand": {"thermal"},
            "radiation": {"nand"},
            "thermal": set(),
        },
        relations={
            "arrhenius_simple": deepcopy(RELATION_LIBRARY["arrhenius_simple"]),
            "fourier_cooling": deepcopy(RELATION_LIBRARY["fourier_cooling"]),
        },
    )

    # Known ground truth (for recovery scoring)
    known = {
        "Ea": 0.9,
        "tau0": 3.15e7,
        "k_Si": 150.0,
        "T_env": 298.0,
    }

    lagrangian = Lagrangian(lambda_complexity=0.005, novelty_bonus=0.03)

    substrate = ActionSubstrate(
        initial_state=initial,
        lagrangian=lagrangian,
        eta_initial=0.8,
        eta_min=0.02,
        eta_decay=0.95,
        n_trajectories=3,
    )

    metrics = substrate.evolve(
        n_epochs=50,
        known_params=known,
    )


if __name__ == "__main__":
    demo()
