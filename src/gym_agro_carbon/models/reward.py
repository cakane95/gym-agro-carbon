from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import numpy as np
from gym_agro_carbon.models.context import ContextSpec, ContextEncoder

@dataclass(frozen=True, slots=True)
class InvestmentConfig:
    """
    Spécification action-agnostic pour les investissements.
    """
    is_investment: bool = False
    duration: int = 0              # H : Horizon de l'investissement (maturité)
    terminal_value: float = 0.0    # V_mature : Potentiel résiduel à maturité
    mu_delta_soc_expected: float = 0.0
    mu_yield_expected: float = 0.0

@dataclass(frozen=True, slots=True)
class RewardSpec:
    """
    Paramètres globaux du modèle de récompense alignés sur GAMA.
    """
    context_spec: ContextSpec
    alpha: float                # Compromis Carbone vs Rendement
    discount_rate: float        # delta : taux d'actualisation

    # Moyennes alignées sur GAMA
    base_mu_delta_soc: Dict[int, float] = field(default_factory=lambda: {
        0: 0.4, # Fallow
        1: 0.6, # Manure
        2: 1.0, # Tree
        3: 0.2  # Baseline
    })
    
    base_mu_yield: Dict[int, float] = field(default_factory=lambda: {
        0: 0.0, 
        1: 0.0, 
        2: 0.0, 
        3: 1.5
    })
    
    sigma_soc: float = 0.14
    sigma_yield: float = 0.5
    
    investment_configs: Dict[int, InvestmentConfig] = field(default_factory=dict)

@dataclass(slots=True)
class RewardModel:
    spec: RewardSpec
    encoder: ContextEncoder

    def _get_annuity_factor(self, H: int) -> float:
        """Calcule la somme géométrique des facteurs d'actualisation (Eq. 2)."""
        delta = self.spec.discount_rate
        return sum(1.0 / ((1 + delta) ** k) for k in range(H + 1))

    def mu_reward(self, action: int, s: int, tau: int) -> float:
        """
        Calcule la valeur de l'action selon l'Eq. (2) sur l'horizon H.
        """
        config = self.spec.investment_configs.get(action)
        H = self.spec.context_spec.M # Horizon de référence (7 ans)
        alpha = self.spec.alpha

        # 1. Neutralisation durant la croissance (1 <= tau < M)
        if config and config.is_investment and 0 < tau < self.spec.context_spec.M:
            return 0.0

        # 2. Calcul pour les actions d'investissement au déclenchement (tau = 0)
        if config and config.is_investment and tau == 0:
            mu_r = (alpha * config.mu_delta_soc_expected + 
                    (1.0 - alpha) * config.mu_yield_expected)
            npv = mu_r * self._get_annuity_factor(config.duration)
            terminal_discounted = config.terminal_value / ((1 + self.spec.discount_rate)**config.duration)
            return float(npv + terminal_discounted)

        # 3. Actions de flux : projection sur l'horizon H
        mu_soc = self.spec.base_mu_delta_soc.get(action, 0.0)
        mu_yield = self.spec.base_mu_yield.get(action, 0.0)
        mu_instantaneous = alpha * mu_soc + (1.0 - alpha) * mu_yield
        
        return float(mu_instantaneous * self._get_annuity_factor(H))

    def get_learning_reward(self, action: int, s: int, tau: int, 
                            real_delta_soc: float, real_yield: float) -> float:
        """Feedback VNA/Neutralisation pour l'apprentissage."""
        config = self.spec.investment_configs.get(action)
        if config and config.is_investment and 0 < tau < self.spec.context_spec.M:
            return 0.0
        if config and config.is_investment and tau == 0:
            return self.mu_reward(action, s, tau)
        return self.spec.alpha * real_delta_soc + (1.0 - self.spec.alpha) * real_yield