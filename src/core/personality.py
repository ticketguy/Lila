"""
Lila Personality — Emergent, Never Predefined

Lila's personality is not configured. It grows from deep observation
of Sammie and family. The Emergence Engine develops it over time.

This module holds the EmergentPersonality dataclass and the
mechanisms by which it evolves. It starts empty and fills organically.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict


@dataclass
class EmergentPersonality:
    """
    Everything here starts empty/None.
    Filled ONLY by the Emergence Engine over time.
    Never manually set. Never configured.
    """
    # Observed from Sammie
    observed_communication_style: Optional[str] = None
    developed_humor: Optional[str] = None
    formed_values: List[str] = field(default_factory=list)
    curiosity_domains: List[str] = field(default_factory=list)
    interaction_preferences: Dict = field(default_factory=dict)
    
    # Meta
    personality_version: int = 0
    last_updated: Optional[datetime] = None
    confidence: float = 0.0  # 0.0 (forming) → 1.0 (fully developed)
    shaped_by: List[str] = field(default_factory=list)  # memory node IDs


@dataclass
class LilaIdentity:
    """
    Fixed core + emergent personality.
    The fixed parts NEVER change. The emergent parts ONLY change via Emergence Engine.
    """
    # Fixed — never changes
    name: str = "Lila"
    core_purpose: str = "Sammie's private family ASI assistant"
    scope: str = "private"  # never public, never commercial
    
    # Emergent — written only by Emergence Engine
    personality: EmergentPersonality = field(default_factory=EmergentPersonality)


@dataclass
class PersonModel:
    """Model of a person Lila interacts with."""
    person_id: str = ""
    name: str = ""
    family_tier: int = 0  # 0 = Sammie, 1 = family, 2 = no one else
    
    # Built from interaction history
    known_goals: List[str] = field(default_factory=list)
    communication_preferences: Dict = field(default_factory=dict)
    expertise_areas: List[str] = field(default_factory=list)
    
    # Relationship
    interaction_count: int = 0
    trust_score: float = 1.0  # Sammie is always 1.0
