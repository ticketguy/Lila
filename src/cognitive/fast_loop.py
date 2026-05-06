"""
Fast Loop — Reactive Response

Trigger: Sammie speaks
Latency: Immediate
Flow: perceive → query memory → think → respond → write raw memory
"""

from ..core.lilacore import LilaCore, LilaResponse
from typing import Optional, Dict


class FastLoop:
    """The reactive cognitive loop. Sammie says something, Lila responds."""
    
    def __init__(self, core: LilaCore):
        self.core = core
    
    def process(self, input_text: str, context: Optional[Dict] = None) -> LilaResponse:
        """Process input through the fast loop."""
        return self.core.think(input_text, context)
