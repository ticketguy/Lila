"""
Emergence Engine — Slow Rhythm (Reflection)

Trigger: Lila is idle
What happens: She reflects on her own memories, finds patterns,
develops personality, updates her understanding of Sammie.

This is where Lila becomes MORE herself over time.
"""

from ..core.lilacore import LilaCore
from ..core.personality import EmergentPersonality


class EmergenceEngine:
    """
    The reflection loop. Runs when Lila has nothing else to do.
    Produces: connective memory, personality updates, insights.
    """
    
    def __init__(self, core: LilaCore):
        self.core = core
        self._reflection_count = 0
    
    def reflect(self):
        """Run one reflection cycle."""
        self._reflection_count += 1
        
        # Ask LilaCore to reflect on recent interactions
        reflection_prompt = (
            "Reflect on recent conversations. "
            "What patterns do you notice? "
            "What does Sammie care about? "
            "What should you remember long-term?"
        )
        
        response = self.core.think(reflection_prompt, context={
            "mode": "reflection",
            "silent": True,  # don't speak this
        })
        
        # Any memory ops from reflection get stored
        # Personality updates happen naturally through the memory writes
        
        return response
