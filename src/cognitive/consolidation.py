"""
Consolidation Daemon — Medium Rhythm

Trigger: Every 15 minutes OR after significant task completion
Reads raw memory → identifies patterns → promotes to higher namespaces

In weight-space terms: reviews recent micro-training writes,
identifies what should be promoted from episodic → personal → wiki.
"""

from ..core.lilacore import LilaCore


class ConsolidationDaemon:
    """
    Background process that consolidates raw memories into structured knowledge.
    """
    
    def __init__(self, core: LilaCore, interval_minutes: int = 15):
        self.core = core
        self.interval = interval_minutes
    
    def run_cycle(self):
        """Run one consolidation cycle."""
        if not self.core.model or not self.core.model.has_memory:
            return
        
        confidence = self.core.model.memory_confidence()
        
        # Promote episodic → personal if accessed frequently
        episodic_mag = confidence.get("episodic", {}).get("mean_magnitude", 0)
        if episodic_mag > 0.01:
            self.core.model.promote_memory("episodic", "personal")
        
        # Promote personal → wiki if very strong
        personal_mag = confidence.get("personal", {}).get("mean_magnitude", 0)
        if personal_mag > 0.05:
            self.core.model.promote_memory("personal", "wiki")
        
        # Apply decay to unused
        self.core.model.memory_decay(hours=0.25)  # 15 min
