"""
ProgressReporter – tick‑based progress updates for sequential agent pipelines.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class ProgressReporter:
    """
    Manages progress reporting for a sequential pipeline.

    The engine's progress bar uses a simple tick model:
      - Each agent consumes two ticks (start and end).
      - Total ticks = max(1, 2 * number_of_agents)
      - Start tick = 2 * agent_index + 1
      - End tick   = 2 * agent_index + 2

    This class encapsulates the tick arithmetic and calls the user‑supplied
    on_progress callback with the computed percentage and optional agent name.
    """

    def __init__(
        self,
        on_progress: Optional[Callable[[int, Optional[str]], None]],
        num_steps: int,
    ):
        """
        Args:
            on_progress: Callback (pct: int, agent_name: Optional[str]) -> None
            num_steps: Total number of agents in the pipeline.
        """
        self.on_progress = on_progress
        self.num_steps = num_steps
        self.total_ticks = max(1, 2 * num_steps)

    def _progress(self, pct: int, agent_name: Optional[str] = None) -> None:
        """Invoke the progress callback if it exists."""
        if self.on_progress:
            self.on_progress(pct, agent_name)
        else:
            logger.debug("Progress: %d%% (agent=%s)", pct, agent_name)

    def start_agent(self, i: int, agent_name: str) -> None:
        """
        Report that agent i (0‑based) has started execution.
        """
        start_tick = 2 * i + 1
        start_pct = int(100 * start_tick / self.total_ticks)
        self._progress(start_pct, agent_name)

    def end_agent(self, i: int, agent_name: str) -> None:
        """
        Report that agent i (0‑based) has finished execution.
        """
        end_tick = 2 * i + 2
        end_pct = int(100 * end_tick / self.total_ticks)
        self._progress(end_pct, agent_name)