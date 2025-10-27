"""
SaveDelayCallback: Prevent checkpoint saving during early training.

This callback blocks checkpoint saving until a minimum number of training steps
have been completed, preventing unnecessary saves during the initial warmup phase
when eval_loss is decreasing rapidly.
"""
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from ..utils import get_logger

logger = get_logger(__name__)


class SaveDelayCallback(TrainerCallback):
    """
    A callback that prevents checkpoint saving before a specified step/epoch threshold.
    
    This is useful when using `save_strategy: best` and you want to avoid saving
    checkpoints during the initial training phase when metrics are improving rapidly.
    
    Args:
        save_delay_steps (int, optional): Minimum number of training steps before 
            allowing checkpoint saves. Takes precedence over save_delay_epochs.
        save_delay_epochs (float, optional): Minimum number of epochs before 
            allowing checkpoint saves. Only used if save_delay_steps is None.
        
    Example:
        In your training script:
        ```python
        from src.callbacks import SaveDelayCallback
        
        # Prevent saves for first 100 steps
        callback = SaveDelayCallback(save_delay_steps=100)
        
        # Or prevent saves for first 0.5 epochs
        callback = SaveDelayCallback(save_delay_epochs=0.5)
        ```
    """
    
    def __init__(
        self, 
        save_delay_steps: int = None,
        save_delay_epochs: float = None
    ):
        if save_delay_steps is None and save_delay_epochs is None:
            raise ValueError(
                "At least one of save_delay_steps or save_delay_epochs must be specified"
            )
        
        self.save_delay_steps = save_delay_steps
        self.save_delay_epochs = save_delay_epochs
        self._logged_delay = False
    
    def on_step_end(
        self, 
        args: TrainingArguments, 
        state: TrainerState, 
        control: TrainerControl, 
        **kwargs
    ):
        """
        Override should_save if we're still in the delay period.
        """
        # Check if we're still in delay period
        in_delay_period = False
        
        if self.save_delay_steps is not None:
            if state.global_step < self.save_delay_steps:
                in_delay_period = True
        elif self.save_delay_epochs is not None:
            if state.epoch is not None and state.epoch < self.save_delay_epochs:
                in_delay_period = True
        
        # Block saving if in delay period
        if in_delay_period and control.should_save:
            if not self._logged_delay:
                delay_info = (
                    f"{self.save_delay_steps} steps" 
                    if self.save_delay_steps is not None 
                    else f"{self.save_delay_epochs} epochs"
                )
                # Format epoch safely; state.epoch can be None
                _epoch = state.epoch if state.epoch is not None else 0.0
                logger.info(
                    f"SaveDelayCallback: Blocking checkpoint saves until {delay_info} "
                    f"(current: step={state.global_step}, epoch={_epoch:.2f})"
                )
                self._logged_delay = True
            
            control.should_save = False
        
        # Log when delay period ends
        if not in_delay_period and not self._logged_delay:
            logger.info(
                f"SaveDelayCallback: Delay period ended at step {state.global_step}. "
                f"Checkpoint saving is now enabled."
            )
            self._logged_delay = True
        
        return control

