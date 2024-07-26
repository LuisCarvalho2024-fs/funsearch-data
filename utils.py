import numpy as np
import os
import scipy
from CodeManipulator.code_utils import get_functions_called

# HELPER DECLARATIONS
# ProgramsDatabase
def _softmax(logits: np.ndarray, temperature: float):
  """Returns the tempered softmax of 1D finite `logits`."""
  if not np.all(np.isfinite(logits)):
    non_finites = set(logits[~np.isfinite(logits)])
    raise ValueError(f'`logits` contains non-finite value(s): {non_finites}')
  if not np.issubdtype(logits.dtype, np.floating):
    logits = np.array(logits, dtype=np.float32)

  result = scipy.special.softmax(logits / temperature, axis=-1)
  # Ensure that probabilities sum to 1 to prevent error in `np.random.choice`.
  index = np.argmax(result)
  result[index] = 1 - np.sum(result[0:index]) - np.sum(result[index+1:])
  return result

# ProgramsDatabase
def reduce_score(scores):
  if (isinstance(scores, float)):
    return scores
  return scores[0]

# TODO: find a better way to identify clusters
# You can likely hash the ordered columns string
# ProgramsDatabase
def _get_signature(columns):
  return len(columns)

def _calls_ancestor(program: str, function_name: str) -> bool:
  """Returns whether the generated function is calling an earlier version."""
  for name in get_functions_called(program):
    # In `program` passed into this function the most recently generated
    # function has already been renamed to `function_name` (without the
    # suffix). Therefore any function call starting with `function_name_v`
    # is a call to an ancestor function.
    if name.startswith(f'{function_name}_v'):
      return True
  return False

def debug_print():
  return os.environ.get('FUNSEARCH_DEBUG', '') == "True"