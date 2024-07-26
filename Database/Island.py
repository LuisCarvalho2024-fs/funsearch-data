from Database.Cluster import Cluster
from CodeManipulator.Function import Function
from CodeManipulator.Program import Program
import utils
import copy
import dataclasses
from CodeManipulator import code_utils
from utils import debug_print
from collections.abc import Mapping, Sequence

import numpy as np

Signature = tuple[float, ...]

class Island:
  def __init__(self, function_name, template):
    self._template_program: Program | None = template
    self.function_name = function_name
    self.functions_per_prompt = 3
    self.cluster_sampling_temperature_init = 0.1
    self.cluster_sampling_temperature_period = 10000

    self._clusters: dict[Signature, Cluster] = {}
    self._num_programs: int = 0

  def register_program(self, program, scores, signature):
    score = utils.reduce_score(scores)
    if signature not in self._clusters:
      self._clusters[signature] = Cluster(score, program)
    else:
      self._clusters[signature].register_program(score, program)
    self._num_programs += 1

  def get_prompt(self):
    signatures = list(self._clusters.keys())
    cluster_scores = np.array([self._clusters[signature].score for signature in signatures])

    period = self.cluster_sampling_temperature_period
    temperature = self.cluster_sampling_temperature_init * (1 - (self._num_programs % period) / period)
    probabilities = utils._softmax(cluster_scores, temperature)
    
    functions_per_prompt = min(len(self._clusters), self.functions_per_prompt)

    idx = np.random.choice(len(signatures), size=functions_per_prompt, p=probabilities)
    chosen_signatures = [signatures[i] for i in idx]
    implementations = []
    scores = []
    for signature in chosen_signatures:
      cluster = self._clusters[signature]
      implementations.append(cluster.sample_program())
      scores.append(cluster.score)

    indices = np.argsort(scores)
    sorted_implementations = [implementations[i] for i in indices]
    version_generated = len(sorted_implementations) + 1
    return self._generate_prompt(sorted_implementations), version_generated

  # TODO: fix prompt leaking program
  def _generate_prompt(self, implementations: Sequence[code_utils.Function]):
    implementations = copy.deepcopy(implementations)
    if debug_print():
      print(implementations)
    
    versioned_functions: list[code_utils.Function] = []
    for i, implementation in enumerate(implementations):
      new_function_name = f'{self.function_name}_v{i}'
      implementation.name = new_function_name
      
      if i >= 1:
        implementation.docstring = (f'Improved version of `{self.function_name}_v{i - 1}`.')
      
      implementation = code_utils.rename_function_calls(str(implementation), self.function_name, new_function_name)
      if debug_print():
        print("THIS IS IMPLEMENTATION: ------------ \n", implementation)
      versioned_functions.append(code_utils.text_to_function(implementation))
    
    next_version = len(implementations)
    new_function_name = f'{self.function_name}_v{next_version}'
    header = dataclasses.replace(
        implementations[-1],
        docstring=('Improved version of ' f'`{self.function_name}_v{next_version - 1}`.'),
        name=new_function_name,
        body=''
    )
    versioned_functions.append(header)
    
    prompt = dataclasses.replace(self._template_program, functions=versioned_functions)
    if debug_print():
      print("THIS IS PROMPT: --------- \n", prompt)
    return str(prompt)
  
  def getBestScore(self):
    bestScore = -1
    for signature, cluster in self._clusters.items():
      clusterScore = cluster.getBestScore()
      if clusterScore > bestScore:
        bestScore = clusterScore
    return bestScore
  
  def getSize(self):
    size = 0
    for signature, cluster in self._clusters.items():
      size += cluster.getSize()
    return size
  
  def hasSample(self, sample):
    for signature, cluster in self._clusters.items():
      if (cluster.hasSample(sample)):
        return True
    return False