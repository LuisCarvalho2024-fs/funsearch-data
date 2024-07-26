"""A programs database that implements the evolutionary algorithm."""
from collections.abc import Mapping, Sequence
import time
from typing import Any

import numpy as np

from CodeManipulator import code_utils
from Database.Prompt import Prompt
from Database.Island import Island
from CodeManipulator.Program import Program
import utils

class ProgramsDatabase:
  def __init__(self, function_name, template):
    self.reset_period = 1 * 15 * 60 # 15min
    self.num_islands = 4
    self.template = template
    
    self.function_name = function_name

    self._islands = []
    for _ in range(self.num_islands):
      self._islands.append(Island(function_name, template))

    self._best_score_per_island = [-float('inf')] * self.num_islands
    self._best_program_per_island = [None] * self.num_islands
    self._best_programs_per_island_cluster = [None] * self.num_islands

    self._last_reset_time = time.time()

  def get_prompt(self):
    island_id = np.random.randint(len(self._islands))
    code, version_generated = self._islands[island_id].get_prompt()
    # print('version_generated', version_generated, '- island_id:', island_id)
    return Prompt(code, version_generated, island_id)

  def _register_program_in_island(self, program, island_id, scores, signature):
    self._islands[island_id].register_program(program, scores, signature)
    score = utils.reduce_score(scores)
    if score > self._best_score_per_island[island_id]:
      self._best_program_per_island[island_id] = program
      self._best_score_per_island[island_id] = score
      self._best_programs_per_island_cluster[island_id] = signature
      # This doesn't make sense, check if can delete
      # self._best_score_per_island[island_id] = score

  def register_program(self, program, island_id, scores, signature):
    
    if island_id is None:
      for island_id in range(len(self._islands)):
        self._register_program_in_island(program, island_id, scores, signature)
    else:
      self._register_program_in_island(program, island_id, scores, signature)

    if (time.time() - self._last_reset_time > self.reset_period):
      self._last_reset_time = time.time()
      self.reset_islands()

  def reset_islands(self):
    indices_sorted_by_score = np.argsort(self._best_score_per_island)
    num_islands_to_reset = self.num_islands // 2
    reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
    keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]
    print("Resetting islands: ")
    for island_id in keep_islands_ids:
      print("island ", island_id, "has been kept with score ", self._best_score_per_island[island_id])
      
    for island_id in reset_islands_ids:
      print("island ", island_id, " has been reset with score ", self._best_score_per_island[island_id])
      self._islands[island_id] = Island(self.function_name, self.template)

      self._best_score_per_island[island_id] = -1
      founder_island_id = np.random.choice(keep_islands_ids)
      founder = self._best_program_per_island[founder_island_id]
      founder_cluster_signature = self._best_programs_per_island_cluster[founder_island_id]
      founder_score = self._best_score_per_island[founder_island_id]
      # TODO: return meaningful columns
      self._register_program_in_island(founder, island_id, founder_score, founder_cluster_signature)

  def database_state(self):
    island_best_scores = []
    island_sizes = []
    for island in self._islands:
      island_best_scores.append(island.getBestScore())
      island_sizes.append(island.getSize())
    
    print("Current Experiment scores")
    print("Island best scores: ", island_best_scores)
    print("Island sizes: ", island_sizes)
    
    print(island_best_scores)
    return island_best_scores, island_sizes
