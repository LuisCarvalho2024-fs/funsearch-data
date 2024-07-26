import numpy as np
from CodeManipulator.Program import Program
import utils

class Cluster:
  def __init__(self, score, programObj: Program):
    self._score = score
    self._programs = [programObj]

  @property
  def score(self):
    return self._score

  def register_program(self, score, programObj):
    if (score > self._score):
      self._score = score
    self._programs.append(programObj)

  def sample_program(self):
    return np.random.choice(self._programs)

  def getBestScore(self):
    return self._score
  
  def getSize(self):
    return len(self._programs)

  def hasSample(self, sample):
    if (type(sample) == str):
      sample_trimmed = ''.join(sample.split())
      for program in self._programs:
        if (type(program.body) == str):
          program_trimmed = ''.join(program.body.split())
          if (sample_trimmed == program_trimmed):
            return True
    return False
        