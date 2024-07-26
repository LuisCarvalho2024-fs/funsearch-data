"""Class for sampling new programs."""
from collections.abc import Collection, Sequence

import numpy as np

from Evaluator import Evaluator
from Database.ProgramsDatabase import ProgramsDatabase
from utils import debug_print

import openai
import re
from datetime import datetime
import csv

client = openai.OpenAI("add your key here");

OPEN_AI_MODEL="gpt-3.5-turbo"
# OPEN_AI_MODEL="gpt-4o-mini"

class LLM:
  def __init__(self):
    self.requests = 0
    self.failed_requests = 0
  
  def get_python_code(self, response):
    if 'python' in response:
      pattern = r"```python(.*?)```"
      matches = re.findall(pattern, response, re.DOTALL)
      if (len(matches)):
        return matches[0]
      else:
        return None
    else:
      return 'PYTHON IS NOT IN THE RESPONSE'
  
  # TODO: fix prompt occasionally leaking the entire program
  def draw_sample(self, prompt: str) -> str:
    """Returns a predicted continuation of `prompt`."""
    if debug_print():
      print("WE ARE SENDING THIS PROMPT: ----------- \n", prompt)
    self.requests = self.requests + 1
    response = client.chat.completions.create(
      model=OPEN_AI_MODEL,
        messages = [
          {"role": "system", "content": 'You are a code completer tool, only output code, in a simple and concise way. Only explain through comments.'},
          {"role": "user", "content": prompt}
      ],
      max_tokens=1000,
      n=1,
      temperature=1,
      presence_penalty=0,
      frequency_penalty=0.1,
    )
    
    if debug_print():
      print("THIS IS RETURNED API: -----------------------------:\n ", response.choices[0].message.content)
    code = self.get_python_code(response.choices[0].message.content)
    
    self.requests = self.requests + 1
    if (code == None):
      self.failed_requests = self.failed_requests + 1
      
    return code

class Sampler:
  """Node that samples program continuations and sends them for analysis."""

  def __init__(self, database, evaluator) -> None:
    self._database = database
    self._evaluator = evaluator
    self._llm = LLM()
    self._samples_run = 0
    self._samples_amount = 0
    self.start_time = datetime.now()
    self.failedBackups = 0

  def sample(self, sample_amount):
    """Continuously gets prompts, samples programs, sends them for analysis."""
    self._samples_left = sample_amount
    self._samples_amount = sample_amount
    while self._samples_run < sample_amount:
      print("This is sample number: ", self._samples_run)
      if (self._samples_run % 100 == 0):
        self.experiment_state()
        database_state = self._database.database_state()
        self.storeResults(self._samples_run, database_state)

      prompt = self._database.get_prompt()
      sample = self._llm.draw_sample(prompt.code)
      # Remove duplicated samples
      if (self.sampleIsRepeated(sample, prompt.island_id)):
        print("Got a repeated sample!")
        self._evaluator.repeated_runs = self._evaluator.repeated_runs + 1
        result = -1
      else:
        result = self._evaluator.analyse(sample, prompt.island_id, prompt.version_generated)
      
      if (result != -1):
        self.backupSample(self._samples_run, sample, result)
      
      self._samples_run = self._samples_run + 1
      
  def experiment_state(self):
    print("Failed runs: ", self._evaluator.failed_runs)
    print("Successful runs: ", self._evaluator.runs - self._evaluator.failed_runs)
    print("Repeated runs: ", self._evaluator.repeated_runs)
    print("Some backup has failed: ", self.failedBackups)
    
  def backupSample(self, sample_num, result, sample):
    filename = self.getBackupFileName()
    try:
      with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow([sample_num, result, sample])
    except:
      # Might be an error with the encoding or file corrupted
      try:
        with open(filename, mode='a', newline='') as file:
          writer = csv.writer(file, delimiter=';')
          writer.writerow([sample_num, result, 'Failed Sample'])
      except:
        # File is corrupted, but experiment in-memory might still be okay."
        self.failedBackups = 1
  
  def getBackupFileName(self):
    return 'backup_4o_norepeat_openai_api_' + self.start_time.strftime('%Y%m%d_%H%M') + '-' + str(self._samples_amount) + '.csv'
  
  def storeResults(self, sample_num, database_state):
    filename = self.getResultsFileName()
    best_scores, island_sizes = database_state
    with open(filename, mode='a', newline='') as file:
      writer = csv.writer(file, delimiter=';')
      for island_id in range(len(best_scores)):
        writer.writerow([sample_num,
                         island_id,
                         best_scores[island_id],
                         island_sizes[island_id]
                         ])

  def getResultsFileName(self):
    return 'results_4o_norepeat_' + self.start_time.strftime('%Y%m%d_%H%M') + '-' + str(self._samples_amount) + '.csv'
  
  def sampleIsRepeated(self, sample, island_id):
    island_of_interest = self._database._islands[island_id]
    if(island_of_interest.hasSample(sample)):
      return True
    return False