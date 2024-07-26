"""Class for evaluating programs proposed by the Sampler."""
import ast
from collections.abc import Sequence
import copy
import numpy as np
import sys

from typing import Any

import subprocess
import tempfile

from CodeManipulator.code_utils import rename_function_calls
from utils import _calls_ancestor, debug_print, _get_signature
from Database.ProgramsDatabase import ProgramsDatabase
import subprocess

class _FunctionLineVisitor(ast.NodeVisitor):
  """Visitor that finds the last line number of a function with a given name."""

  def __init__(self, target_function_name: str) -> None:
    self._target_function_name: str = target_function_name
    self._function_end_line: int | None = None

  def visit_FunctionDef(self, node: Any): 
    """Collects the end line number of the target function."""
    if node.name == self._target_function_name:
      self._function_end_line = node.end_lineno
    self.generic_visit(node)

  @property
  def function_end_line(self):
    """Line number of the final line of function `target_function_name`."""
    assert self._function_end_line is not None
    return self._function_end_line

class Sandbox:
  """Sandbox for executing generated code."""

  def run(self, program: str, function_name_to_run: str, test_input: str, timeout_seconds):
    lines = program.split('\n')
    filtered_lines = [line for line in lines if not line.startswith('@funsearch')]
    program = '\n'.join(filtered_lines)

    template_run = """
if __name__ == "__main__":
  result, columns = run_evaluate()
  print(result, ' - ', columns)
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
      temp_file.write(program)
      temp_file.write(template_run)

    try:
      result = subprocess.run(['python', temp_file.name], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
      return [-1, False, []]
    
    if result.returncode == 0:
      result_str = result.stdout.decode('utf-8')
      if debug_print():
        print("Output:", result_str)
    else:
      # Failure could be a code-error or parsing error.
      if debug_print():        
        print("Output: FAILED ------------------------------------------ WHY:\n")
        print(program)
        print("Error:", result.stdout.decode('utf-8'))
      return [-1, False, []]

    # Usually, result is on second-to-last position
    # Refactor to hardcode a hash to the print so we can jump directly to it.
    try:
      results = result_str.split('\n')
      parts = results[-2].split('-')
      score = float(parts[0].strip())
      columns = eval(parts[1].strip())
      print(score)
      if (score == -1):
        return [-1, False, []]
      return [score, True, columns]
    except:
      return [-1, False, []]

# I suppose the sequence is actually inputs to run the function, in this case we likely won't need it;
# OR, actually, use different parts of the dataset as validation set.
class Evaluator:
  def __init__(self, database, template_program, function_name, function_name_to_run, inputs: Sequence[Any], timeout_seconds = 60):
    self._database = database
    self._template_program = template_program
    self._function_name = function_name
    self._function_name_to_run = function_name_to_run
    self._inputs = inputs
    self._timeout_seconds = timeout_seconds
    self._sandbox = Sandbox()
    self.runs = 0
    self.failed_runs = 0
    self.repeated_runs = 0

  def analyse(self, sample: str, island_id, version_generated):
    """Compiles the sample into a program and executes it on test inputs."""
    self.runs = self.runs + 1
    new_function, program = self._sample_to_program(sample, version_generated, self._template_program, self._function_name)

    scores_per_test = {}
    for current_input in self._inputs:
      test_output, runs_ok, columns = self._sandbox.run(program, self._function_name_to_run, current_input, self._timeout_seconds)
      if debug_print():
        print("runs ok:", runs_ok, "- calls_ancestor:", _calls_ancestor(program, self._function_name), "- score:",  test_output, "- columns: ", columns)
      if (runs_ok and not _calls_ancestor(program, self._function_name) and test_output is not None):
        if not isinstance(test_output, (int, float)):
          raise ValueError('@function.run did not return an int/float score.')
        scores_per_test[current_input] = test_output
    if scores_per_test:
      signature = _get_signature(columns)
      self._database.register_program(new_function, island_id, scores_per_test, signature)
      
    if (not runs_ok):
      self.failed_runs = self.failed_runs + 1
      return -1  
    
    return scores_per_test

  # TODO: Move the imports to top of code, though it should not make any difference
  def separateImports(self, code):
    if code == None:
      return '', ''
    code_lines = code.split('\n')
    
    # Filter out lines containing "import" while preserving indentation
    cleaned_code_lines = []
    imported = []
    for line in code_lines:
      if 'import' not in line:
        cleaned_code_lines.append(line)
      else:
        imported.append(line)
    
    import_code = '\n'.join(imported)
    
    # Remove the actual function definition
    cleaned_code_lines_2 = []
    for line in cleaned_code_lines:
      if not 'df: pd.DataFrame' in line:
        cleaned_code_lines_2.append(line)
    cleaned_code2 = '\n'.join(cleaned_code_lines_2)
    
    return import_code, cleaned_code2

  def _trim_function_body(self, generated_code: str):
    """Extracts the body of the generated function, trimming anything after it."""
    if not generated_code:
      return ''

    code = f'def fake_function_header():\n{generated_code}'
    tree = None
    
    # We keep trying and deleting code from the end until the parser succeeds.
    while tree is None:
      try:
        tree = ast.parse(code)
      except SyntaxError as e:
        code = '\n'.join(code.splitlines()[:e.lineno - 1])
    if not code:
      # Nothing could be saved. Generally, indicates leaked function definition or wrong indentation
      return ''

    visitor = _FunctionLineVisitor('fake_function_header')
    visitor.visit(tree)
    body_lines = code.splitlines()[1:visitor.function_end_line]
    return '\n'.join(body_lines) + '\n\n'
  
  # TODO: Fix the body part below, it's sometimes running with the base part
  def _sample_to_program(self, generated_code, version_generated, template_program, function_name):
    """Returns the compiled generated function and the full runnable program."""
    import_code, clean_code = self.separateImports(generated_code)
    body = self._trim_function_body(clean_code)
    if debug_print():
      print("THIS IS BODY: ---------------------------\n", body)
    if version_generated is not None:
      body = rename_function_calls(body, f'{function_name}_v{version_generated}', function_name)

    program = copy.deepcopy(template_program)
    program.before_functions = program.before_functions + '\n' + import_code
    evolved_function = program.get_function(function_name)
    evolved_function.body = body

    if debug_print():
      print("THIS IS FUNCTION NAME: --------------------", function_name)
      print("THIS IS EVOLVED FUNCTION: -------------\n", evolved_function)
      print("THIS IS PROGRAM: ------------- \n", program)
    return evolved_function, str(program)
  