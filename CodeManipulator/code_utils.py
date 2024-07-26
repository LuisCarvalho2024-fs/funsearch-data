# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tools for manipulating Python code.

It implements 2 classes representing unities of code:
- Function, containing all the information we need about functions: name, args,
  body and optionally a return type and a docstring.
- Program, which contains a before_functions (which could be imports, global
  variables and classes, ...) and a list of Functions.
"""
import ast
from collections.abc import Iterator, MutableSet, Sequence
import io
import tokenize

from CodeManipulator.Function import Function
from CodeManipulator.Program import Program
from CodeManipulator.ProgramVisitor import ProgramVisitor

# TODO: CHECK WHERE TO MOVE ALL OF THE BELOW, MAY NOT NEED THE YIELD_DECORATOR
def text_to_program(text: str) -> Program:
  try:
    # We assume that the program is composed of some before_functions (e.g. imports,
    # classes, assignments, ...) followed by a sequence of functions.
    tree = ast.parse(text)
    visitor = ProgramVisitor(text)
    visitor.visit(tree)
    return visitor.return_program()
  except Exception as e:
    # Failed parsing, usually identation related
    # print('Failed parsing %s', text)
    raise e


def text_to_function(text: str) -> Function:
  """Returns Function object by parsing input text using Python AST."""
  program = text_to_program(text)
  if len(program.functions) != 1:
    raise ValueError(f'Only one function expected, got {len(program.functions)}'f':\n{program.functions}')
  return program.functions[0]


def _tokenize(code: str) -> Iterator[tokenize.TokenInfo]:
  """Transforms `code` into Python tokens."""
  code_bytes = code.encode()
  code_io = io.BytesIO(code_bytes)
  return tokenize.tokenize(code_io.readline)


def _untokenize(tokens: Sequence[tokenize.TokenInfo]) -> str:
  """Transforms a list of Python tokens into code."""
  code_bytes = tokenize.untokenize(tokens)
  return code_bytes.decode()


def _yield_token_and_is_call(code: str) -> Iterator[tuple[tokenize.TokenInfo, bool]]:
  """Yields each token with a bool indicating whether it is a function call."""
  try:
    tokens = _tokenize(code)
    prev_token = None
    is_attribute_access = False
    for token in tokens:
      if (prev_token and  # If the previous token exists and
          prev_token.type == tokenize.NAME and  # it is a Python identifier
          token.type == tokenize.OP and  # and the current token is a delimiter
          token.string == '('):  # and in particular it is '('.
        yield prev_token, not is_attribute_access
        is_attribute_access = False
      else:
        if prev_token:
          is_attribute_access = (prev_token.type == tokenize.OP and prev_token.string == '.')
          yield prev_token, False
      prev_token = token
    if prev_token:
      yield prev_token, False
  except Exception as e:
    # Failed parsing, usually identation related
    # print('Failed parsing %s', code)
    raise e


def rename_function_calls(code, source_name, target_name) -> str:
  """Renames function calls from `source_name` to `target_name`."""
  if source_name not in code:
    return code
  modified_tokens = []
  for token, is_call in _yield_token_and_is_call(code):
    if is_call and token.string == source_name:
      # Replace the function name token
      modified_token = tokenize.TokenInfo(
          type=token.type,
          string=target_name,
          start=token.start,
          end=token.end,
          line=token.line,
      )
      modified_tokens.append(modified_token)
    else:
      modified_tokens.append(token)
  return _untokenize(modified_tokens)


def get_functions_called(code: str) -> MutableSet[str]:
  """Returns the set of all functions called in `code`."""
  return set(token.string for token, is_call in _yield_token_and_is_call(code) if is_call)


def yield_decorated(code: str, module: str, name: str) -> Iterator[str]:
  """Yields names of functions decorated with `@module.name` in `code`."""
  tree = ast.parse(code)
  for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef):
      for decorator in node.decorator_list:
        attribute = None
        if isinstance(decorator, ast.Attribute):
          attribute = decorator
        elif isinstance(decorator, ast.Call):
          attribute = decorator.func
        if (attribute is not None and attribute.value.id == module and attribute.attr == name):
          yield node.name
