from CodeManipulator.Function import Function
from CodeManipulator.Program import Program
import ast

class ProgramVisitor(ast.NodeVisitor):
  """Parses code to collect all required information to produce a `Program`.

  Note that we do not store function decorators.
  """

  def __init__(self, sourcecode: str):
    self._codelines: list[str] = sourcecode.splitlines()

    self._before_functions: str = ''
    self._functions: list[Function] = []
    self._current_function: str | None = None

  def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
    """Collects all information about the function being parsed."""
    if node.col_offset == 0:  # We only care about first level functions.
      self._current_function = node.name
      if not self._functions:
        self._before_functions = '\n'.join(self._codelines[:node.lineno - 1])
      function_end_line = node.end_lineno
      body_start_line = node.body[0].lineno - 1
      # Extract the docstring.
      docstring = None
      if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
        docstring = f'  """{ast.literal_eval(ast.unparse(node.body[0]))}"""'
        if len(node.body) > 1:
          body_start_line = node.body[1].lineno - 1
        else:
          body_start_line = function_end_line

      self._functions.append(Function(
          name=node.name,
          args=ast.unparse(node.args),
          return_type=ast.unparse(node.returns) if node.returns else None,
          docstring=docstring,
          body='\n'.join(self._codelines[body_start_line:function_end_line]),
      ))
    self.generic_visit(node)

  def return_program(self) -> Program:
    return Program(before_functions=self._before_functions, functions=self._functions)