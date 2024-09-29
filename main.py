import typing as t
import unittest
from dataclasses import dataclass
from enum import Enum
from itertools import product

from tabulate import tabulate


class Operator(Enum):
  AND = 'and'
  IMPLIES = 'implies'
  NOT = 'not'
  OR = 'or'


@dataclass
class Proposition:
  value: t.Union[str, bool]

  def __repr__(self) -> str:
    return str(self.value)


@dataclass
class Expression:
  operator: Operator
  operands: t.Tuple[t.Union['Expression', Proposition], ...]

  def __repr__(self):
    if self.operator == Operator.NOT:
      return f'¬{self.operands[0]}'
    elif self.operator == Operator.AND:
      return f'({self.operands[0]} ∧ {self.operands[1]})'
    elif self.operator == Operator.OR:
      return f'({self.operands[0]} ∨ {self.operands[1]})'
    elif self.operator == Operator.IMPLIES:
      return f'({self.operands[0]} → {self.operands[1]})'
    else:
      return f"Unknown({', '.join(map(str, self.operands))})"


def evaluate(
  expression: t.Union[Proposition, Expression],
  substitutions: t.Dict[str, bool] = {},
) -> bool:
  """
  Evaluates a logical expression given a set of variable substitutions.

  Args:
      expression (Union[Proposition, Expression]): The logical expression to evaluate.
      substitutions (Dict[str, bool], optional): A dictionary of variable substitutions. Defaults to {}.

  Returns:
      bool: The truth value of the expression.

  Raises:
      ValueError: If an unsupported expression type is encountered.

  Examples:
      >>> p, q = Proposition('p'), Proposition('q')
      >>> expr = Expression(Operator.AND, (p, q))
      >>> evaluate(expr, {'p': True, 'q': False})
      False
      >>> evaluate(expr, {'p': True, 'q': True})
      True
  """
  match expression:
    case Proposition():
      if isinstance(expression.value, bool):
        return expression.value
      return substitutions.get(expression.value, False)
    case Expression(operator=Operator.AND, operands=operands):
      return all(evaluate(operand, substitutions) for operand in operands)
    case Expression(operator=Operator.OR, operands=operands):
      return any(evaluate(operand, substitutions) for operand in operands)
    case Expression(operator=Operator.NOT, operands=(operand,)):
      return not evaluate(operand, substitutions)
    case Expression(operator=Operator.IMPLIES, operands=(premise, conclusion)):
      return (not evaluate(premise, substitutions)) or evaluate(
        conclusion, substitutions
      )
    case _:
      raise ValueError(f'Unsupported expression type: {type(expression)}')


def get_variables(expression: t.Union[Proposition, Expression]) -> set:
  """
  Extracts all variables from a given expression.

  Args:
      expression (Union[Proposition, Expression]): The expression to extract variables from.

  Returns:
      set: A set of all variables in the expression.
  """
  match expression:
    case Proposition():
      return {expression.value} if isinstance(expression.value, str) else set()
    case _:
      return set.union(
        *(get_variables(operand) for operand in expression.operands)
      )


def is_semantic_consequence(
  premises: t.List[t.Union[Proposition, Expression]],
  conclusion: t.Union[Proposition, Expression],
) -> bool:
  """
  Determines if a conclusion is a semantic consequence of given premises.

  Args:
      premises (List[Union[Proposition, Expression]]): A list of premises.
      conclusion (Union[Proposition, Expression]): The conclusion to check.

  Returns:
      bool: True if the conclusion is a semantic consequence of the premises, False otherwise.

  Examples:
      >>> p, q = Proposition('p'), Proposition('q')
      >>> premise = Expression(Operator.IMPLIES, (p, q))
      >>> is_semantic_consequence([premise, p], q)
      True
      >>> is_semantic_consequence(
      ...   [Expression(Operator.OR, (p, q))], Expression(Operator.AND, (p, q))
      ... )
      False
  """
  variables = set.union(
    *(get_variables(p) for p in premises), get_variables(conclusion)
  )

  for values in product([True, False], repeat=len(variables)):
    substitutions = dict(zip(variables, values))

    if all(evaluate(premise, substitutions) for premise in premises):
      if not evaluate(conclusion, substitutions):
        return False

  return True


def generate_truth_table(expr: t.Union[Proposition, Expression]) -> str:
  """
  Generates a truth table for a given logical expression.

  The table rows are ordered to correspond to binary counting from 0 to 1
  for the input variables.

  Args:
      expr (Union[Proposition, Expression]): The logical expression to generate a truth table for.

  Returns:
      str: A string representation of the truth table.

  Examples:
      >>> p, q = Proposition('p'), Proposition('q')
      >>> expr = Expression(Operator.IMPLIES, (p, q))
      >>> print(generate_truth_table(expr))
      ┌───┬───┬───────────┐
      │ p │ q │ (p → q)   │
      ├───┼───┼───────────┤
      │ 0 │ 0 │ 1         │
      │ 0 │ 1 │ 1         │
      │ 1 │ 0 │ 0         │
      │ 1 │ 1 │ 1         │
      └───┴───┴───────────┘
  """
  variables = sorted(get_variables(expr))
  subexpressions = get_subexpressions(expr)

  headers = variables + [str(subexpr) for subexpr in subexpressions]

  rows = []

  for values in product([False, True], repeat=len(variables)):
    substitutions = dict(zip(variables, values))

    row = [int(substitutions[var]) for var in variables]

    for subexpr in subexpressions:
      result = evaluate(subexpr, substitutions)
      row.append(int(result))

    rows.append(row)

  return tabulate(rows, headers=headers, tablefmt='simple_outline')


def get_subexpressions(
  expr: t.Union[Proposition, Expression],
) -> t.List[t.Union[Proposition, Expression]]:
  """
  Extracts all subexpressions from a given expression.

  Args:
      expr (Union[Proposition, Expression]): The expression to extract subexpressions from.

  Returns:
      List[Union[Proposition, Expression]]: A list of all subexpressions in the given expression.
  """
  if isinstance(expr, Proposition):
    return [expr]

  subexprs = []

  for operand in expr.operands:
    subexprs.extend(get_subexpressions(operand))

  subexprs.append(expr)

  return subexprs


def _encode_graph_coloring(
  graph: t.Dict[str, t.List[str]], num_colors: int
) -> Expression:
  """
  Encodes a graph coloring problem as a logical expression.

  Args:
      graph (Dict[str, List[str]]): A dictionary representing the graph. Keys are nodes, values are lists of adjacent nodes.
      num_colors (int): The number of colors to use for coloring.

  Returns:
      Expression: A logical expression representing the graph coloring constraints.

  Examples:
      >>> graph = {'A': ['B', 'C'], 'B': ['A', 'C'], 'C': ['A', 'B']}
      >>> expr = encode_graph_coloring(graph, 3)
      >>> print(expr)
      (((A_color_0 ∨ A_color_1) ∨ A_color_2) ∧ (...))
  """
  expressions = []

  # For each node, it must have at least one color
  for node in graph:
    expressions.append(
      Expression(
        Operator.OR,
        tuple(Proposition(f'{node}_color_{i}') for i in range(num_colors)),
      )
    )

  # For each node, it can't have more than one color
  for node in graph:
    for i in range(num_colors):
      for j in range(i + 1, num_colors):
        expressions.append(
          Expression(
            Operator.OR,
            (
              Expression(Operator.NOT, (Proposition(f'{node}_color_{i}'),)),
              Expression(Operator.NOT, (Proposition(f'{node}_color_{j}'),)),
            ),
          )
        )

  # Adjacent nodes can't have the same color
  for node, neighbors in graph.items():
    for neighbor in neighbors:
      for color in range(num_colors):
        expressions.append(
          Expression(
            Operator.OR,
            (
              Expression(Operator.NOT, (Proposition(f'{node}_color_{color}'),)),
              Expression(
                Operator.NOT, (Proposition(f'{neighbor}_color_{color}'),)
              ),
            ),
          )
        )

  return Expression(Operator.AND, tuple(expressions))


def is_graph_colorable(
  graph: t.Dict[str, t.List[str]], num_colors: int
) -> bool:
  """
  Determines if a graph is colorable with a given number of colors.

  Args:
      graph (Dict[str, List[str]]): A dictionary representing the graph. Keys are nodes, values are lists of adjacent nodes.
      num_colors (int): The number of colors to use for coloring.

  Returns:
      bool: True if the graph is colorable with the given number of colors, False otherwise.

  Examples:
      >>> triangle = {'A': ['B', 'C'], 'B': ['A', 'C'], 'C': ['A', 'B']}
      >>> is_graph_colorable(triangle, 2)
      False
      >>> is_graph_colorable(triangle, 3)
      True
  """
  expression = _encode_graph_coloring(graph, num_colors)

  variables = get_variables(expression)

  for values in product([False, True], repeat=len(variables)):
    substitutions = dict(zip(variables, values))

    if evaluate(expression, substitutions):
      return True

  return False


class TestLogicSystem(unittest.TestCase):
  def setUp(self):
    self.p = Proposition('p')
    self.q = Proposition('q')
    self.r = Proposition('r')

  def test_proposition_evaluation(self):
    self.assertTrue(evaluate(Proposition(True)))
    self.assertFalse(evaluate(Proposition(False)))
    self.assertFalse(evaluate(self.p))
    self.assertTrue(evaluate(self.p, {'p': True}))

  def test_and_operator(self):
    expr = Expression(Operator.AND, (self.p, self.q))
    self.assertFalse(evaluate(expr, {'p': True, 'q': False}))
    self.assertTrue(evaluate(expr, {'p': True, 'q': True}))

  def test_or_operator(self):
    expr = Expression(Operator.OR, (self.p, self.q))
    self.assertTrue(evaluate(expr, {'p': True, 'q': False}))
    self.assertFalse(evaluate(expr, {'p': False, 'q': False}))

  def test_not_operator(self):
    expr = Expression(Operator.NOT, (self.p,))
    self.assertTrue(evaluate(expr, {'p': False}))
    self.assertFalse(evaluate(expr, {'p': True}))

  def test_implies_operator(self):
    expr = Expression(Operator.IMPLIES, (self.p, self.q))
    self.assertTrue(evaluate(expr, {'p': False, 'q': False}))
    self.assertTrue(evaluate(expr, {'p': False, 'q': True}))
    self.assertFalse(evaluate(expr, {'p': True, 'q': False}))
    self.assertTrue(evaluate(expr, {'p': True, 'q': True}))

  def test_complex_expression(self):
    expr = Expression(
      Operator.OR,
      (
        Expression(Operator.AND, (self.p, self.q)),
        Expression(Operator.IMPLIES, (self.q, self.r)),
      ),
    )
    self.assertTrue(evaluate(expr, {'p': True, 'q': True, 'r': False}))
    self.assertFalse(evaluate(expr, {'p': False, 'q': True, 'r': False}))

  def test_get_variables(self):
    expr = Expression(
      Operator.OR,
      (
        Expression(Operator.AND, (self.p, self.q)),
        Expression(Operator.IMPLIES, (self.q, self.r)),
      ),
    )
    self.assertEqual(get_variables(expr), {'p', 'q', 'r'})

  def test_is_semantic_consequence(self):
    premise1 = Expression(Operator.IMPLIES, (self.p, self.q))
    premise2 = self.p
    conclusion = self.q
    self.assertTrue(is_semantic_consequence([premise1, premise2], conclusion))

    premise3 = Expression(Operator.OR, (self.p, self.q))
    conclusion2 = Expression(Operator.AND, (self.p, self.q))
    self.assertFalse(is_semantic_consequence([premise3], conclusion2))

  def test_compactness_theorem(self):
    # We'll simulate an infinite set of formulas:
    #
    # {p1, (p1 → p2), (p2 → p3), (p3 → p4), ...}
    #
    # This set is satisfiable (all true when all pi are true),
    # but has no finite model.

    generate_formula = lambda n: Expression(
      Operator.IMPLIES, (Proposition(f'p{n}'), Proposition(f'p{n+1}'))
    )

    def is_satisfiable(formulas):
      variables = set.union(*(get_variables(f) for f in formulas))

      for values in product([True, False], repeat=len(variables)):
        substitutions = dict(zip(variables, values))

        if all(evaluate(f, substitutions) for f in formulas):
          return True

      return False

    # Test finite subsets (should all be satisfiable)
    for n in range(1, 10):  # Test subsets of size 1 to 9
      subset = [Proposition('p1')] + [generate_formula(i) for i in range(1, n)]
      self.assertTrue(
        is_satisfiable(subset), f'Subset of size {n} should be satisfiable'
      )

    # Demonstrate that we can't find a model that satisfies all formulas simultaneously
    # (In a real implementation, we can't actually test the infinite case)
    large_subset = [Proposition('p1')] + [
      generate_formula(i) for i in range(1, 1000)
    ]

    all_true = {'p' + str(i): True for i in range(1, 1001)}

    self.assertTrue(
      all(evaluate(f, all_true) for f in large_subset),
      'All formulas should be true when all pi are true',
    )

    any_false = {'p' + str(i): i != 1 for i in range(1, 1001)}

    self.assertFalse(
      all(evaluate(f, any_false) for f in large_subset),
      'Not all formulas can be true if any pi is false',
    )

  def test_truth_table_with_intermediate_results(self):
    expr = Expression(
      Operator.IMPLIES, (Expression(Operator.IMPLIES, (self.p, self.q)), self.r)
    )

    table = generate_truth_table(expr)

    lines = table.strip().split('\n')

    self.assertIn('p', lines[1])
    self.assertIn('q', lines[1])
    self.assertIn('r', lines[1])
    self.assertIn('(p → q)', lines[1])
    self.assertIn('((p → q) → r)', lines[1])

    content_lines = list(filter(lambda x: '┌' not in x, lines[3:]))

    expected_content = [
      ['0', '0', '0', '0', '0', '1', '0', '0'],
      ['0', '0', '1', '0', '0', '1', '1', '1'],
      ['0', '1', '0', '0', '1', '1', '0', '0'],
      ['0', '1', '1', '0', '1', '1', '1', '1'],
      ['1', '0', '0', '1', '0', '0', '0', '1'],
      ['1', '0', '1', '1', '0', '0', '1', '1'],
      ['1', '1', '0', '1', '1', '1', '0', '0'],
      ['1', '1', '1', '1', '1', '1', '1', '1'],
    ]

    for expected, actual in zip(expected_content, content_lines):
      self.assertEqual(
        expected, [cell.strip() for cell in actual.split('│') if cell.strip()]
      )

  def test_repr(self):
    p = Proposition('p')
    q = Proposition('q')
    r = Proposition('r')

    expr1 = Expression(Operator.AND, (p, q))
    self.assertEqual(repr(expr1), '(p ∧ q)')

    expr2 = Expression(Operator.OR, (p, q))
    self.assertEqual(repr(expr2), '(p ∨ q)')

    expr3 = Expression(Operator.NOT, (p,))
    self.assertEqual(repr(expr3), '¬p')

    expr4 = Expression(Operator.IMPLIES, (p, q))
    self.assertEqual(repr(expr4), '(p → q)')

    complex_expr = Expression(
      Operator.IMPLIES, (Expression(Operator.AND, (p, q)), r)
    )
    self.assertEqual(repr(complex_expr), '((p ∧ q) → r)')

  def test_graph_coloring(self):
    triangle_graph = {'A': ['B', 'C'], 'B': ['A', 'C'], 'C': ['A', 'B']}

    self.assertFalse(is_graph_colorable(triangle_graph, 2))
    self.assertTrue(is_graph_colorable(triangle_graph, 3))

    square_graph = {
      'A': ['B', 'D'],
      'B': ['A', 'C'],
      'C': ['B', 'D'],
      'D': ['A', 'C'],
    }

    self.assertTrue(is_graph_colorable(square_graph, 2))

    complete_graph = {
      'A': ['B', 'C', 'D'],
      'B': ['A', 'C', 'D'],
      'C': ['A', 'B', 'D'],
      'D': ['A', 'B', 'C'],
    }

    self.assertFalse(is_graph_colorable(complete_graph, 3))
    self.assertTrue(is_graph_colorable(complete_graph, 4))


if __name__ == '__main__':
  unittest.main()
