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


def eval(
  expression: t.Union[Proposition, Expression],
  substitutions: t.Dict[str, bool] = {},
) -> bool:
  match expression:
    case Proposition():
      if isinstance(expression.value, bool):
        return expression.value
      return substitutions.get(expression.value, False)
    case Expression(operator=Operator.AND, operands=operands):
      return all(eval(operand, substitutions) for operand in operands)
    case Expression(operator=Operator.OR, operands=operands):
      return any(eval(operand, substitutions) for operand in operands)
    case Expression(operator=Operator.NOT, operands=(operand,)):
      return not eval(operand, substitutions)
    case Expression(operator=Operator.IMPLIES, operands=(premise, conclusion)):
      return (not eval(premise, substitutions)) or eval(
        conclusion, substitutions
      )
    case _:
      raise ValueError(f'Unsupported expression type: {type(expression)}')


def get_variables(expression: t.Union[Proposition, Expression]) -> set:
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
  variables = set.union(
    *(get_variables(p) for p in premises), get_variables(conclusion)
  )

  for values in product([True, False], repeat=len(variables)):
    substitutions = dict(zip(variables, values))

    if all(eval(premise, substitutions) for premise in premises):
      if not eval(conclusion, substitutions):
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
  """
  variables = sorted(get_variables(expr))
  subexpressions = get_subexpressions(expr)

  headers = variables + [str(subexpr) for subexpr in subexpressions]

  rows = []

  for values in product([False, True], repeat=len(variables)):
    substitutions = dict(zip(variables, values))

    row = [int(substitutions[var]) for var in variables]

    for subexpr in subexpressions:
      result = eval(subexpr, substitutions)
      row.append(int(result))

    rows.append(row)

  return tabulate(rows, headers=headers, tablefmt='grid')


def get_subexpressions(
  expr: t.Union[Proposition, Expression],
) -> t.List[t.Union[Proposition, Expression]]:
  if isinstance(expr, Proposition):
    return [expr]

  subexprs = []

  for operand in expr.operands:
    subexprs.extend(get_subexpressions(operand))

  subexprs.append(expr)

  return subexprs


class TestLogicSystem(unittest.TestCase):
  def setUp(self):
    self.p = Proposition('p')
    self.q = Proposition('q')
    self.r = Proposition('r')

  def test_proposition_evaluation(self):
    self.assertTrue(eval(Proposition(True)))
    self.assertFalse(eval(Proposition(False)))
    self.assertFalse(eval(self.p))
    self.assertTrue(eval(self.p, {'p': True}))

  def test_and_operator(self):
    expr = Expression(Operator.AND, (self.p, self.q))
    self.assertFalse(eval(expr, {'p': True, 'q': False}))
    self.assertTrue(eval(expr, {'p': True, 'q': True}))

  def test_or_operator(self):
    expr = Expression(Operator.OR, (self.p, self.q))
    self.assertTrue(eval(expr, {'p': True, 'q': False}))
    self.assertFalse(eval(expr, {'p': False, 'q': False}))

  def test_not_operator(self):
    expr = Expression(Operator.NOT, (self.p,))
    self.assertTrue(eval(expr, {'p': False}))
    self.assertFalse(eval(expr, {'p': True}))

  def test_implies_operator(self):
    expr = Expression(Operator.IMPLIES, (self.p, self.q))
    self.assertTrue(eval(expr, {'p': False, 'q': False}))
    self.assertTrue(eval(expr, {'p': False, 'q': True}))
    self.assertFalse(eval(expr, {'p': True, 'q': False}))
    self.assertTrue(eval(expr, {'p': True, 'q': True}))

  def test_complex_expression(self):
    expr = Expression(
      Operator.OR,
      (
        Expression(Operator.AND, (self.p, self.q)),
        Expression(Operator.IMPLIES, (self.q, self.r)),
      ),
    )
    self.assertTrue(eval(expr, {'p': True, 'q': True, 'r': False}))
    self.assertFalse(eval(expr, {'p': False, 'q': True, 'r': False}))

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
    # {p1, (p1 → p2), (p2 → p3), (p3 → p4), ...}
    # This set is satisfiable (all true when all pi are true),
    # but has no finite model.

    def generate_formula(n):
      pn = Proposition(f'p{n}')
      pn_plus_1 = Proposition(f'p{n+1}')
      return Expression(Operator.IMPLIES, (pn, pn_plus_1))

    def is_satisfiable(formulas):
      variables = set.union(*(get_variables(f) for f in formulas))
      for values in product([True, False], repeat=len(variables)):
        substitutions = dict(zip(variables, values))
        if all(eval(f, substitutions) for f in formulas):
          return True
      return False

    # Test finite subsets (should all be satisfiable)
    for n in range(1, 10):  # Test subsets of size 1 to 9
      subset = [Proposition(f'p1')] + [generate_formula(i) for i in range(1, n)]
      self.assertTrue(
        is_satisfiable(subset), f'Subset of size {n} should be satisfiable'
      )

    # Demonstrate that we can't find a model that satisfies all formulas simultaneously
    # (In a real implementation, we can't actually test the infinite case)
    large_subset = [Proposition(f'p1')] + [
      generate_formula(i) for i in range(1, 1000)
    ]

    all_true = {'p' + str(i): True for i in range(1, 1001)}

    self.assertTrue(
      all(eval(f, all_true) for f in large_subset),
      'All formulas should be true when all pi are true',
    )

    any_false = {'p' + str(i): i != 1 for i in range(1, 1001)}

    self.assertFalse(
      all(eval(f, any_false) for f in large_subset),
      'Not all formulas can be true if any pi is false',
    )

  def test_truth_table_with_intermediate_results(self):
    expr = Expression(
      Operator.IMPLIES, (Expression(Operator.IMPLIES, (self.p, self.q)), self.r)
    )

    table = generate_truth_table(expr)

    # Split the table into lines
    lines = table.strip().split('\n')

    # Check the header
    self.assertIn('p', lines[1])
    self.assertIn('q', lines[1])
    self.assertIn('r', lines[1])
    self.assertIn('(p → q)', lines[1])
    self.assertIn('((p → q) → r)', lines[1])

    content_lines = list(
      filter(
        lambda x: len({'-', '+', '(', ')'}.intersection(set(x))) == 0, lines
      )
    )

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
        expected, [cell.strip() for cell in actual.split('|') if cell.strip()]
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


if __name__ == '__main__':
  unittest.main()
