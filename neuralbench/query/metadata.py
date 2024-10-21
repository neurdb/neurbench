import pglast
from pglast.visitors import Visitor
from pglast import ast
import os
import glob
import argparse
from pprint import pprint


def print_ast(node):
    stmt = node[0].stmt
    pprint(stmt(depth=30, skip_none=True))


class SQLInfoExtractor(Visitor):
    def __init__(self):
        super().__init__()
        self.tables = set()
        self.joins = set()
        self.predicates = set()
        self.aliasname_fullname = {}

    @property
    def info(self):
        return {
            'tables': self.tables,
            'joins': self.joins,
            'predicates': self.predicates,
            'aliasname_fullname': self.aliasname_fullname,
        }

    def visit(self, ancestors, node):

        # Extract table names and aliases
        if isinstance(node, ast.RangeVar):
            table_name = node.relname
            self.tables.add(table_name)
            if hasattr(node, 'alias') and node.alias is not None:
                alias_name = node.alias.aliasname
                self.aliasname_fullname[alias_name] = table_name

        # Extract explicit join information
        if isinstance(node, ast.JoinExpr):
            if isinstance(node.larg, ast.RangeVar) and isinstance(node.rarg, ast.RangeVar):
                left_table = node.larg.relname
                right_table = node.rarg.relname
                if node.larg.alias:
                    left_table = node.larg.alias.aliasname
                if node.rarg.alias:
                    right_table = node.rarg.alias.aliasname
                if node.quals and isinstance(node.quals, ast.A_Expr):
                    lexpr = node.quals.lexpr.fields if hasattr(node.quals.lexpr, 'fields') else []
                    rexpr = node.quals.rexpr.fields if hasattr(node.quals.rexpr, 'fields') else []
                    if len(lexpr) == 2 and len(rexpr) == 2:
                        self.joins.add([lexpr[0].sval, lexpr[1].sval, rexpr[0].sval, rexpr[1].sval])

        # Extract implicit join conditions from WHERE clause
        if isinstance(node, ast.A_Expr):
            if node.name[0].sval == '=':
                lexpr = node.lexpr.fields if hasattr(node.lexpr, 'fields') else []
                rexpr = node.rexpr.fields if hasattr(node.rexpr, 'fields') else []
                # for join, both have values
                if len(lexpr) == 2 and len(rexpr) == 2:
                    self.joins.add((lexpr[0].sval, lexpr[1].sval, rexpr[0].sval, rexpr[1].sval))
                elif len(lexpr) == 1 and len(rexpr) == 1:
                    self.joins.add(("", lexpr[0].sval, "", rexpr[0].sval))

            # Extract other predicates
            if node.name[0].sval in ('>', '=', '<', '<=', '>='):
                lexpr = node.lexpr.fields if hasattr(node.lexpr, 'fields') else []
                if hasattr(node.rexpr, 'val'):
                    if hasattr(node.rexpr.val, 'ival'):
                        rexpr = node.rexpr.val.ival
                    elif hasattr(node.rexpr.val, 'sval'):
                        rexpr = node.rexpr.val.sval
                    else:
                        rexpr = None
                    if len(lexpr) == 2:
                        self.predicates.add((lexpr[0].sval, lexpr[1].sval, node.name[0].sval, rexpr))
                    elif len(lexpr) == 1:
                        self.predicates.add(("", lexpr[0].sval, node.name[0].sval, rexpr))
                elif hasattr(node.rexpr, 'arg'):
                    if hasattr(node.rexpr.arg, 'val'):
                        if hasattr(node.rexpr.arg.val, 'sval'):
                            rexpr = node.rexpr.arg.val.sval
                            if len(lexpr) == 2:
                                self.predicates.add((lexpr[0].sval, lexpr[1].sval, node.name[0].sval, rexpr))
                            elif len(lexpr) == 1:
                                self.predicates.add(("", lexpr[0].sval, node.name[0].sval, rexpr))

        # Handle subqueries and additional join conditions
        if isinstance(node, ast.SubLink):
            # Extract join conditions in subquery
            subselect = node.subselect
            if subselect:
                self(subselect)

        if isinstance(node, ast.BoolExpr):
            # Traverse all arguments in AND/OR expressions
            for arg in node.args:
                self(arg)

