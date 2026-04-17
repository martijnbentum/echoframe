'''Repository-specific style checks for echoframe.'''

from __future__ import annotations

import ast
import io
import re
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path


MAX_LINE_LENGTH = 90
TARGET_LINE_LENGTH = 80
PYTHON_ROOTS = ('echoframe',)
WRAPPED_IMPORT_RE = re.compile(r'^\s*from\s+\S+\s+import\s*\(')
ARG_DOC_RE = re.compile(r'^([a-zA-Z_][a-zA-Z0-9_]*):\s{2,}\S')
DOCSTRING_EXEMPTIONS = {
    'echoframe.codebooks/TokenCodebooks/token_count',
    'echoframe.compaction/broken_reference',
    'echoframe.compaction/build_shard_health_report',
    'echoframe.compaction/build_compaction_plan',
    'echoframe.compaction/create_compaction_journal',
    'echoframe.compaction/update_compaction_journal',
    'echoframe.compaction/list_compaction_journal',
    'echoframe.compaction/run_compaction_plan',
    'echoframe.lmdb_helper/open_env',
    'echoframe.lmdb_helper/open_databases',
    'echoframe.lmdb_helper/read_txn',
    'echoframe.lmdb_helper/write_txn',
    'echoframe.lmdb_helper/scan_prefix_in_txn',
    'echoframe.lmdb_helper/encode_phraser_key',
    'echoframe.lmdb_helper/get_metadata',
    'echoframe.lmdb_helper/get_many_metadata',
    'echoframe.lmdb_helper/list_metadata',
    'echoframe.lmdb_helper/get_metadata_in_txn',
    'echoframe.lmdb_helper/list_metadata_in_txn',
    'echoframe.lmdb_helper/write_metadata',
    'echoframe.lmdb_helper/delete_tag_keys',
    'echoframe.lmdb_helper/delete_shard_keys',
    'echoframe.lmdb_helper/delete_phraser_keys',
    'echoframe.lmdb_helper/shard_key',
    'echoframe.lmdb_helper/tag_key',
    'echoframe.lmdb_helper/phraser_scan_prefix',
    'echoframe.lmdb_helper/phraser_scan_key',
    'echoframe.model_registry/ModelMetadata/to_dict',
}


@dataclass
class Finding:
    path: Path
    line: int
    level: str
    message: str

    def format(self) -> str:
        return f'{self.level}: {self.path}:{self.line}: {self.message}'


def main() -> int:
    findings = []
    for path in iter_python_files():
        source = path.read_text()
        findings.extend(check_line_lengths(path, source))
        findings.extend(check_wrapped_imports(path, source))
        findings.extend(check_quote_style(path, source))
        findings.extend(check_ast_rules(path, source))

    findings.sort(key=lambda item: (str(item.path), item.line, item.level))
    for finding in findings:
        print(finding.format())

    error_count = sum(f.level == 'ERROR' for f in findings)
    warning_count = sum(f.level == 'WARN' for f in findings)
    if findings:
        print(f'{error_count} error(s), {warning_count} warning(s)')
    return 1 if error_count else 0


def iter_python_files():
    for root_name in PYTHON_ROOTS:
        root = Path(root_name)
        if not root.exists():
            continue
        for path in sorted(root.rglob('*.py')):
            yield path


def check_line_lengths(path, source):
    findings = []
    for lineno, line in enumerate(source.splitlines(), start=1):
        line_length = len(line)
        if line_length > MAX_LINE_LENGTH:
            message = f'line too long ({line_length} > {MAX_LINE_LENGTH})'
            findings.append(Finding(path, lineno, 'ERROR', message))
        elif line_length > TARGET_LINE_LENGTH:
            message = f'line exceeds target ({line_length} > {TARGET_LINE_LENGTH})'
            findings.append(Finding(path, lineno, 'WARN', message))
    return findings


def check_wrapped_imports(path, source):
    findings = []
    for lineno, line in enumerate(source.splitlines(), start=1):
        if WRAPPED_IMPORT_RE.match(line):
            findings.append(Finding(path, lineno, 'ERROR',
                'wrapped from-import with parentheses is not allowed'))
    return findings


def check_quote_style(path, source):
    findings = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    except tokenize.TokenError:
        return findings
    for token in tokens:
        if token.type != tokenize.STRING:
            continue
        token_text = token.string
        if _is_docstring_candidate(token_text):
            continue
        if _uses_double_quotes_without_need(token_text):
            findings.append(Finding(path, token.start[0], 'ERROR',
                'use single quotes unless double quotes are needed'))
    return findings


def check_simple_if_return_layout(path, source):
    findings = []
    lines = source.splitlines()
    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped.startswith('if ') or not stripped.endswith(':'):
            continue
        if lineno >= len(lines):
            continue
        next_line = lines[lineno]
        if not next_line.strip().startswith('return '):
            continue
        combined = stripped[:-1] + ': ' + next_line.strip()
        if len(combined) > MAX_LINE_LENGTH:
            continue
        findings.append(Finding(path, lineno, 'ERROR',
            'simple if-return blocks should stay on one line'))
    return findings


def check_ast_rules(path, source):
    findings = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        findings.append(Finding(path, exc.lineno or 1, 'ERROR',
            f'syntax error: {exc.msg}'))
        return findings
    findings.extend(check_module_function_order(path, tree))
    findings.extend(check_class_method_order(path, tree))
    findings.extend(check_public_docstrings(path, tree))
    findings.extend(check_small_multiline_dict_literals(path, source, tree))
    findings.extend(check_simple_if_return_layout(path, source))
    findings.extend(check_direct_nested_calls(path, source, tree))
    findings.extend(check_multiline_inline_raise(path, source, tree))
    findings.extend(check_return_dict_literals(path, tree))
    findings.extend(check_return_comprehensions(path, tree))
    findings.extend(check_vararg_usage(path, tree))
    findings.extend(check_docstring_blocks(path, tree))
    return findings


def check_module_function_order(path, tree):
    findings = []
    seen_private = False
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith('_'):
            seen_private = True
            continue
        if seen_private:
            findings.append(Finding(path, node.lineno, 'WARN',
                'public function appears after private helper'))
    return findings


def check_class_method_order(path, tree):
    findings = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        stage = 'dunder'
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            name = item.name
            if _is_dunder(name):
                if stage != 'dunder':
                    findings.append(Finding(path, item.lineno, 'WARN',
                        f'dunder method {name} should appear before other methods'))
                continue
            if name.startswith('_'):
                stage = 'private'
                continue
            if stage == 'private':
                findings.append(Finding(path, item.lineno, 'WARN',
                    f'public method {name} appears after private helper'))
            stage = 'public'
    return findings


def check_public_docstrings(path, tree):
    findings = []
    module_name = python_module_name(path)
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            findings.extend(check_public_docstrings_in_class(path,
                module_name, node))
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith('_'):
                continue
            if is_docstring_exempt(module_name, node.name):
                continue
            docstring = ast.get_docstring(node, clean=False)
            if docstring is None:
                findings.append(Finding(path, node.lineno, 'WARN',
                    f'public function {node.name} is missing a docstring'))
                continue
            if needs_parameter_block(node):
                if not has_parameter_block(docstring):
                    findings.append(Finding(path, node.lineno, 'WARN',
                        f'public function {node.name} is missing '
                        'parameter explanations'))
    return findings


def check_public_docstrings_in_class(path, module_name, class_node):
    findings = []
    for node in class_node.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != '__init__' and node.name.startswith('_'):
            continue
        if is_docstring_exempt(module_name, class_node.name, node.name):
            continue
        docstring = ast.get_docstring(node, clean=False)
        if docstring is None:
            message = f'public method {class_node.name}.{node.name} '
            message += 'is missing a docstring'
            findings.append(Finding(path, node.lineno, 'WARN', message))
            continue
        if needs_parameter_block(node):
            if has_parameter_block(docstring):
                continue
            message = f'public method {class_node.name}.{node.name} '
            message += 'is missing parameter explanations'
            findings.append(Finding(path, node.lineno, 'WARN', message))
    return findings


def check_vararg_usage(path, tree):
    findings = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.args.vararg is not None:
            findings.append(Finding(path, node.lineno, 'WARN',
                f'{node.name} uses *args; this should need strong justification'))
        if node.args.kwarg is not None:
            findings.append(Finding(path, node.lineno, 'WARN',
                f'{node.name} uses **kwargs; this should need strong justification'))
    return findings


def check_small_multiline_dict_literals(path, source, tree):
    findings = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        if node.lineno == getattr(node, 'end_lineno', node.lineno):
            continue
        item_count = len(node.keys)
        if item_count > 3:
            continue
        source_text = ast.get_source_segment(source, node)
        if source_text is not None:
            compact = ' '.join(source_text.split())
            if len(compact) > MAX_LINE_LENGTH:
                continue
        findings.append(Finding(path, node.lineno, 'ERROR',
            'small dict literals must stay on one line'))
    return findings


def check_direct_nested_calls(path, source, tree):
    findings = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not is_direct_nested_call(node):
            continue
        source_text = ast.get_source_segment(source, node)
        if source_text is None:
            continue
        compact = ' '.join(source_text.split())
        is_multiline = '\n' in source_text
        if not is_multiline and len(compact) <= 40:
            continue
        findings.append(Finding(path, node.lineno, 'ERROR',
            'direct nested function calls are not allowed'))
    return findings


def check_multiline_inline_raise(path, source, tree):
    findings = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise):
            continue
        exc = node.exc
        if not isinstance(exc, ast.Call):
            continue
        if exc.lineno == getattr(exc, 'end_lineno', exc.lineno):
            continue
        if not has_inline_string_argument(exc):
            continue
        findings.append(Finding(path, node.lineno, 'ERROR',
            'build exception messages in a variable before multi-line raise'))
    return findings


def check_return_dict_literals(path, tree):
    findings = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Return):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        dict_node = node.value
        if dict_node.lineno == getattr(dict_node, 'end_lineno', dict_node.lineno):
            continue
        item_count = len(dict_node.keys)
        if item_count <= 1:
            continue
        message = 'prefer building multi-line dict literals in a variable '
        message += 'before returning'
        findings.append(Finding(path, dict_node.lineno, 'WARN', message))
    return findings


def check_return_comprehensions(path, tree):
    findings = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Return):
            continue
        if node.value is None:
            continue
        for item in ast.walk(node.value):
            if not isinstance(item, (ast.DictComp, ast.ListComp, ast.SetComp)):
                continue
            if item.lineno == getattr(item, 'end_lineno', item.lineno):
                continue
            findings.append(Finding(path, item.lineno, 'WARN',
                'prefer explicit loops over multi-line return '
                'comprehensions'))
    return findings


def check_docstring_blocks(path, tree):
    findings = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith('_'):
            continue
        docstring = ast.get_docstring(node, clean=False)
        if not docstring or '\n' not in docstring:
            continue
        lines = [line.rstrip() for line in docstring.splitlines()[1:]
            if line.strip()]
        arg_lines = [line for line in lines if ':' in line]
        if not arg_lines:
            continue
        if any(not ARG_DOC_RE.match(line.strip()) for line in arg_lines):
            findings.append(Finding(path, node.lineno, 'WARN',
                f'docstring parameter block for {node.name} is malformed'))
    return findings


def _is_docstring_candidate(token_text):
    prefixes = ('"""', "'''", 'r"""', "r'''", 'u"""', "u'''",
        'f"""', "f'''", 'fr"""', "fr'''", 'rf"""', "rf'''")
    lowered = token_text.lower()
    return lowered.startswith(prefixes)


def _uses_double_quotes_without_need(token_text):
    prefix = ''
    while token_text and token_text[0] in 'rRuUbBfF':
        prefix += token_text[0]
        token_text = token_text[1:]
    if not token_text.startswith('"') or token_text.startswith('"""'):
        return False
    if "'" in token_text and '"' not in token_text[1:-1]:
        return False
    body = token_text[1:-1]
    if "'" in body:
        return False
    return True


def _is_dunder(name):
    return name.startswith('__') and name.endswith('__')


def has_parameter_block(docstring):
    if '\n' not in docstring:
        return False
    lines = [line.strip() for line in docstring.splitlines()[1:] if line.strip()]
    return any(ARG_DOC_RE.match(line) for line in lines)


def is_docstring_exempt(module_name, function_name, method_name=None):
    if method_name is None:
        key = f'{module_name}/{function_name}'
    else:
        key = f'{module_name}/{function_name}/{method_name}'
    return key in DOCSTRING_EXEMPTIONS


def needs_parameter_block(node):
    parameter_count = len(real_parameters(node))
    return parameter_count >= 2


def is_direct_nested_call(node):
    if isinstance(node.func, ast.Call):
        return True
    for arg in node.args:
        if isinstance(arg, ast.Call):
            return True
    for keyword in node.keywords:
        if isinstance(keyword.value, ast.Call):
            return True
    return False


def has_inline_string_argument(node):
    for arg in node.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return True
        if isinstance(arg, ast.JoinedStr):
            return True
    return False


def python_module_name(path):
    parts = list(path.with_suffix('').parts)
    return '.'.join(parts)


def real_parameters(node):
    parameters = []
    posonlyargs = getattr(node.args, 'posonlyargs', [])
    for arg in posonlyargs + node.args.args + node.args.kwonlyargs:
        if arg.arg in ('self', 'cls'):
            continue
        parameters.append(arg)
    return parameters


if __name__ == '__main__':
    raise SystemExit(main())
