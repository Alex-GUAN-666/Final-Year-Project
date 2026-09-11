#!/usr/bin/env python3
"""Deterministic structural/provenance audit; NEVER executes supplied/generated code.

All source bytes and all JSON fields are ingested. AST parsing is static. Coverage
is not a certification that every mathematical derivation has been checked.
"""
import argparse
import ast
import csv
import hashlib
import json
import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = ('problem', 'generated_solution', 'expected_answer', 'problem_type',
          'generated_python_code', 'execution_result', 'is_correct')


def sha(value):
    if isinstance(value, str):
        value = value.encode('utf-8')
    return hashlib.sha256(value).hexdigest()


def normalize_question(value):
    # Conservative: no punctuation removal or mathematical equivalence claims.
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())


def load_source(path, kind):
    raw = path.read_bytes()
    text = raw.decode('utf-8-sig')
    errors = []
    physical = []
    if kind == 'jsonl':
        rows = []
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                errors.append({'line': number, 'error': 'blank line'})
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append({'line': number, 'error': str(exc)})
                continue
            if not isinstance(row, dict):
                errors.append({'line': number, 'error': 'not a JSON object'})
                continue
            rows.append(row)
            physical.append(number)
    else:
        # CSV quoted newlines are parsed as part of one field, not extra rows.
        import io
        reader = csv.DictReader(io.StringIO(text, newline=''))
        rows = []
        for row in reader:
            rows.append(row)
            physical.append(reader.line_num)
    return rows, {'filename': path.name, 'bytes': len(raw), 'sha256': sha(raw),
                  'records': len(rows), 'physical_lines': len(text.splitlines()),
                  'parse_errors': errors, 'record_physical_line_or_csv_end': physical}


def eligible(row):
    # Exact historical pre-tokenizer predicate; not a strengthened substitute.
    return bool(row.get('is_correct') and row.get('generated_solution')
                and row.get('generated_python_code'))


def groups(rows, key):
    grouped = defaultdict(list)
    for number, row in enumerate(rows, 1):
        grouped[key(row)].append(number)
    return grouped


def duplicate_summary(rows, key):
    grouped = groups(rows, key)
    duplicate_groups = [ids for _, ids in sorted(grouped.items()) if len(ids) > 1]
    return {'unique': len(grouped), 'duplicate_groups': len(duplicate_groups),
            'repeat_occurrences': len(rows) - len(grouped), 'row_groups': duplicate_groups}


def code_info(code):
    if code in ('N/A', 'None', ''):
        return {'status': 'placeholder_' + (code or 'empty')}
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError) as exc:
        return {'status': 'syntax_error', 'error': str(exc).split(' (')[0]}
    modules, calls, solve_nodes = [], [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or '')
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == 'solve':
            solve_nodes.append(node)
    literal_return = False
    solve_hashes = []
    for node in solve_nodes:
        body = [n for n in node.body if not (isinstance(n, ast.Expr)
                and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))]
        if len(body) == 1 and isinstance(body[0], ast.Return):
            value = body[0].value
            literal_return |= (isinstance(value, ast.Constant) and
                               isinstance(value.value, (int, float, complex)) and
                               not isinstance(value.value, bool)) or (
                isinstance(value, ast.UnaryOp) and isinstance(value.op, (ast.UAdd, ast.USub))
                and isinstance(value.operand, ast.Constant)
                and isinstance(value.operand.value, (int, float, complex)))
        solve_hashes.append(sha(ast.dump(ast.Module(body=body, type_ignores=[]),
                                        annotate_fields=True, include_attributes=False)))
    return {'status': 'parsed', 'ast_sha256': sha(ast.dump(tree, include_attributes=False)),
            'solve_body_sha256': solve_hashes, 'imports': sorted(set(modules)),
            'solve_definitions': len(solve_nodes), 'literal_numeric_return_only': literal_return,
            'sensitive_call_names': sorted(set(calls) & {'exec', 'eval', 'open', '__import__',
                  'system', 'popen', 'run', 'Popen', 'remove', 'unlink', 'rmtree', 'input'})}


def count_summary(rows, infos):
    counts = Counter((r.get('problem_type'), json.dumps(r.get('is_correct'))) for r in rows)
    return {
        'records': len(rows),
        'type_correctness': [{'problem_type': t, 'is_correct': json.loads(c), 'count': n}
                            for (t, c), n in sorted(counts.items())],
        'eligible_count': sum(eligible(r) for r in rows),
        'execution_status': dict(sorted(Counter(
            'skipped_proof' if r['execution_result'] == 'Skipped: Classified as PROOF' else
            'skipped_non_computable' if r['execution_result'] == 'Skipped: Classified as NON_COMPUTABLE' else
            'no_code' if r['execution_result'] == 'No code to execute' else
            'execution_failed' if r['execution_result'].startswith('Execution failed:') else
            'recorded_output' for r in rows).items())),
        'execution_failure_messages': dict(sorted(Counter(
            r['execution_result'] for r in rows
            if r['execution_result'].startswith('Execution failed:')).items())),
        'code_status': dict(sorted(Counter(i['status'] for i in infos).items())),
        'imports': dict(sorted(Counter(m for info in infos for m in info.get('imports', [])).items())),
        'literal_numeric_return_only_rows': [i for i, info in enumerate(infos, 1)
                                               if info.get('literal_numeric_return_only')],
        'sensitive_call_rows': [{'row': i, 'calls': info['sensitive_call_names']}
                               for i, info in enumerate(infos, 1)
                               if info.get('sensitive_call_names')],
    }


def overlap(rows, question_key, code_key, target_rows, target_infos, eligible_ids):
    question_index = groups(target_rows, lambda r: normalize_question(r['problem']))
    exact_question_index = groups(target_rows, lambda r: r['problem'])
    code_index = groups(target_rows, lambda r: r['generated_python_code'])
    ast_index = defaultdict(list)
    for i, info in enumerate(target_infos, 1):
        if info.get('ast_sha256'):
            ast_index[info['ast_sha256']].append(i)
    mapped = []
    for i, row in enumerate(rows, 1):
        q = row[question_key]
        matches = question_index.get(normalize_question(q), [])
        code = row.get(code_key, '') if code_key else ''
        ci = code_info(code) if code else {}
        mapped.append({'row': i, 'question_sha256': sha(q),
            'exact_question_source_rows': exact_question_index.get(q, []),
            'normalized_question_source_rows': matches,
            'eligible_source_rows': sorted(set(matches) & eligible_ids),
            'exact_code_source_rows': code_index.get(code, []) if code else [],
            'code_ast_source_rows': ast_index.get(ci.get('ast_sha256'), [])})
    return {'records': len(rows),
            'exact_question_matches': sum(bool(m['exact_question_source_rows']) for m in mapped),
            'normalized_question_matches': sum(bool(m['normalized_question_source_rows']) for m in mapped),
            'eligible_question_matches': sum(bool(m['eligible_source_rows']) for m in mapped),
            'exact_code_matches': sum(bool(m['exact_code_source_rows']) for m in mapped),
            'code_ast_matches': sum(bool(m['code_ast_source_rows']) for m in mapped),
            'mappings': mapped,
            'code_overlap_caution': 'Identical short programs can solve different questions; code-only overlap does not establish contamination.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--output-dir', type=Path)
    ap.add_argument('--retained-source-lines', type=Path,
                    help='Optional JSON list of exact 1-based final token-filtered distilled source lines.')
    args = ap.parse_args()
    root = args.root.resolve()
    out = args.output_dir or root / 'reports'
    args.retained_source_lines = args.retained_source_lines or root / 'reports/retained_source_lines.json'
    out.mkdir(parents=True, exist_ok=True)
    files = {
        'distilled': ('data/raw/results_v3.jsonl', 'jsonl'),
        'arithmetic_train': ('data/raw/arithmetic_logic_data_revised_v2.jsonl', 'jsonl'),
        'reasoning_eval': ('data/raw/random_samples_for_inference.csv', 'csv'),
        'arithmetic_eval': ('data/raw/arithmetic_problems_v3.csv', 'csv'),
        'historical_eval': ('reports/cot_historical_paired_results.csv', 'csv'),
        'base_results': ('archive/evaluation/evaluation_results_base_cot_v5(1).csv', 'csv'),
        'merged_results': ('archive/evaluation/evaluation_results_merged_cot_v5(1).csv', 'csv'),
    }
    datasets, manifest = {}, {}
    for name, (path, kind) in files.items():
        datasets[name], manifest[name] = load_source(root / path, kind)
    rows = datasets['distilled']
    infos = [code_info(r['generated_python_code']) for r in rows]
    ids = {i for i, row in enumerate(rows, 1) if eligible(row)}
    candidates = [row for row in rows if eligible(row)]
    cand_infos = [info for row, info in zip(rows, infos) if eligible(row)]
    report = {'audit_format_version': 1,
              'coverage': {'all_input_bytes_read': True, 'all_json_records_and_fields_parsed': True,
                'generated_code_executed': False, 'model_or_tokenizer_run': False,
                'mathematical_derivations_all_verified': False,
                'scope': 'Exhaustive structural/static/identity audit; selected mathematical issue review.'},
              'manifest': manifest,
              'schema': {k: {'present': sum(k in r for r in rows),
                            'types': dict(sorted(Counter(type(r.get(k)).__name__ for r in rows).items())),
                            'empty_strings': sum(r.get(k) == '' for r in rows),
                            'nulls': sum(r.get(k) is None for r in rows)} for k in SCHEMA},
              'unexpected_keys': sorted({k for r in rows for k in r} - set(SCHEMA)),
              'all': count_summary(rows, infos),
              'eligible': count_summary(candidates, cand_infos),
              'eligible_source_lines': sorted(ids),
              'duplicates': {}, 'overlap': {}}
    # candidate-only row numbers differ from physical lines; store both explicitly.
    report['eligible']['row_numbering'] = '1-based order within eligible_source_lines, not physical JSONL lines'
    for scope, scoped_rows in [('all', rows), ('eligible', candidates)]:
        report['duplicates'][scope] = {
            'full_record': duplicate_summary(scoped_rows, lambda r: json.dumps(r, sort_keys=True, ensure_ascii=False)),
            'exact_question': duplicate_summary(scoped_rows, lambda r: r['problem']),
            'normalized_question': duplicate_summary(scoped_rows, lambda r: normalize_question(r['problem']))}
    report['duplicate_question_conflicts'] = []
    for question, line_ids in sorted(groups(rows, lambda r: normalize_question(r['problem'])).items()):
        if len(line_ids) < 2:
            continue
        duplicate_rows = [rows[i - 1] for i in line_ids]
        labels = sorted({r['expected_answer'] for r in duplicate_rows})
        types = sorted({r['problem_type'] for r in duplicate_rows})
        statuses = sorted({json.dumps(r['is_correct']) for r in duplicate_rows})
        if len(labels) > 1 or len(types) > 1 or len(statuses) > 1:
            report['duplicate_question_conflicts'].append({'source_lines': line_ids,
                'question_sha256': sha(question), 'expected_answers': labels,
                'problem_types': types, 'correctness_statuses': [json.loads(s) for s in statuses]})
    for name, qkey, ckey in [
        ('arithmetic_train', 'problem', 'generated_python_code'),
        ('reasoning_eval', 'arithmetical question', 'generated_python_code'),
        ('arithmetic_eval', 'arithmetical question', 'code for solving the question'),
        ('historical_eval', 'question', None),
        ('base_results', 'question', 'extracted_code'),
        ('merged_results', 'question', 'extracted_code')]:
        report['overlap'][name] = overlap(datasets[name], qkey, ckey, rows, infos, ids)
    projection_index = groups(rows, lambda r: json.dumps([r['problem'], r['expected_answer'],
                          r['generated_python_code'], r['generated_solution']], ensure_ascii=False))
    projection_maps = []
    for i, r in enumerate(datasets['reasoning_eval'], 1):
        key = json.dumps([r['arithmetical question'], r['answer to the question'],
                          r['generated_python_code'], r['generated_solution']], ensure_ascii=False)
        matched = projection_index.get(key, [])
        projection_maps.append({'evaluation_row': i, 'exact_four_field_source_lines': matched,
                                'eligible_source_lines': sorted(set(matched) & ids)})
    report['reasoning_eval_exact_projection'] = {'count': sum(bool(m['exact_four_field_source_lines'])
           for m in projection_maps), 'fields': ['problem', 'expected_answer',
           'generated_python_code', 'generated_solution'], 'mappings': projection_maps}
    eval_keys = {normalize_question(r['arithmetical question']) for r in datasets['reasoning_eval']}
    first20_keys = {normalize_question(r['arithmetical question']) for r in datasets['reasoning_eval'][:20]}
    remaining = [r for r in candidates if normalize_question(r['problem']) not in eval_keys]
    report['prospective_question_holdout'] = {
        'full_eval_unique_questions': len(eval_keys),
        'full_eval_matching_eligible_records': sum(normalize_question(r['problem']) in eval_keys for r in candidates),
        'remaining_eligible_records_before_token_filter': len(remaining),
        'remaining_unique_questions_before_token_filter': len({normalize_question(r['problem']) for r in remaining}),
        'first20_matching_eligible_records': sum(normalize_question(r['problem']) in first20_keys for r in candidates),
        'interpretation': 'Prospective revised experiment only; cannot repair an already-trained historical checkpoint.'}
    for k in ('problem', 'generated_solution', 'generated_python_code', 'expected_answer', 'execution_result'):
        lengths = [len(r[k]) for r in rows]
        report.setdefault('field_lengths', {})[k] = {'characters': sum(lengths), 'min': min(lengths),
                         'median': statistics.median(lengths), 'max': max(lengths)}
    report['reasoning_tags'] = dict(sorted(Counter(f"{r['generated_solution'].count('<think>')} open, "
          f"{r['generated_solution'].count('</think>')} close" for r in rows).items()))
    report['standalone_script_cutoff_3120_chars'] = {'all_above_cutoff': sum(len(r['generated_solution']) > 3120 for r in rows),
          'eligible_above_cutoff': sum(len(r['generated_solution']) > 3120 for r in candidates),
          'interpretation': 'Saved corpus cannot be assumed generated end-to-end by supplied script with its current cutoff.'}
    # Selected reference-format checks evaluate our own scalar formulas only.
    # They do not run supplied programs or certify complete solution text.
    selected = [
        (111, '8*sqrt(2)', 8 * math.sqrt(2), 'radical notation'),
        (408, '27**3', 27 ** 3, 'answer contains an equality'),
        (439, '20*sqrt(5)*pi', 20 * math.sqrt(5) * math.pi, 'implicit multiplication'),
        (729, '1500', 1500, 'unit suffix milligrams'),
        (911, '2.75', 2.75, 'currency marker'),
        (913, '(1/3)**10', (1 / 3) ** 10, 'parenthesized LaTeX power'),
        (927, '180', 180, 'unit suffix seconds'),
        (937, '(3/4)*sqrt(pi)', 0.75 * math.sqrt(math.pi), 'radical and fraction'),
        (991, '3*sqrt(5)/5', 3 * math.sqrt(5) / 5, 'nested radical/fraction'),
        (1001, '36*pi', 36 * math.pi, 'implicit multiplication'),
        (1060, '15*pi', 15 * math.pi, 'implicit multiplication'),
        (1100, '0.476', 0.476, 'percentage as requested percentage-number units'),
    ]
    report['selected_numeric_reference_checks'] = []
    for line, formula, reference_value, note in selected:
        row = rows[line - 1]
        actual = float(row['execution_result'])
        report['selected_numeric_reference_checks'].append({
            'source_line': line, 'recorded_is_correct': row['is_correct'],
            'stored_expected_answer': row['expected_answer'],
            'stored_execution_result': row['execution_result'],
            'independent_reference_formula': formula, 'independent_value': reference_value,
            'abs_difference': abs(actual - reference_value),
            'matches_numeric_reference_within_1e-9': abs(actual - reference_value) < 1e-9,
            'format_issue': note,
            'scope': 'Selected output/reference equivalence only; generated program and proof not executed/verified.'})
    report['selected_reference_precision_issue'] = {
        'source_line': 1122, 'stored_expected_answer': rows[1121]['expected_answer'],
        'stored_execution_result': rows[1121]['execution_result'],
        'independent_formula': 'standard normal CDF at -1 = 0.5*erfc(1/sqrt(2))',
        'independent_value': 0.5 * math.erfc(1 / math.sqrt(2)),
        'interpretation': 'Saved False reflects the rounded 0.15866 reference and strict numeric tolerance; recorded output matches the normal CDF.'}
    report['provenance_fields_absent'] = ['original_dataset_name', 'source_question_id', 'split',
          'source_url', 'license', 'teacher_model_digest', 'prompt_version', 'generation_timestamp', 'generation_seed']
    report['final_token_filter'] = {'established': False,
        'historical_logged_count': 143, 'source_line_ids': None,
        'interpretation': '251-stage candidate overlap is established; final143 membership requires actual historical tokenizer/template filtering.'}
    if args.retained_source_lines:
        retained = json.loads(args.retained_source_lines.read_text(encoding='utf-8'))
        if not isinstance(retained, list) or any(type(i) is not int for i in retained):
            raise ValueError('retained file must be a JSON list of integer source lines')
        retained = set(retained)
        if not retained <= ids:
            raise ValueError('retained source lines must be a subset of historical candidates')
        retained_rows = [rows[i - 1] for i in sorted(retained)]
        report['final_token_filter'] = {'established': True, 'selection_kind': 'reconstructed',
            'original_tokenizer_artifact_authenticated': False, 'source_line_ids': sorted(retained),
            'count': len(retained), 'membership_input_sha256': sha(args.retained_source_lines.read_bytes()),
            'normalized_question_duplicates': duplicate_summary(retained_rows, lambda r: normalize_question(r['problem'])),
            'interpretation': 'Reconstructed selection reproduces historical counts; original Kaggle tokenizer artifact is absent, so byte-identical historical membership is not authenticated.'}
        metadata_path = args.retained_source_lines.parent / 'historical_token_filter.json'
        if metadata_path.exists():
            token_metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            recorded_retained = {r['line'] for r in token_metadata['lengths'] if r['retained']}
            if recorded_retained != retained:
                raise ValueError('Tokenization metadata and retained-source-lines disagree')
            report['final_token_filter']['tokenizer_provenance'] = {
                k: v for k,v in token_metadata.items() if k != 'lengths'}
            report['final_token_filter']['tokenization_metadata_sha256'] = sha(metadata_path.read_bytes())
            report['final_token_filter']['previously_flagged_source616'] = next(
                r for r in token_metadata['lengths'] if r['line'] == 616)
        for name, section in report['overlap'].items():
            for m in section['mappings']:
                m['final_retained_source_lines'] = sorted(set(m['normalized_question_source_rows']) & retained)
            section['final_retained_question_matches'] = sum(bool(m['final_retained_source_lines'])
                                                              for m in section['mappings'])
            section['final_retained_unique_matched_questions'] = len({
                m['question_sha256'] for m in section['mappings'] if m['final_retained_source_lines']})
            section['final_retained_matched_source_count'] = len({
                i for m in section['mappings'] for i in m['final_retained_source_lines']})
    per_row = []
    for i, (row, info) in enumerate(zip(rows, infos), 1):
        per_row.append({'source_line': manifest['distilled']['record_physical_line_or_csv_end'][i-1],
            'row_sha256': sha(json.dumps(row, sort_keys=True, ensure_ascii=False)),
            'question_sha256': sha(row['problem']),
            'normalized_question_sha256': sha(normalize_question(row['problem'])),
            'problem_type': row['problem_type'], 'is_correct': row['is_correct'],
            'eligible_before_token_filter': eligible(row), 'field_characters': {k: len(v) for k,v in row.items() if isinstance(v,str)},
            'code': info})
    (out / 'recovered_data_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (out / 'recovered_data_rows.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in per_row), encoding='utf-8')
    print(json.dumps({'records': len(rows), 'eligible': len(candidates),
        'eligible_unique_questions': report['duplicates']['eligible']['normalized_question']['unique'],
        'reasoning_eval_question_overlap': report['overlap']['reasoning_eval']['eligible_question_matches'],
        'reasoning_eval_exact_four_field_projections': report['reasoning_eval_exact_projection']['count'],
        'historical20_question_overlap': report['overlap']['historical_eval']['eligible_question_matches'],
        'final_token_filter': report['final_token_filter']['established'],
        'reconstructed_retained_records': report['final_token_filter'].get('count'),
        'historical20_reconstructed_training_overlap': report['overlap']['historical_eval'].get('final_retained_question_matches'),
        'original_tokenizer_artifact_authenticated': report['final_token_filter'].get('original_tokenizer_artifact_authenticated', False)}, indent=2))


if __name__ == '__main__':
    main()
