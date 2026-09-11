"""Read every supplied dataset record; no exec/eval or user program execution.

Only a closed arithmetic AST subset is interpreted. All other code is parsed
and inspected statically. Source bytes are never rewritten.
"""
import ast
import collections
import csv
import hashlib
import io
import json
import math
import operator
import re
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'data' / 'raw'
OUT = ROOT / 'reports'
OUT.mkdir(parents=True, exist_ok=True)
BIN = {ast.Add:operator.add, ast.Sub:operator.sub, ast.Mult:operator.mul,
       ast.Div:operator.truediv, ast.FloorDiv:operator.floordiv, ast.Mod:operator.mod,
       ast.Pow:operator.pow, ast.BitAnd:operator.and_, ast.BitOr:operator.or_,
       ast.BitXor:operator.xor, ast.LShift:operator.lshift, ast.RShift:operator.rshift}
UNARY = {ast.USub:operator.neg, ast.UAdd:operator.pos, ast.Invert:operator.invert}
MF = {k:getattr(math,k) for k in ('sqrt','log','log10','sin','cos','tan','factorial','degrees','radians','comb','prod','gcd')}
BUILTINS = {'int':int, 'abs':abs, 'sum':sum, 'len':len, 'min':min, 'max':max, 'pow':pow}

def expr(node, env=None):
    env = env or {}
    if isinstance(node, ast.Constant) and type(node.value) in (int,float,type(None)):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in env: return env[node.id]
        if node.id in ('pi','e'): return getattr(math,node.id)
        raise ValueError('unresolved name '+node.id)
    if isinstance(node, ast.BinOp) and type(node.op) in BIN:
        a,b=expr(node.left,env),expr(node.right,env)
        if isinstance(node.op,(ast.Pow,ast.LShift,ast.RShift)) and abs(b)>10000:
            raise ValueError('arithmetic resource bound')
        return BIN[type(node.op)](a,b)
    if isinstance(node,ast.UnaryOp) and type(node.op) in UNARY:
        return UNARY[type(node.op)](expr(node.operand,env))
    if isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name) and node.value.id=='math' and node.attr in ('pi','e'):
        return getattr(math,node.attr)
    if isinstance(node,(ast.List,ast.Tuple)):
        return [expr(x,env) for x in node.elts]
    if isinstance(node,ast.Subscript):
        return expr(node.value,env)[expr(node.slice,env)]
    if isinstance(node,ast.Call) and not node.keywords:
        func = None
        if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='math':
            func=MF.get(node.func.attr)
        elif isinstance(node.func,ast.Name):
            func=BUILTINS.get(node.func.id) or MF.get(node.func.id)
        if func:
            return func(*[expr(x,env) for x in node.args])
    raise ValueError('unsupported arithmetic AST: '+type(node).__name__)

def simple_return(code):
    tree=ast.parse(code)
    solves=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='solve']
    if len(solves)!=1: raise ValueError('solve definition count != 1')
    env={}
    for node in solves[0].body:
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):
            continue
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            mods=[x.name for x in node.names] if isinstance(node,ast.Import) else [node.module]
            if mods!=['math']: raise ValueError('non-math import')
            continue
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            env[node.targets[0].id]=expr(node.value,env)
        elif isinstance(node,ast.Return): return expr(node.value,env)
        else: raise ValueError('unsupported function statement: '+type(node).__name__)
    raise ValueError('missing return')

def numeric(text):
    if re.fullmatch(r'[+-]?\d+',text): return int(text)
    return float(text)

def equal(a,b):
    if type(a)==int and type(b)==int: return a==b
    return math.isclose(a,b,rel_tol=0,abs_tol=1e-9)

def read(name):
    raw=(SRC/name).read_bytes();text=raw.decode('utf-8-sig')
    positions=[]
    if name.endswith('.jsonl'):
        rows=[]
        for n,line in enumerate(text.splitlines(),1):
            if not line.strip(): raise ValueError('blank JSONL row')
            rows.append(json.loads(line));positions.append((n,n))
    else:
        reader=csv.DictReader(io.StringIO(text));rows=[]
        reader.fieldnames;prev=reader.line_num
        for row in reader:
            rows.append(row);positions.append((prev+1,reader.line_num));prev=reader.line_num
    return rows,positions,dict(name=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),physical_lines=len(text.splitlines()),rows=len(rows))

def normalize(text):
    return re.sub(r'\s+',' ',text.replace('$Calculate#','Calculate')).strip().casefold()

datasets={}
manifest=[]
rowaudits=[]
for name,pk,ck,ak in [
    ('arithmetic_logic_data_revised_v2.jsonl','problem','generated_python_code','generated_solution'),
    ('arithmetic_problems_v3.csv','arithmetical question','code for solving the question','answer to the question'),
    ('random_samples_for_inference.csv','arithmetical question','generated_python_code','answer to the question')]:
    rows,positions,meta=read(name);datasets[name]=(rows,pk,ck)
    signatures=collections.defaultdict(list);full=collections.defaultdict(list)
    issues=[];imports=collections.Counter();constant_only=[];simple_result=collections.Counter();operators=collections.Counter()
    for n,(row,(start,end)) in enumerate(zip(rows,positions),1):
        assert None not in row and all(x is not None for x in row.values())
        signatures[normalize(row[pk])].append(n)
        full[json.dumps(row,sort_keys=True,ensure_ascii=False)].append(n)
        rec=dict(file=name,row=n,physical_start=start,physical_end=end,question=row[pk],code_sha256=hashlib.sha256(row[ck].encode()).hexdigest())
        tree=ast.parse(row[ck]);rec['syntax_valid']=True
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):imports.update(x.name for x in node.names)
            if isinstance(node,ast.ImportFrom):imports[node.module]+=1
        ops=sorted(set(type(x.op).__name__ for x in ast.walk(tree) if isinstance(x,(ast.BinOp,ast.UnaryOp))))
        operators.update(ops);rec['operators']=ops
        solves=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='solve']
        if len(solves)==1 and len(solves[0].body)==1 and isinstance(solves[0].body[0],ast.Return) and isinstance(solves[0].body[0].value,ast.Constant):constant_only.append(n)
        rec['solve_definition_count']=len(solves)
        try:
            calculated=simple_return(row[ck]);rec['arithmetic_ast_result']=str(calculated)
            answer=row[ak]
            if name.endswith('jsonl'):
                m=re.fullmatch(r'The answer is (.+)\.',answer)
                if not m:
                    rec['reference_match']='no_numeric_reference';simple_result['no_numeric_reference']+=1
                else:
                    given=numeric(m.group(1));match=equal(calculated,given);rec['reference_match']=match;simple_result['matched' if match else 'mismatched']+=1
            else:
                given=numeric(answer);match=equal(calculated,given);rec['reference_match']=match;simple_result['matched' if match else 'mismatched']+=1
            if rec.get('reference_match') is False:
                issues.append(dict(row=n,physical_start=start,physical_end=end,issue='code_reference_mismatch',calculated=str(calculated),stored_answer=answer,question=row[pk]))
        except (ValueError,TypeError,KeyError,ZeroDivisionError,OverflowError) as e:
            rec['arithmetic_ast_status']=str(e);simple_result['not_evaluated']+=1
        if name=='arithmetic_problems_v3.csv':
            q=row[pk];s=q.removeprefix('Calculate the value of ')
            for k,v in [('XOR','^'),('AND','&'),('OR','|')]:s=s.replace(k,v)
            q_value=expr(ast.parse(s,mode='eval').body)
            rec['question_expression_result']=str(q_value)
            if not equal(q_value,numeric(row[ak])):
                rec['question_reference_match']=False
                given=numeric(row[ak])
                issues.append(dict(row=n,physical_start=start,physical_end=end,issue='question_reference_mismatch',question=q,question_result=str(q_value),stored_answer=row[ak],absolute_difference=str(abs(q_value-given)),relative_difference=abs(q_value-given)/max(abs(q_value),abs(given)),passes_paired_cot_isclose_rel_tol_1e_3_abs_tol_0=math.isclose(q_value,given,rel_tol=1e-3,abs_tol=0),passes_batch_generator_abs_tol_1e_9=abs(q_value-given)<1e-9))
            else:rec['question_reference_match']=True
        if 'generated_solution' in row:
            s=row['generated_solution'];rec['solution_chars']=len(s);rec['solution_sha256']=hashlib.sha256(s.encode()).hexdigest()
            rec['think_open_count']=s.count('<think>');rec['think_close_count']=s.count('</think>')
            rec['boxed_count']=s.count('\\boxed')
            rec['placeholder_solution']=s.strip()=='...'
        rowaudits.append(rec)
    meta.update(schema=list(rows[0]),empty_fields={k:sum(not str(r[k]).strip() for r in rows) for k in rows[0]},
        unique_questions=len(signatures),duplicate_question_groups=[v for v in signatures.values() if len(v)>1],
        duplicate_full_record_groups=[v for v in full.values() if len(v)>1],
        syntax_valid_rows=len(rows),imports=dict(imports),constant_only_solve_rows=constant_only,
        closed_ast_numeric_checks=dict(simple_result),issues=issues,operator_row_counts=dict(operators))
    if 'problem_type' in rows[0]:meta['problem_type_counts']=dict(collections.Counter(x['problem_type'] for x in rows))
    if 'is_correct' in rows[0]:meta['is_correct_counts']=dict(collections.Counter(str(x['is_correct']) for x in rows))
    if 'generated_solution' in rows[0]:
        lengths=sorted(len(x['generated_solution']) for x in rows)
        meta['solution_chars']={'min':min(lengths),'median':(lengths[(len(lengths)-1)//2]+lengths[len(lengths)//2])/2,'max':max(lengths),'total':sum(lengths)}
        meta['placeholder_solution_rows']=[i for i,x in enumerate(rows,1) if x['generated_solution'].strip()=='...']
    manifest.append(meta)

overlap=[]
names=list(datasets)
for i,na in enumerate(names):
    ra,pa,ca=datasets[na]
    for nb in names[i+1:]:
        rb,pb,cb=datasets[nb]
        qa={normalize(x[pa]) for x in ra};qb={normalize(x[pb]) for x in rb}
        codesa={ast.dump(ast.parse(x[ca]),include_attributes=False) for x in ra};codesb={ast.dump(ast.parse(x[cb]),include_attributes=False) for x in rb}
        overlap.append({'a':na,'b':nb,'normalized_question_overlap':len(qa&qb),'code_ast_overlap':len(codesa&codesb)})

scripts=[]
for name in ('add_more_data.py','generate_arithmetic_data.py','batch_generate_and_verify_v3_1.py'):
    raw=(ROOT/'archive'/'scripts'/name).read_bytes();s=raw.decode('utf-8');ast.parse(s)
    scripts.append(dict(name=name,bytes=len(raw),lines=len(s.splitlines()),sha256=hashlib.sha256(raw).hexdigest(),syntax_valid=True))

report={'datasets':manifest,'cross_file_overlap':overlap,'scripts':scripts,'environment':{'python_version':sys.version.split()[0],'os':platform.system(),'machine':platform.machine(),'dependencies':'Python standard library only'},'method':'All source bytes decoded and all records parsed. Closed AST interpreter for arithmetic expressions only; generated Python was not exec/eval or imported. Complex reference programs statically inspected.','tolerance_notes':{'this_audit':'Exact integer equality; otherwise abs_tol=1e-9, rel_tol=0.','batch_generate_and_verify_v3_1.py':'compare_answers line233 uses sympy Abs(result_val-expected_val)<1e-9, after evalf(30); this audit does not execute that parser.','paired_cot_notebooks':'math.isclose(result_decimal, ground_truth_decimal, rel_tol=1e-3), default abs_tol=0; all14 displayed-input discrepancies remain accepted under that tolerance.','no_metric_inference':'Source data inconsistencies do not themselves establish a change in historical model accuracy.'}}
(OUT/'data_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n', encoding='utf-8')
with (OUT/'row_audit.jsonl').open('w', encoding='utf-8') as f:
    for r in rowaudits:f.write(json.dumps(r,ensure_ascii=False)+'\n')

# Independent rotation check: derive results from the question using strings.
# No AST interpreter or provided solve() formula is used for this check.
rotations=[]
for n,row in enumerate(datasets['arithmetic_logic_data_revised_v2.jsonl'][0],1):
    q=row['problem']
    if not any(w in q.lower() for w in ('cyclic','circular','rotate')):continue
    width=int(re.search(r'(\d+)-bit',q).group(1))
    h=re.search(r'0x[0-9a-fA-F]+',q).group(0)
    k=int(re.search(r'by (\d+)',q).group(1))%width
    direction='left' if 'left' in q else 'right'
    bits=format(int(h,16),f'0{width}b')
    rotated=(bits[k:]+bits[:k]) if direction=='left' else (bits[-k:]+bits[:-k] if k else bits)
    result=int(rotated,2)
    given=int(re.fullmatch(r'The answer is (.+)\.',row['generated_solution']).group(1))
    rotations.append(dict(row=n,width=width,input_hex=h,shift=k,direction=direction,binary_string_rotation_result=str(result),stored_solution_answer=str(given),matches=given==result))
(OUT/'rotation_independent_checks.json').write_text(json.dumps({'method':'Fixed-width binary-string rotation independently derived from each question; supplied code not executed.','rows':rotations},indent=2)+'\n')

# Row 43: independently written numerical counterexample with fixed brackets.
def f43(x):return math.asinh(x)+math.sin(x)
def bisect43(lo,hi):
    if (f43(lo)-5)*(f43(hi)-5)>=0:raise ValueError('invalid root bracket')
    for _ in range(60):
        mid=(lo+hi)/2
        if (f43(lo)-5)*(f43(mid)-5)<=0:hi=mid
        else:lo=mid
    return (lo+hi)/2
x1,x2=bisect43(32,33),bisect43(33,34)
a,b=x1-2,7-x2
counterexample={'function':'f(x)=log(sqrt(x^2+1)+x)+sin(x), log=natural logarithm','equivalent_stable_form':'asinh(x)+sin(x)','method':'Bisection of independently written formula; fixed brackets [32,33] and [33,34], 60 iterations.','roots_of_f_equals_5':[x1,x2],'counterexample':{'a':a,'b':b,'f(a+2)':f43(a+2),'f(b-7)':f43(b-7),'a_plus_b':a+b,'claimed_answer':5},'derivative':'1/sqrt(x^2+1)+cos(x), which takes negative values; oddness does not imply injectivity.'}
(OUT/'row43_counterexample.json').write_text(json.dumps(counterexample,indent=2)+'\n')
print('Audited every record in 3 datasets. Details: reports/data_audit.json and reports/row_audit.jsonl')
