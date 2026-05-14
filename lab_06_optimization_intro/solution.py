from pathlib import Path
import json
import numpy as np
from scipy.optimize import minimize

out_dir = Path(__file__).parent / 'artifacts'
out_dir.mkdir(exist_ok=True)

# Целевая функция

def f(x):
    return 100.0 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

# Ограничения: x0 + x1 >= 1.2, x0 >= 0, x1 >= 0
cons = (
    {'type': 'ineq', 'fun': lambda x: x[0] + x[1] - 1.2},
    {'type': 'ineq', 'fun': lambda x: x[0]},
    {'type': 'ineq', 'fun': lambda x: x[1]},
)

x0 = np.array([0.2, 0.2])

methods = ['SLSQP', 'trust-constr']
results = {}

for m in methods:
    res = minimize(f, x0=x0, method=m, constraints=cons)
    results[m] = {
        'success': bool(res.success),
        'status': int(res.status),
        'message': str(res.message),
        'x': [float(v) for v in res.x],
        'fun': float(res.fun),
        'nit': int(res.nit),
    }

# Без ограничений (BFGS) как базовый сценарий
res_bfgs = minimize(f, x0=x0, method='BFGS')
results['BFGS_unconstrained'] = {
    'success': bool(res_bfgs.success),
    'status': int(res_bfgs.status),
    'message': str(res_bfgs.message),
    'x': [float(v) for v in res_bfgs.x],
    'fun': float(res_bfgs.fun),
    'nit': int(res_bfgs.nit),
}

(out_dir / 'results.json').write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
print('Lab 6 completed. Artifacts in:', out_dir)
