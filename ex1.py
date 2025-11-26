import numpy as np
import matplotlib.pyplot as plt

print("=== ЗАПУСК ОПТИМІЗАЦІЙНИХ ЗАДАЧ ===")

# === 1. SciPy.optimize ===
try:
    from scipy.optimize import minimize
    print("\n1. SciPy.optimize - Запуск...")
    
    def production_cost(x):
        material_cost = 100 * x[0]
        energy_cost = 50 * x[1] + 30 * x[2]
        quality_penalty = 200 * np.exp(-0.1 * x[1]) + 150 * np.exp(-0.05 * x[2])
        return material_cost + energy_cost + quality_penalty

    constraints = [
        {'type': 'ineq', 'fun': lambda x: x[0] - 10},
        {'type': 'ineq', 'fun': lambda x: 100 - x[0]},
    ]
    bounds = [(10, 100), (50, 300), (1, 10)]
    result = minimize(production_cost, [50, 150, 5], method='SLSQP', bounds=bounds, constraints=constraints)
    
    print("SciPy.optimize - Успішно")
    print(f"Оптимальна швидкість: {result.x[0]:.2f} од/год")
    print(f"Оптимальна температура: {result.x[1]:.2f} °C")
    print(f"Оптимальний тиск: {result.x[2]:.2f} бар")
    print(f"Мінімальні витрати: {result.fun:.2f} грошових одиниць")
except Exception as e:
    print(f"SciPy.optimize - Помилка: {e}")

# === 2. CVXPY ===
try:
    import cvxpy as cp
    print("\n2. CVXPY - Запуск...")
    
    expected_returns = np.array([0.12, 0.08, 0.15, 0.06])
    covariance_matrix = np.array([
        [0.1, 0.02, 0.03, 0.01],
        [0.02, 0.05, 0.01, 0.005],
        [0.03, 0.01, 0.15, 0.02],
        [0.01, 0.005, 0.02, 0.03]
    ])
    
    weights = cp.Variable(len(expected_returns))
    portfolio_risk = cp.quad_form(weights, covariance_matrix)
    constraints = [cp.sum(weights) == 1, weights >= 0]
    problem = cp.Problem(cp.Minimize(portfolio_risk), constraints)
    problem.solve()
    
    print("CVXPY - Успішно")
    assets = ['Акції IT', 'Облігації', 'Акції енергетики', 'Депозити']
    for asset, weight in zip(assets, weights.value):
        print(f"{asset}: {weight*100:.2f}%")
    print(f"Ризик портфеля (дисперсія): {portfolio_risk.value:.4f}")
except Exception as e:
    print(f"CVXPY - Помилка: {e}")

# === 3. PuLP ===
try:
    from pulp import *
    print("\n3. PuLP - Запуск...")
    
    prob = LpProblem("Logistics_Optimization", LpMinimize)
    x11 = LpVariable("Warehouse1_Store1", 0, None, LpInteger)
    x12 = LpVariable("Warehouse1_Store2", 0, None, LpInteger)
    x21 = LpVariable("Warehouse2_Store1", 0, None, LpInteger)
    x22 = LpVariable("Warehouse2_Store2", 0, None, LpInteger)
    
    prob += 5*x11 + 7*x12 + 6*x21 + 4*x22, "Total_Transportation_Cost"
    prob += x11 + x21 >= 100, "Store1_Demand"
    prob += x12 + x22 >= 150, "Store2_Demand"
    prob += x11 + x12 <= 120, "Warehouse1_Capacity"
    prob += x21 + x22 <= 200, "Warehouse2_Capacity"
    prob.solve()
    
    print("PuLP - Успішно")
    print(f"Статус: {LpStatus[prob.status]}")
    print(f"Вантаж зі складу 1 до магазину 1: {x11.varValue} од.")
    print(f"Вантаж зі складу 1 до магазину 2: {x12.varValue} од.")
    print(f"Вантаж зі складу 2 до магазину 1: {x21.varValue} од.")
    print(f"Вантаж зі складу 2 до магазину 2: {x22.varValue} од.")
    print(f"Мінімальні транспортні витрати: {value(prob.objective)} грошових одиниць")
except Exception as e:
    print(f"PuLP - Помилка: {e}")

# === 4. GEKKO ===
try:
    from gekko import GEKKO
    print("\n4. GEKKO - Запуск...")
    
    m = GEKKO(remote=False)
    temperature = m.Var(value=300, lb=250, ub=400)
    pressure = m.Var(value=10, lb=5, ub=20)
    residence_time = m.Var(value=10, lb=5, ub=30)
    
    conversion = m.Intermediate(0.8 * (1 - m.exp(-0.1 * temperature/300)))
    selectivity = m.Intermediate(0.9 * (1 - m.exp(-0.05 * pressure)))
    product_yield = conversion * selectivity * residence_time
    
    energy_consumption = m.Intermediate(2 * temperature + 5 * pressure)
    
    m.Maximize(product_yield)
    m.Equation(energy_consumption <= 1000)
  m.solve(disp=False)
    
    print("GEKKO - Успішно")
    print(f"Оптимальна температура: {temperature.value[0]:.2f} K")
    print(f"Оптимальний тиск: {pressure.value[0]:.2f} атм")
    print(f"Оптимальний час перебування: {residence_time.value[0]:.2f} хв")
    print(f"Максимальний вихід продукту: {product_yield.value[0]:.4f}")
except Exception as e:
    print(f"GEKKO - Помилка: {e}")

# === 5. Pyomo ===
try:
    from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory
    print("\n5. Pyomo - Запуск...")
    
    model = ConcreteModel()
    hours = range(6)  # Скорочено для прикладу
    
    model.heating = Var(hours, bounds=(0, 50))
    model.cooling = Var(hours, bounds=(0, 40))
    
    electricity_price = [0.08 if 7 <= h <= 22 else 0.05 for h in hours]
    
    model.cost = Objective(
        expr=sum(electricity_price[h] * (model.heating[h] + model.cooling[h]) 
                for h in hours)
    )
    
    def comfort_rule(model, h):
        return model.heating[h] >= 20 and model.cooling[h] >= 15
    
    model.comfort = Constraint(hours, rule=comfort_rule)
    
    solver = SolverFactory('glpk')
    results = solver.solve(model)
    
    print("Pyomo - Успішно")
    total_cost = sum(electricity_price[h] * (model.heating[h].value + model.cooling[h].value) 
                    for h in hours)
    print(f"Загальні витрати на енергію: {total_cost:.2f} грошових одиниць")
    print("Годинне споживання:")
    for h in hours:
        print(f"Година {h}: обігрів={model.heating[h].value:.1f}кВт, охолодження={model.cooling[h].value:.1f}кВт")
        
except Exception as e:
    print(f"Pyomo - Помилка: {e}")

print("\n=== ВИКОНАННЯ ЗАВЕРШЕНО ===")
