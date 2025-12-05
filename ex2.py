import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, linprog
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ProductionOptimizer:
    """
    Клас для оптимізації виробничих процесів та параметрів
    """
    
    def __init__(self):
        """Ініціалізація оптимізатора"""
        self.data = None
        self.optimization_results = {}
        self.config = self._load_config()
        
    def _load_config(self):
        """Завантаження конфігурації з файлу або створення за замовчуванням"""
        config = {
            'cost_coefficients': [100, 50, 30],  # коефіцієнти витрат
            'price_coefficients': [150, 80, 40],  # коефіцієнти цін
            'quality_params': [0.1, 0.05],        # параметри якості
            'bounds': [(10, 100), (50, 300), (1, 10)],  # межі змінних
            'constraints': [                       # обмеження
                {'type': 'ineq', 'name': 'min_speed', 'value': 10},
                {'type': 'ineq', 'name': 'max_speed', 'value': 100},
                {'type': 'ineq', 'name': 'min_temp', 'value': 50},
                {'type': 'ineq', 'name': 'max_temp', 'value': 300},
                {'type': 'ineq', 'name': 'budget', 'value': 50000}
            ]
        }
        return config
    
    def load_data(self, filename=None):
        """
        Завантаження даних з файлу або створення тестових даних
        
        Parameters:
        filename (str): Шлях до файлу з даними
        
        Returns:
        bool: Статус успішності завантаження
        """
        try:
            if filename and os.path.exists(filename):
                print(f"Завантаження даних з файлу: {filename}")
                self.data = pd.read_csv(filename)
                print(f"Дані успішно завантажені: {len(self.data)} рядків")
            else:
                print("Створення тестових даних...")
                self._generate_test_data()
            
            return True
            
        except Exception as e:
            print(f"Помилка при завантаженні даних: {e}")
            return False
    
    def _generate_test_data(self):
        """Генерація тестових даних для демонстрації"""
        np.random.seed(42)
        n_samples = 100
        
        # Генерація виробничих даних
        self.data = pd.DataFrame({
            'production_speed': np.random.uniform(10, 100, n_samples),
            'temperature': np.random.uniform(50, 300, n_samples),
            'pressure': np.random.uniform(1, 10, n_samples),
            'material_cost': np.random.uniform(800, 1200, n_samples),
            'energy_cost': np.random.uniform(400, 800, n_samples),
            'quality_score': np.random.uniform(0.7, 1.0, n_samples),
            'profit': np.random.uniform(-50, 500, n_samples)
        })
        
        print(f"Створено тестові дані: {n_samples} записів")
    
    def production_cost_function(self, x, user_params=None):
        """
        Функція витрат виробництва
        
        Parameters:
        x (array): Вектор параметрів [швидкість, температура, тиск]
        user_params (dict): Користувацькі параметри
        
        Returns:
        float: Значення функції витрат
        """
        try:
            # Використання користувацьких параметрів або параметрів за замовчуванням
            if user_params:
                cost_coeff = user_params.get('cost_coefficients', self.config['cost_coefficients'])
                quality_params = user_params.get('quality_params', self.config['quality_params'])
            else:
                cost_coeff = self.config['cost_coefficients']
                quality_params = self.config.get('quality_params', [0.1, 0.05])
            
            # Розрахунок витрат
            material_cost = cost_coeff[0] * x[0]
            energy_cost = cost_coeff[1] * x[1] + cost_coeff[2] * x[2]
            
            # Штраф за якість
            quality_penalty = 200 * np.exp(-quality_params[0] * x[1]) + \
                            150 * np.exp(-quality_params[1] * x[2])
            
            total_cost = material_cost + energy_cost + quality_penalty
            
            return total_cost
            
        except Exception as e:
            print(f"Помилка у функції витрат: {e}")
            return float('inf')
    
    def profit_function(self, x, user_params=None):
        """
        Функція прибутку (для максимізації)
        
        Parameters:
        x (array): Вектор параметрів
        user_params (dict): Користувацькі параметри
        
        Returns:
        float: Значення прибутку (негативне для мінімізації)
        """
        try:
            if user_params:
                price_coeff = user_params.get('price_coefficients', self.config['price_coefficients'])
            else:
                price_coeff = self.config['price_coefficients']
            
            # Дохід від продажу
            revenue = price_coeff[0] * x[0] + price_coeff[1] * x[1] + price_coeff[2] * x[2]
            
            # Витрати
            cost = self.production_cost_function(x, user_params)
            
            # Прибуток (негативний для мінімізації)
            profit = -(revenue - cost)
            
            return profit
            
        except Exception as e:
            print(f"Помилка у функції прибутку: {e}")
            return float('inf')
    
    def create_constraints(self, constraint_type='default'):
        """
        Створення обмежень для оптимізації
        
        Parameters:
        constraint_type (str): Тип обмежень
        
        Returns:
        list: Список обмежень
        """
        constraints = []
        
        if constraint_type == 'default':
            # Технологічні обмеження
            constraints.append({'type': 'ineq', 'fun': lambda x: x[0] - 10})      # мінімальна швидкість
            constraints.append({'type': 'ineq', 'fun': lambda x: 100 - x[0]})    # максимальна швидкість
            constraints.append({'type': 'ineq', 'fun': lambda x: x[1] - 50})     # мінімальна температура
            constraints.append({'type': 'ineq', 'fun': lambda x: 300 - x[1]})    # максимальна температура
        
        elif constraint_type == 'budget':
            # Бюджетні обмеження
            constraints.append({'type': 'ineq', 'fun': lambda x: 50000 - self.production_cost_function(x)})
        
        elif constraint_type == 'quality':
            # Обмеження за якістю
            constraints.append({'type': 'ineq', 'fun': lambda x: 0.8 - np.exp(-0.1 * x[1])})
            constraints.append({'type': 'ineq', 'fun': lambda x: 0.9 - np.exp(-0.05 * x[2])})
        
        return constraints
    
    def optimize_production(self, objective='cost', method='SLSQP', user_params=None):
        """
        Оптимізація виробничих параметрів
        
        Parameters:
        objective (str): Цільова функція ('cost' або 'profit')
        method (str): Метод оптимізації
        user_params (dict): Користувацькі параметри
        
        Returns:
        dict: Результати оптимізації
        """
        print(f"\n{'='*60}")
        print(f"ПОЧАТОК ОПТИМІЗАЦІЇ")
        print(f"Ціль: {objective}, Метод: {method}")
        print(f"{'='*60}")
        
        try:
            # Вибір цільової функції
            if objective == 'cost':
                objective_function = lambda x: self.production_cost_function(x, user_params)
                print("Ціль: мінімізація витрат")
            elif objective == 'profit':
                objective_function = lambda x: self.profit_function(x, user_params)
                print("Ціль: максимізація прибутку")
            else:
                raise ValueError(f"Невідома цільова функція: {objective}")
            
            # Початкове наближення
            x0 = [50, 150, 5]  # середні значення
            
            # Створення обмежень
            constraints = self.create_constraints('default')
            
            # Виконання оптимізації
            print("Виконання оптимізації...")
            result = minimize(
                objective_function,
                x0,
                method=method,
                bounds=self.config['bounds'],
                constraints=constraints,
                options={'disp': True, 'maxiter': 1000}
            )
            
            if result.success:
                print(f"\n✓ ОПТИМІЗАЦІЯ УСПІШНА")
                
                # Збереження результатів
                self.optimization_results = {
                    'success': True,
                    'objective': objective,
                    'method': method,
                    'optimal_values': result.x.tolist(),
                    'optimal_function_value': result.fun,
                    'iterations': result.nit,
                    'message': result.message
                }
                
                # Розрахунок додаткових показників
                optimal_cost = self.production_cost_function(result.x)
                revenue = -(result.fun) + optimal_cost if objective == 'profit' else 0
                
                self.optimization_results.update({
                    'production_cost': optimal_cost,
                    'revenue': revenue,
                    'profit': revenue - optimal_cost if objective == 'profit' else -optimal_cost
                })
                
            else:
                print(f"\n ОПТИМІЗАЦІЯ НЕ ВДАЛАСЬ: {result.message}")
                self.optimization_results = {
                    'success': False,
                    'message': result.message
                }
            
            return self.optimization_results
            
        except Exception as e:
            print(f"\n ПОМИЛКА ПРИ ОПТИМІЗАЦІЇ: {e}")
            return {'success': False, 'error': str(e)}
    
    def linear_optimization(self):
        """
        Лінійна оптимізація за допомогою linprog
        """
        print("\n" + "="*60)
        print("ЛІНІЙНА ОПТИМІЗАЦІЯ (ПЛАНОВЕ ЗАВДАННЯ)")
        print("="*60)
        
        try:
            # Коефіцієнти цільової функції (мінімізація витрат)
            c = [-150, -80, -40]  # негативні для максимізації доходу
            
            # Коефіцієнти обмежень
            A = [
                [1, 0, 0],    # мінімальна швидкість
                [-1, 0, 0],   # максимальна швидкість
                [0, 1, 0],    # мінімальна температура
                [0, -1, 0],   # максимальна температура
                [100, 50, 30] # бюджетне обмеження
            ]
            b = [10, -100, 50, -300, 50000]
            
            # Межі змінних
            bounds = self.config['bounds']
            
            # Виконання лінійної оптимізації
            result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
            
            if result.success:
                print(" Лінійна оптимізація успішна")
                
                linear_results = {
                    'optimal_values': result.x.tolist(),
                    'optimal_profit': -result.fun,
                    'status': result.message
                }
                
                print(f"Оптимальні значення: {result.x}")
                print(f"Максимальний прибуток: {-result.fun:.2f}")
                return linear_results
            else:
                print(f" Лінійна оптимізація не вдалась: {result.message}")
                return None
                
        except Exception as e:
            print(f" Помилка при лінійній оптимізації: {e}")
            return None
    
    def analyze_results(self):
        """Аналіз та візуалізація результатів"""
        if not self.optimization_results.get('success'):
            print("Немає результатів для аналізу")
            return
        
        print("\n" + "="*60)
        print("АНАЛІЗ РЕЗУЛЬТАТІВ ОПТИМІЗАЦІЇ")
        print("="*60)
        
        # Вивід результатів
        params = ['Швидкість', 'Температура', 'Тиск']
        units = ['од/год', '°C', 'бар']
        optimal = self.optimization_results['optimal_values']
        
        print("\nОПТИМАЛЬНІ ПАРАМЕТРИ:")
        for i, (param, unit, value) in enumerate(zip(params, units, optimal)):
            print(f"  {param}: {value:.2f} {unit}")
        
        print(f"\nВИТРАТИ НА ВИРОБНИЦТВО: {self.optimization_results['production_cost']:.2f}")
        
        if self.optimization_results['objective'] == 'profit':
            print(f"ДОХІД: {self.optimization_results['revenue']:.2f}")
            print(f"ПРИБУТОК: {self.optimization_results['profit']:.2f}")
        
        print(f"\nІТЕРАЦІЙ: {self.optimization_results['iterations']}")
        
        # Візуалізація
        self._visualize_results()
    
    def _visualize_results(self):
        """Візуалізація результатів оптимізації"""
        if not self.optimization_results.get('success'):
            return
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Результати оптимізації виробничих параметрів', fontsize=16)
            
            # 1. Оптимальні параметри
            params = ['Швидкість', 'Температура', 'Тиск']
            optimal = self.optimization_results['optimal_values']
            
            axes[0, 0].bar(params, optimal, color=['blue', 'red', 'green'])
            axes[0, 0].set_title('Оптимальні значення параметрів')
            axes[0, 0].set_ylabel('Значення')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Вплив параметрів на витрати
            if self.data is not None:
                sample_size = min(50, len(self.data))
                sample = self.data.sample(sample_size)
                
                axes[0, 1].scatter(sample['production_speed'], sample['material_cost'], 
                                  alpha=0.6, label='Матеріали')
                axes[0, 1].scatter(sample['temperature'], sample['energy_cost'], 
                                  alpha=0.6, label='Енергія', color='red')
                axes[0, 1].set_title('Вплив параметрів на витрати')
                axes[0, 1].set_xlabel('Параметри')
                axes[0, 1].set_ylabel('Витрати')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Порівняння економічних показників
            if self.optimization_results['objective'] == 'profit':
                indicators = ['Витрати', 'Дохід', 'Прибуток']
                values = [
                    self.optimization_results['production_cost'],
                    self.optimization_results['revenue'],
                    self.optimization_results['profit']
                ]
                
                axes[1, 0].bar(indicators, values, color=['red', 'green', 'blue'])
                axes[1, 0].set_title('Економічні показники')
                axes[1, 0].set_ylabel('Грошові одиниці')
                axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Просторова візуалізація
            x = np.linspace(10, 100, 20)
            y = np.linspace(50, 300, 20)
            X, Y = np.meshgrid(x, y)
            
            # Функція витрат для двох змінних
            Z = np.array([[self.production_cost_function([xi, yi, 5]) 
                          for xi in x] for yi in y])
            
            contour = axes[1, 1].contourf(X, Y, Z, levels=20, cmap='viridis')
            axes[1, 1].scatter(optimal[0], optimal[1], color='red', s=100, 
                              marker='*', label='Оптимум')
            axes[1, 1].set_title('Поверхня витрат')
            axes[1, 1].set_xlabel('Швидкість (од/год)')
            axes[1, 1].set_ylabel('Температура (°C)')
            axes[1, 1].legend()
            plt.colorbar(contour, ax=axes[1, 1])
            
            plt.tight_layout()
            
            # Збереження графіка
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"optimization_results_{timestamp}"
            
            # Збереження у PNG
            plt.savefig(f"{base_filename}.png", dpi=300, bbox_inches='tight')
            
            # Збереження у PDF
            plt.savefig(f"{base_filename}.pdf", bbox_inches='tight')
            
            # Збереження у SVG
            plt.savefig(f"{base_filename}.svg", format='svg', bbox_inches='tight')
            
            print(f"\nГрафік збережено у файлах:")
            print(f"  • {base_filename}.png (зображення)")
            print(f"  • {base_filename}.pdf (документ)")
            print(f"  • {base_filename}.svg (векторний графік)")
            
            plt.show()
            
        except Exception as e:
            print(f"Помилка при візуалізації: {e}")
    
    def save_results(self, filename=None):
        """
        Збереження результатів у файл
        
        Parameters:
        filename (str): Ім'я файлу для збереження
        """
        if not self.optimization_results.get('success'):
            print("Немає результатів для збереження")
            return False
        
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"optimization_report_{timestamp}"
            
            # Збереження у JSON
            json_filename = f"{filename}.json"
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(self.optimization_results, f, indent=2, ensure_ascii=False)
            
            # Збереження у TXT
            txt_filename = f"{filename}.txt"
            with open(txt_filename, 'w', encoding='utf-8') as f:
                f.write("="*60 + "\n")
                f.write("ЗВІТ ПРО ОПТИМІЗАЦІЮ ВИРОБНИЦТВА\n")
                f.write("="*60 + "\n\n")
                
                f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Ціль: {self.optimization_results.get('objective', 'N/A')}\n")
                f.write(f"Метод: {self.optimization_results.get('method', 'N/A')}\n\n")
                
                f.write("ОПТИМАЛЬНІ ПАРАМЕТРИ:\n")
                f.write("-"*40 + "\n")
                params = ['Швидкість (од/год)', 'Температура (°C)', 'Тиск (бар)']
                optimal = self.optimization_results['optimal_values']
                for param, value in zip(params, optimal):
                    f.write(f"{param}: {value:.2f}\n")
                
                f.write("\nЕКОНОМІЧНІ ПОКАЗНИКИ:\n")
                f.write("-"*40 + "\n")
                f.write(f"Витрати: {self.optimization_results.get('production_cost', 0):.2f}\n")
                
                if self.optimization_results.get('revenue'):
                    f.write(f"Дохід: {self.optimization_results.get('revenue', 0):.2f}\n")
                    f.write(f"Прибуток: {self.optimization_results.get('profit', 0):.2f}\n")
                
                f.write(f"\nІтерації: {self.optimization_results.get('iterations', 0)}\n")
                f.write(f"Статус: {self.optimization_results.get('message', 'N/A')}\n")
            
            print(f"\nРезультати збережено у файлах:")
            print(f"  • {json_filename}")
            print(f"  • {txt_filename}")
            
            return True
            
        except Exception as e:
            print(f"Помилка при збереженні результатів: {e}")
            return False

    def save_to_csv(self, filename=None):
        """
        Збереження результатів та даних у CSV файл
        """
        if not self.optimization_results.get('success'):
            print("Немає результатів для збереження у CSV")
            return False
        
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"optimization_data_{timestamp}"
            
            # 1. Збереження оптимальних параметрів у CSV
            params_df = pd.DataFrame({
                'Параметр': ['Швидкість', 'Температура', 'Тиск'],
                'Значення': self.optimization_results['optimal_values'],
                'Одиниці': ['од/год', '°C', 'бар']
            })
            params_df.to_csv(f"{filename}_parameters.csv", index=False, encoding='utf-8-sig')
            
            # 2. Збереження економічних показників у CSV
            economics_df = pd.DataFrame({
                'Показник': ['Витрати', 'Дохід', 'Прибуток', 'Ціль', 'Метод', 'Ітерації'],
                'Значення': [
                    self.optimization_results.get('production_cost', 0),
                    self.optimization_results.get('revenue', 0),
                    self.optimization_results.get('profit', 0),
                    self.optimization_results.get('objective', 'N/A'),
                    self.optimization_results.get('method', 'N/A'),
                    self.optimization_results.get('iterations', 0)
                ]
            })
            economics_df.to_csv(f"{filename}_economics.csv", index=False, encoding='utf-8-sig')
            
            # 3. Якщо є оригінальні дані, зберігаємо їх теж
            if self.data is not None:
                self.data.to_csv(f"{filename}_raw_data.csv", index=False, encoding='utf-8-sig')
            
            print(f"Дані збережено у CSV файлах:")
            print(f"  • {filename}_parameters.csv")
            print(f"  • {filename}_economics.csv")
            if self.data is not None:
                print(f"  • {filename}_raw_data.csv")
            
            return True
            
        except Exception as e:
            print(f"Помилка при збереженні у CSV: {e}")
            return False
        
    def display_menu(self):
        """Відображення головного меню"""
        print("\n" + "="*60)
        print("СИСТЕМА ОПТИМІЗАЦІЇ ВИРОБНИЧИХ ПАРАМЕТРІВ")
        print("="*60)
        print("1. Завантажити дані")
        print("2. Оптимізувати витрати")
        print("3. Оптимізувати прибуток")
        print("4. Лінійна оптимізація")
        print("5. Аналіз результатів")
        print("6. Зберегти результати (JSON/TXT)")
        print("7. Зберегти у CSV")
        print("8. Візуалізація та збереження графіків")
        print("9. Вихід")
        print("="*60)

def main():
    """
    Головна функція програми
    """
    print("\n" + "="*60)
    print("ПРОГРАМА ОПТИМІЗАЦІЇ ПРОЦЕСІВ ТА ПАРАМЕТРІВ")
    print("За допомогою SciPy.optimize")
    print("="*60)
    
    # Створення оптимізатора
    optimizer = ProductionOptimizer()
    
    # Основний цикл програми
    while True:
        optimizer.display_menu()

        try:
            choice = input("\nВиберіть опцію (1-8): ").strip()
            
            if choice == '1':
                # Завантаження даних
                filename = input("Введіть ім'я файлу: ").strip()
                
                if filename:
                    # Перевіряємо, чи існує файл
                    if os.path.exists(filename):
                        # Файл існує - завантажуємо
                        success = optimizer.load_data(filename)
                        if success:
                            # Показуємо весь вміст завантажених даних
                            print(f"\n{'='*60}")
                            print("ДАНІ УСПІШНО ЗАВАНТАЖЕНІ")
                            print(f"{'='*60}")
                            print(f"Файл: {filename}")
                            print(f"Кількість записів: {len(optimizer.data)}")
                            print(f"Колонки: {list(optimizer.data.columns)}")
                            
                            print(f"\n{'='*60}")
                            print("ПОВНИЙ ВМІСТ ФАЙЛУ:")
                            print(f"{'='*60}")
                            
                            # Показуємо весь DataFrame
                            with pd.option_context('display.max_rows', None, 
                                                   'display.max_columns', None,
                                                   'display.width', 1000):
                                print(optimizer.data)
                            
                            # Показуємо статистику в будь-якому випадку
                            print(f"\n{'='*60}")
                            print("СТАТИСТИКА ДАНИХ:")
                            print(f"{'='*60}")
                            print(optimizer.data.describe())
                    else:
                        # Файл не існує - створюємо тестові дані і зберігаємо у файл
                        print(f"\nФайл '{filename}' не знайдено. Створення тестових даних і збереження у файл...")
                        
                        # Створюємо тестові дані
                        optimizer._generate_test_data()
                        
                        # Додаємо розширення .txt, якщо його немає
                        if not filename.endswith('.txt'):
                            filename = filename + '.txt'
                        
                        try:
                            # Зберігаємо дані у TXT файл 
                            optimizer.data.to_csv(filename, sep='\t', index=False)
                            print(f"Створено файл '{filename}' з {len(optimizer.data)} записами")
                            
                            # Завантажуємо дані з нового файлу
                            success = optimizer.load_data(filename)
                            if success:
                                print(f"\n{'='*60}")
                                print("ДАНІ З НОВОГО ФАЙЛУ:")
                                print(f"{'='*60}")
                                
                                # Показуємо весь вміст нового файлу
                                with pd.option_context('display.max_rows', None, 
                                                       'display.max_columns', None,
                                                       'display.width', 1000):
                                    print(optimizer.data)
                                
                                print(f"\n{'='*60}")
                                print("СТАТИСТИКА НОВИХ ДАНИХ:")
                                print(f"{'='*60}")
                                print(optimizer.data.describe())
                        except Exception as e:
                            print(f"Помилка при створенні файлу: {e}")
                            print("Дані створені тільки в пам'яті.")
            
            elif choice == '2':
                # Оптимізація витрат - запуск всіх методів
                print("\nЗапуск оптимізації витрат за всіма доступними методами...")
                print("="*60)
                
                methods = ['SLSQP', 'COBYLA', 'trust-constr', 'Nelder-Mead', 'BFGS']
                all_results = []
                
                # Користувацькі параметри
                use_custom = input("Використати користувацькі параметри? (так/ні): ").strip().lower()
                user_params = None
                
                if use_custom in ['так', 'т']:
                    try:
                        cost_coeff = list(map(float, input("Коефіцієнти витрат (3 числа через пробіл): ").split()))
                        quality_params = list(map(float, input("Параметри якості (2 числа через пробіл): ").split()))
                        user_params = {
                            'cost_coefficients': cost_coeff,
                            'quality_params': quality_params
                        }
                        print("Користувацькі параметри застосовані")
                    except:
                        print("Помилка введення, використовуються параметри за замовчуванням")
                else:
                    print("Використовуються параметри за замовчуванням")
                
                for method in methods:
                    print(f"\n{'='*40}")
                    print(f"Метод: {method}")
                    print(f"{'='*40}")
                    
                    try:
                        result = optimizer.optimize_production(
                            objective='cost', 
                            method=method, 
                            user_params=user_params
                        )
                        
                        if result.get('success'):
                            all_results.append({
                                'method': method,
                                'optimal_values': result.get('optimal_values'),
                                'production_cost': result.get('production_cost')
                            })
                            print(f"Метод {method} успішний")
                        else:
                            print(f"Метод {method} не вдалося виконати")
                    except Exception as e:
                        print(f"Метод {method} викликав помилку: {str(e)[:100]}...")
                
                # Порівняння результатів
                if all_results:
                    print("\n" + "="*60)
                    print("ПОРІВНЯННЯ РЕЗУЛЬТАТІВ ВСІХ МЕТОДІВ")
                    print("="*60)
                    
                    # Знаходимо найкращий результат (мінімальні витрати)
                    best_result = min(all_results, key=lambda x: x['production_cost'])
                    
                    print("\nНайкращий результат:")
                    print(f"Метод: {best_result['method']}")
                    print(f"Витрати: {best_result['production_cost']:.2f}")
                    print(f"Оптимальні параметри: {best_result['optimal_values']}")
                    
                    # Оновлюємо результат для подальшого аналізу
                    print(f"\nРезультати методу {best_result['method']} будуть використані для подальшого аналізу")
                    
                else:
                    print("\nЖоден з методів не дав успішних результатів")
                    print("Спробуйте інші методи або перевірте параметри")
                
            elif choice == '3':
                # Оптимізація прибутку - запуск всіх методів
                print("\nЗапуск оптимізації прибутку за всіма доступними методами...")
                print("="*60)
                
                methods = ['SLSQP', 'COBYLA', 'trust-constr', 'Nelder-Mead', 'BFGS']
                all_results = []
                
                for method in methods:
                    print(f"\n{'='*40}")
                    print(f"Метод: {method}")
                    print(f"{'='*40}")
                    
                    result = optimizer.optimize_production(
                        objective='profit',
                        method=method
                    )
                    
                    if result.get('success'):
                        all_results.append({
                            'method': method,
                            'optimal_values': result.get('optimal_values'),
                            'profit': result.get('profit')
                        })
                
                # Порівняння результатів
                if all_results:
                    print("\n" + "="*60)
                    print("ПОРІВНЯННЯ РЕЗУЛЬТАТІВ ВСІХ МЕТОДІВ")
                    print("="*60)
                    
                    # Знаходимо найкращий результат (максимальний прибуток)
                    best_result = max(all_results, key=lambda x: x['profit'])
                    
                    print("\nНайкращий результат:")
                    print(f"Метод: {best_result['method']}")
                    print(f"Прибуток: {best_result['profit']:.2f}")
                    print(f"Оптимальні параметри: {best_result['optimal_values']}")
            
            elif choice == '4':
                # Лінійна оптимізація
                results = optimizer.linear_optimization()
            
            elif choice == '5':
                # Аналіз результатів
                optimizer.analyze_results()
            
            elif choice == '6':
                # Збереження результатів
                filename = input("Введіть ім'я файлу (без розширення) [опційно]: ").strip()
                if filename:
                    optimizer.save_results(filename)
                else:
                    optimizer.save_results()
            
            elif choice == '7':
                # Збереження у CSV
                filename = input("Введіть ім'я файлу (без розширення) [опційно]: ").strip()
                if filename:
                    optimizer.save_to_csv(filename)
                else:
                    optimizer.save_to_csv()
            
            elif choice == '8':
                # Візуалізація та збереження графіків
                optimizer._visualize_results()
            
            elif choice == '9':
                # Вихід
                print("\nДякую за використання програми!")
                print("Результати оптимізації збережено у файлах з префіксом 'optimization_'")
                break
            
            else:
                print("Невірний вибір. Спробуйте ще раз.")
        
        except KeyboardInterrupt:
            print("\n\nПрограму перервано користувачем")
            break
        except Exception as e:
            print(f"\nСталася помилка: {e}")
            print("Спробуйте ще раз")

# Постановка задача
"""
ЗАВДАННЯ ОПТИМІЗАЦІЇ ВИРОБНИЧИХ ПАРАМЕТРІВ

Постановка задачі:
Знайти такі значення технічних параметрів виробництва (швидкість виробництва, температура процесу, тиск), які мінімізують загальні витрати або максимізують прибуток при дотриманні технологічних обмежень та бюджетних лімітів.

Математична модель:
Мінімізувати: f(x) = C_m * x1 + C_e x2 + C_px3 + Q(x2, x3) 
Де:
x1 - швидкість виробництва [10, 100] од/год
x2 - температура процесу [50, 300] °C
x3 - тиск [1, 10] бар
C_m, C_e, C_р коефіцієнти витрат
Q(x2, x3) функція штрафу за якість

Обмеження:
1. Технологічні: 10 ≤ x1 ≤ 100, 50 ≤ x2 ≤ 300, 1 ≤ x3 ≤ 10
2. Бюджетні: Загальні витрати ≤ 50000
3. Якості: Коефіцієнти якості ≥ заданих значень
"""

# Алгоритм розв'язання (словесний опис):
"""
АЛГОРИТМ РОЗВ'ЯЗАННЯ ЗАДАЧІ ОПТИМІЗАЦІЇ:

1. Ініціалізація системи:
- Завантаження конфігурації
- Створення об'єкта оптимізатора

2. Завантаження та підготовка даних
- Зчитування даних з файлу або генерація тестових даних
- Перевірка коректності даних
- Обробка відсутніх значень

3. Визначення цільової функції:
- Для мінімізації витрат: функція витрат виробництва
- Для максимізації прибутку: функція прибутку (дохід витрати)

4. Визначення обмежень:
- Технологічні обмеження (межі параметрів)
- Бюджетні обмеження
- Обмеження якості

5. Виконання оптимізації:
- Вибір методу оптимізації (SLSQP, COBYLA, trust-constr)
- Задання початкового наближення
- Виклик функції minimize 3 SciPy.optimize
- Обробка результатів оптимізації

6. Аналіз результатів.
- Перевірка успішності оптимізації
- Вивід оптимальних значень параметрів
- Розрахунок економічних показників

7. Візуалізація:
- Побудова графіків оптимальних параметрів
- Візуалізація поверхні витрат
- Порівняльні діаграми

8. Збереження результатів:
- Експорт результатів у JSON формат
- Генерація текстового звіту
- Збереження графіків у файли

9. Взаємодія з користувачем:
- Меню для вибору операцій
- Введення користувацьких параметрів.
- Інформування про хід виконання
- Обробка помилок та виключних ситуацій
"""

if __name__ == "__main__":
    main()
