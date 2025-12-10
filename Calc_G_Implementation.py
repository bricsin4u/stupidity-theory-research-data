"""
Calc_G_Implementation.py
Реализация расчета уровня глупости G и оптимизированного G*
Проект "Теория глупости" - Игорь Сергеевич Петренко
Версия модели/кода: v0.2
Дата: 2025-09-10
Формула: G = α₁·(B/I) + α₂·(D·S/E) + α₃·((1 − C/200)/R); рекомендации: α₁+α₂+α₃ ≈ 1; домены: I,E,C∈(0,200], B,D,S∈[0,1], R∈(0,1].
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

@dataclass
class PersonProfile:
    """Профиль индивида для расчета G"""
    # Основные интеллектуальные показатели
    IQ: float  # 0-200, стандартный z-балл
    EQ: float  # 0-200, эмоциональный интеллект
    CQ: float  # 0-200, культурный интеллект
    
    # Факторы риска
    B: float   # 0-1, когнитивные искажения (BIAS)
    D: float   # 0-1, цифровая перегрузка (DOI)
    S: float   # 0-1, социальное давление
    R: float   # 0-1, ресурс самоконтроля
    
    # Метаданные
    person_id: str = ""
    age: Optional[int] = None
    education: Optional[str] = None
    culture: Optional[str] = None
    
    # Версионирование скоринга и качество данных (опционально)
    vBIAS_scoring: Optional[str] = None  # версия скоринга BIAS (например, "vBIAS_scoring=2025-09-10")
    vDOI_scoring: Optional[str] = None   # версия скоринга DOI (например, "vDOI_scoring=2025-09-10")
    qc_score: Optional[float] = None     # качество данных [0,1] для downweight в анализе/обучении

class StupidityCalculator:
    """Основной класс для расчета уровня глупости"""
    
    def __init__(self, alpha1: float = 0.4, alpha2: float = 0.35, alpha3: float = 0.25):
        """
        Инициализация с коэффициентами модели
        
        Args:
            alpha1: Вес когнитивных искажений
            alpha2: Вес цифрового шума и социального давления
            alpha3: Вес культурного интеллекта и самоконтроля
        """
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        
        # Проверка корректности коэффициентов
        if not np.isclose(alpha1 + alpha2 + alpha3, 1.0, atol=0.01):
            print(f"Предупреждение: Сумма альфа-коэффициентов = {alpha1 + alpha2 + alpha3:.3f}")
    
    def calculate_g(self, profile: PersonProfile) -> float:
        """
        Расчет уровня глупости G
        
        G = α₁×(B/I) + α₂×(D×S/E) + α₃×((1-C/200)/R)
        
        Args:
            profile: Профиль индивида
            
        Returns:
            Уровень глупости G (чем выше, тем хуже)
        """
        # Проверка валидности данных
        self._validate_profile(profile)
        
        # Расчет компонентов
        term1 = self.alpha1 * (profile.B / profile.IQ)
        term2 = self.alpha2 * ((profile.D * profile.S) / profile.EQ)
        term3 = self.alpha3 * ((1 - profile.CQ/200) / profile.R)
        
        G = term1 + term2 + term3
        
        return G
    
    def calculate_g_components(self, profile: PersonProfile) -> Dict[str, float]:
        """
        Расчет компонентов G для детального анализа
        
        Returns:
            Словарь с компонентами и их вкладом в G
        """
        self._validate_profile(profile)
        
        term1 = self.alpha1 * (profile.B / profile.IQ)
        term2 = self.alpha2 * ((profile.D * profile.S) / profile.EQ)
        term3 = self.alpha3 * ((1 - profile.CQ/200) / profile.R)
        total = term1 + term2 + term3
        
        # Безопасные относительные вклады (если total == 0, доли определяем как 0)
        if total > 0:
            rel = {
                'bias_contribution': term1 / total,
                'digital_contribution': term2 / total,
                'cultural_contribution': term3 / total
            }
        else:
            rel = {
                'bias_contribution': 0.0,
                'digital_contribution': 0.0,
                'cultural_contribution': 0.0
            }
        
        return {
            'cognitive_bias_component': term1,
            'digital_social_component': term2,
            'cultural_control_component': term3,
            'total_G': total,
            'relative_contributions': rel
        }

    def _validate_profile(self, profile: PersonProfile) -> None:
        """Валидация профиля индивида"""
        # Проверка диапазонов
        if not (0 < profile.IQ <= 200):
            raise ValueError(f"IQ должен быть в диапазоне (0, 200], получен: {profile.IQ}")
        if not (0 < profile.EQ <= 200):
            raise ValueError(f"EQ должен быть в диапазоне (0, 200], получен: {profile.EQ}")
        if not (0 < profile.CQ <= 200):
            raise ValueError(f"CQ должен быть в диапазоне (0, 200], получен: {profile.CQ}")
        
        # B, D, S должны быть в [0,1]
        for param, value in [('B', profile.B), ('D', profile.D), ('S', profile.S)]:
            if not (0 <= value <= 1):
                raise ValueError(f"{param} должен быть в диапазоне [0, 1], получен: {value}")
        
        # R должен быть строго положительным для избежания деления на ноль
        if not (0 < profile.R <= 1):
            raise ValueError(f"R должен быть в диапазоне (0, 1], получен: {profile.R}")
        
        # При наличии qc_score проверяем домен
        if profile.qc_score is not None and not (0 <= profile.qc_score <= 1):
            raise ValueError(f"qc_score должен быть в диапазоне [0, 1], получен: {profile.qc_score}")

class InterventionOptimizer:
    """Класс для оптимизации интервенций и расчета G*"""
    
    def __init__(self, calculator: StupidityCalculator):
        self.calculator = calculator
    
    def calculate_g_star(self, profile: PersonProfile, interventions: Dict[str, float]) -> float:
        """
        Расчет оптимизированного уровня глупости G* после интервенций
        
        Args:
            profile: Исходный профиль
            interventions: Словарь изменений {'delta_IQ': 5, 'delta_EQ': 10, ...}
            
        Returns:
            Новый уровень глупости G*
        """
        # Создание нового профиля с учетом интервенций
        new_profile = PersonProfile(
            IQ=max(1, min(200, profile.IQ + interventions.get('delta_IQ', 0))),
            EQ=max(1, min(200, profile.EQ + interventions.get('delta_EQ', 0))),
            CQ=max(1, min(200, profile.CQ + interventions.get('delta_CQ', 0))),
            B=max(0, min(1, profile.B - interventions.get('delta_B', 0))),  # Уменьшение искажений
            D=max(0, min(1, profile.D - interventions.get('delta_D', 0))),  # Уменьшение шума
            S=max(0, min(1, profile.S - interventions.get('delta_S', 0))),  # Уменьшение давления
            R=max(0, min(1, profile.R + interventions.get('delta_R', 0))),  # Увеличение самоконтроля
            person_id=profile.person_id + "_optimized"
        )
        
        return self.calculator.calculate_g(new_profile)
    
    def optimize_interventions(self, profile: PersonProfile, budget: float, 
                             intervention_costs: Dict[str, float],
                             max_changes: Dict[str, float]) -> Dict[str, float]:
        """
        Поиск оптимального набора интервенций в рамках бюджета
        
        Args:
            profile: Исходный профиль
            budget: Доступный бюджет
            intervention_costs: Стоимость единицы изменения каждого параметра
            max_changes: Максимально возможные изменения каждого параметра
            
        Returns:
            Оптимальный набор интервенций
        """
        def objective(x):
            """Целевая функция - минимизируем G*"""
            interventions = {
                'delta_IQ': x[0],
                'delta_EQ': x[1], 
                'delta_CQ': x[2],
                'delta_B': x[3],
                'delta_D': x[4],
                'delta_S': x[5],
                'delta_R': x[6]
            }
            return self.calculate_g_star(profile, interventions)
        
        def budget_constraint(x):
            """Ограничение по бюджету"""
            cost = (x[0] * intervention_costs.get('IQ', 100) +
                   x[1] * intervention_costs.get('EQ', 80) +
                   x[2] * intervention_costs.get('CQ', 90) +
                   x[3] * intervention_costs.get('B', 120) +
                   x[4] * intervention_costs.get('D', 50) +
                   x[5] * intervention_costs.get('S', 60) +
                   x[6] * intervention_costs.get('R', 70))
            return budget - cost
        
        # Ограничения
        constraints = [{'type': 'ineq', 'fun': budget_constraint}]
        
        # Границы изменений
        bounds = [
            (0, max_changes.get('IQ', 20)),
            (0, max_changes.get('EQ', 25)),
            (0, max_changes.get('CQ', 30)),
            (0, max_changes.get('B', 0.3)),
            (0, max_changes.get('D', 0.4)),
            (0, max_changes.get('S', 0.2)),
            (0, max_changes.get('R', 0.3))
        ]
        
        # Начальная точка
        x0 = [5, 5, 5, 0.1, 0.1, 0.05, 0.1]
        
        # Оптимизация
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            return {
                'delta_IQ': result.x[0],
                'delta_EQ': result.x[1],
                'delta_CQ': result.x[2],
                'delta_B': result.x[3],
                'delta_D': result.x[4],
                'delta_S': result.x[5],
                'delta_R': result.x[6],
                'optimized_G': result.fun,
                'total_cost': budget - budget_constraint(result.x)
            }
        else:
            raise RuntimeError(f"Оптимизация не сошлась: {result.message}")

class StupidityAnalyzer:
    """Класс для анализа и визуализации результатов"""
    
    def __init__(self, calculator: StupidityCalculator):
        self.calculator = calculator
    
    def analyze_population(self, profiles: List[PersonProfile]) -> pd.DataFrame:
        """
        Анализ популяции индивидов
        
        Args:
            profiles: Список профилей
            
        Returns:
            DataFrame с результатами анализа
        """
        results = []
        
        for profile in profiles:
            G = self.calculator.calculate_g(profile)
            components = self.calculator.calculate_g_components(profile)
            
            results.append({
                'person_id': profile.person_id,
                'IQ': profile.IQ,
                'EQ': profile.EQ,
                'CQ': profile.CQ,
                'B': profile.B,
                'D': profile.D,
                'S': profile.S,
                'R': profile.R,
                'G': G,
                'bias_component': components['cognitive_bias_component'],
                'digital_component': components['digital_social_component'],
                'cultural_component': components['cultural_control_component'],
                'age': profile.age,
                'education': profile.education,
                'culture': profile.culture
            })
        
        return pd.DataFrame(results)
    
    def plot_g_distribution(self, df: pd.DataFrame, title: str = "Распределение уровня глупости G"):
        """Визуализация распределения G"""
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(df['G'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Уровень глупости G')
        plt.ylabel('Частота')
        plt.title('Гистограмма G')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.boxplot(df['G'])
        plt.ylabel('Уровень глупости G')
        plt.title('Ящик с усами G')
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def plot_component_analysis(self, df: pd.DataFrame):
        """Анализ вклада компонентов в G"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Корреляционная матрица
        corr_vars = ['IQ', 'EQ', 'CQ', 'B', 'D', 'S', 'R', 'G']
        corr_matrix = df[corr_vars].corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   ax=axes[0,0], fmt='.2f')
        axes[0,0].set_title('Корреляционная матрица')
        
        # Вклад компонентов
        components = ['bias_component', 'digital_component', 'cultural_component']
        component_means = df[components].mean()
        
        axes[0,1].pie(component_means, labels=['Когнитивные\nискажения', 'Цифровой шум\n+ соц. давление', 
                                              'Культурный IQ\n+ самоконтроль'], 
                     autopct='%1.1f%%', startangle=90)
        axes[0,1].set_title('Средний вклад компонентов в G')
        
        # Scatter plots
        axes[1,0].scatter(df['IQ'], df['G'], alpha=0.6, color='red')
        axes[1,0].set_xlabel('IQ')
        axes[1,0].set_ylabel('G')
        axes[1,0].set_title('G vs IQ')
        axes[1,0].grid(True, alpha=0.3)
        
        axes[1,1].scatter(df['B'], df['G'], alpha=0.6, color='orange')
        axes[1,1].set_xlabel('Когнитивные искажения (B)')
        axes[1,1].set_ylabel('G')
        axes[1,1].set_title('G vs Когнитивные искажения')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Пример использования
def example_usage():
    """Пример использования классов"""
    
    # Создание калькулятора
    calc = StupidityCalculator(alpha1=0.4, alpha2=0.35, alpha3=0.25)
    
    # Пример профиля
    profile = PersonProfile(
        IQ=110, EQ=95, CQ=130,
        B=0.35, D=0.70, S=0.55, R=0.40,
        person_id="example_001",
        age=25, education="высшее", culture="русская"
    )
    
    # Расчет G
    G = calc.calculate_g(profile)
    print(f"Уровень глупости G: {G:.4f}")
    
    # Детальный анализ
    components = calc.calculate_g_components(profile)
    print("\nКомпоненты G:")
    for key, value in components.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v:.4f}")
        else:
            print(f"{key}: {value:.4f}")
    
    # Оптимизация интервенций
    optimizer = InterventionOptimizer(calc)
    
    intervention_costs = {
        'IQ': 200, 'EQ': 150, 'CQ': 180,
        'B': 250, 'D': 100, 'S': 120, 'R': 130
    }
    
    max_changes = {
        'IQ': 15, 'EQ': 20, 'CQ': 25,
        'B': 0.25, 'D': 0.35, 'S': 0.20, 'R': 0.30
    }
    
    try:
        optimal_interventions = optimizer.optimize_interventions(
            profile, budget=2000, 
            intervention_costs=intervention_costs,
            max_changes=max_changes
        )
        
        print(f"\nОптимальные интервенции (бюджет: 2000):")
        for key, value in optimal_interventions.items():
            print(f"{key}: {value:.3f}")
            
        # Расчет G*
        G_star = optimal_interventions['optimized_G']
        improvement = ((G - G_star) / G) * 100
        print(f"\nУлучшение: {improvement:.1f}% (G: {G:.4f} → G*: {G_star:.4f})")
        
    except Exception as e:
        print(f"Ошибка оптимизации: {e}")

# === Утилиты скоринга EQ/BIAS и нормирования ===

def logistic(x: float) -> float:
    """Логистическая функция σ(x) для приведения к [0,1]"""
    return float(1.0 / (1.0 + np.exp(-x)))


def z_to_0_200(z: float) -> float:
    """Преобразование z-балла в шкалу 0-200 (IQ-подобная: 100+15·z), с отсечкой в [1,200]"""
    return float(np.clip(100.0 + 15.0 * z, 1.0, 200.0))


def compute_z(score: float, ref_mean: float, ref_sd: float) -> float:
    """Стандартизация: z = (score − μ_ref) / σ_ref"""
    if ref_sd <= 0:
        raise ValueError("ref_sd must be > 0")
    return (score - ref_mean) / ref_sd


def reverse_likert(items: List[float], reverse_idx: List[int], scale_min: float, scale_max: float) -> List[float]:
    """Реверс по ключу для Лайкерт-скал: v' = min+max − v для заданных индексов"""
    arr = np.array(items, dtype=float)
    for idx in reverse_idx:
        if idx < 0 or idx >= len(arr):
            raise IndexError(f"reverse index out of range: {idx}")
        arr[idx] = scale_min + scale_max - arr[idx]
    return arr.tolist()


def score_teique(items: List[float], reverse_idx: List[int], scale_min: float, scale_max: float,
                 ref_mean: float, ref_sd: float) -> Dict[str, float]:
    """
    Скоринг TEIQue (Trait-EI):
    - Реверс по ключу
    - Среднее по фасетам/пунктам как Trait EI (упрощённая агрегация)
    - z-нормировка по референсу
    - Преобразование в 0-200
    """
    vals = np.array(reverse_likert(items, reverse_idx, scale_min, scale_max), dtype=float)
    trait_ei = float(np.mean(vals))
    z = compute_z(trait_ei, ref_mean, ref_sd)
    eq200 = z_to_0_200(z)
    return {"trait_ei_mean": trait_ei, "z_eq": z, "EQ_200": eq200}


def score_wleis(subscales: Dict[str, List[float]], reverse_idx: Dict[str, List[int]],
                scale_min: float, scale_max: float, ref_mean: float, ref_sd: float) -> Dict[str, float]:
    """
    Скоринг WLEIS: 4 субшкалы (SEA, OEA, UOE, ROE) → средние по субшкалам → общий индекс как среднее субшкал
    Далее: z-нормировка и преобразование в 0-200
    """
    sub_means: Dict[str, float] = {}
    all_means: List[float] = []
    for name, items in subscales.items():
        r_idx = reverse_idx.get(name, [])
        vals = np.array(reverse_likert(items, r_idx, scale_min, scale_max), dtype=float)
        m = float(np.mean(vals))
        sub_means[name] = m
        all_means.append(m)
    total_mean = float(np.mean(all_means))
    z = compute_z(total_mean, ref_mean, ref_sd)
    eq200 = z_to_0_200(z)
    return {**sub_means, "total_mean": total_mean, "z_eq": z, "EQ_200": eq200}


def score_esci(self_scores: Optional[List[float]] = None, others_scores: Optional[List[List[float]]] = None,
               ref_mean: float = 0.0, ref_sd: float = 1.0) -> Dict[str, float]:
    """
    Скоринг ESCI (360):
    - Усреднение self и усреднение по рейтёрам others (если присутствуют)
    - Комбинированный индекс как среднее из доступных компонентов
    - z-нормировка и преобразование в 0-200
    Примечание: надёжность по рейтёрам (ICC) и веса self/others задаются вне этой функции на уровне анализа.
    """
    comps: List[float] = []
    res: Dict[str, float] = {}
    if self_scores is not None and len(self_scores) > 0:
        self_mean = float(np.mean(self_scores))
        res["self_mean"] = self_mean
        comps.append(self_mean)
    if others_scores is not None and len(others_scores) > 0:
        others_means = [float(np.mean(r)) for r in others_scores if len(r) > 0]
        if len(others_means) > 0:
            others_mean = float(np.mean(others_means))
            res["others_mean"] = others_mean
            comps.append(others_mean)
    if len(comps) == 0:
        raise ValueError("At least one of self_scores or others_scores must be provided")
    combined = float(np.mean(comps))
    z = compute_z(combined, ref_mean, ref_sd)
    eq200 = z_to_0_200(z)
    res.update({"combined_mean": combined, "z_eq": z, "EQ_200": eq200})
    return res


def score_bias(subscales: Dict[str, List[float]], reverse_idx: Dict[str, List[int]],
               scale_min: float, scale_max: float, ref_mean: float, ref_sd: float, gamma_B: float = 2.0) -> Dict[str, float]:
    """
    Скоринг BIAS-Scale:
    - Реверс по ключам внутри каждой субшкалы
    - Среднее по каждой субшкале, затем общий B_raw как среднее субшкал
    - Нормирование: z_B по референсу и логистическое преобразование B = σ(γ_B·z_B)
    """
    sub_means: List[float] = []
    for name, items in subscales.items():
        r_idx = reverse_idx.get(name, [])
        vals = np.array(reverse_likert(items, r_idx, scale_min, scale_max), dtype=float)
        sub_means.append(float(np.mean(vals)))
    b_raw = float(np.mean(sub_means))
    z_b = compute_z(b_raw, ref_mean, ref_sd)
    b = logistic(gamma_B * z_b)
    return {"B_raw": b_raw, "z_B": z_b, "B": b}




def attention_check_failures(values: List[Optional[float]], checks: List[Tuple[int, float]], tolerance: float = 0.0) -> Dict[str, float]:
    """Подсчет нарушений attention checks.
    checks: список (index, expected_value). tolerance — допустимое отклонение.
    """
    fail = 0
    for idx, exp in checks:
        if idx < 0 or idx >= len(values):
            continue
        v = values[idx]
        if v is None or (abs(float(v) - float(exp)) > tolerance):
            fail += 1
    return {"attn_fails": float(fail)}


def qc_summary(values: List[Optional[float]], duration_sec: float, normative_sec: float,
               checks: List[Tuple[int, float]],
               max_allowed_run: int = 8, min_completion: float = 0.9,
               cutoff_ratio: float = 0.5,
               weights: Dict[str, float] = None) -> Dict[str, float]:
    """Сводный QC: агрегирует индикаторы в qc_score∈[0,1] (1 — наилучшее качество).
    По умолчанию равные веса индикаторов.
    """
    ls = detect_longstring(values, max_allowed_run)
    sp = detect_speeder(duration_sec, normative_sec, cutoff_ratio)
    cm = is_complete(values, min_completion)
    ac = attention_check_failures(values, checks)

    w = weights or {"longstring": 1.0, "speeder": 1.0, "low_completion": 1.0, "attn": 1.0}
    total_w = sum(w.values())
    # Индикаторы как 0/1: отсутствие проблемы — 1, наличие — 0 для суббалла
    sub = {
        "longstring": 1.0 - ls["flag_longstring"],
        "speeder": 1.0 - sp["flag_speeder"],
        "low_completion": 1.0 - cm["flag_low_completion"],
        "attn": 1.0 - float(ac["attn_fails"] > 0)
    }
    qc = (w["longstring"] * sub["longstring"] +
           w["speeder"] * sub["speeder"] +
           w["low_completion"] * sub["low_completion"] +
           w["attn"] * sub["attn"]) / max(1e-9, total_w)

    return {**ls, **sp, **cm, **ac, "qc_score": float(np.clip(qc, 0.0, 1.0))}


if __name__ == "__main__":
    example_usage()


def cronbach_alpha(matrix: List[List[float]]) -> float:
    """Оценка надёжности по Кронбаху α для матрицы ответов формы (n_респондентов × n_пунктов).
    Возвращает NaN, если данных недостаточно или дисперсия суммы нулевая.
    Формула: α = k/(k-1) * (1 - ΣVar(item)/Var(Σitems))."""
    X = np.array(matrix, dtype=float)
    if X.ndim != 2 or X.shape[1] < 2 or X.shape[0] < 2:
        return float('nan')
    # Дисперсии пунктов (несмещённые)
    item_vars = X.var(axis=0, ddof=1)
    # Дисперсия суммы пунктов (несмещённая)
    total_var = X.sum(axis=1).var(ddof=1)
    k = X.shape[1]
    if total_var <= 0 or k <= 1:
        return float('nan')
    alpha = (k / (k - 1.0)) * (1.0 - float(np.sum(item_vars)) / float(total_var))
    # Ограничим в [0,1] для стабильности отчётов
    return float(np.clip(alpha, 0.0, 1.0))


def mcdonald_omega_total(matrix: List[List[float]]) -> float:
    """Приближённая оценка McDonald's ω_total на основе однофакторной модели через первую главную компоненту.
    Алгоритм:
    1) Берём ковариационную матрицу пунктов S;
    2) Находим крупнейшее собственное значение λ1 и собственный вектор v;
    3) Факторные нагрузки ≈ v * sqrt(λ1);
    4) Уникальные дисперсии ψ_i = Var_i - loading_i^2;
    5) ω_total = (sum(loadings))^2 / ((sum(loadings))^2 + sum(ψ_i)).
    Возвращает NaN, если данных недостаточно.
    """
    X = np.array(matrix, dtype=float)
    if X.ndim != 2 or X.shape[1] < 2 or X.shape[0] < 2:
        return float('nan')
    # Ковариационная матрица пунктов
    S = np.cov(X, rowvar=False, ddof=1)
    # Численная стабильность
    if not np.all(np.isfinite(S)):
        return float('nan')
    # Собственные значения/векторы
    vals, vecs = np.linalg.eigh(S)
    idx = int(np.argmax(vals))
    lambda1 = float(vals[idx])
    if lambda1 <= 0:
        return float('nan')
    v = vecs[:, idx]
    loadings = v * np.sqrt(lambda1)
    # Уникальные дисперсии (неотрицательные)
    uniq = np.clip(np.diag(S) - loadings**2, 0.0, None)
    num = float(np.sum(loadings)) ** 2
    den = num + float(np.sum(uniq))
    if den <= 0:
        return float('nan')
    omega = num / den
    return float(np.clip(omega, 0.0, 1.0))


def efa_pca(matrix: List[List[float]], n_factors: int = 1) -> Dict[str, np.ndarray]:
    """Простая EFA на основе PCA над корреляционной матрицей.
    Возвращает словарь с ключами: loadings (n_items×n_factors), eigenvalues, explained_var_ratio, communalities.
    Робастные шаги: median/MAD стандартизация, подавление NaN/Inf, симметризация и безопасная нормировка.
    """
    X = np.array(matrix, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 2 or n_factors < 1:
        raise ValueError("Incorrect input shape for EFA/PCA")
    p = X.shape[1]
    # Робастная стандартизация по столбцам: median/MAD с фолбеком на std и 1.0
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    scale = 1.4826 * mad
    st = X.std(axis=0, ddof=1)
    scale = np.where(np.isfinite(scale) & (scale >= 1e-8), scale, st)
    scale = np.where(np.isfinite(scale) & (scale >= 1e-8), scale, 1.0)
    Xr = (X - med) / (scale + 1e-12)
    # Нормализованная ковариация -> корреляционная матрица, с подавлением NaN/Inf
    S = np.cov(Xr, rowvar=False, ddof=1)
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
    S = 0.5 * (S + S.T)
    d = np.clip(np.diag(S), 1e-12, None)
    D = np.sqrt(d)
    R = S / (D[None, :] * D[:, None] + 1e-12)
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(R, 1.0)
    # Собственные значения/векторы
    vals, vecs = np.linalg.eigh(R)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    k = int(min(n_factors, p))
    # PCA loadings = eigenvectors * sqrt(eigenvalues)
    L = vecs[:, :k] * np.sqrt(np.clip(vals[:k], 0.0, None))
    total_var = float(R.shape[0])  # сумма собственных значений корр. матрицы равна числу переменных
    evr = np.clip(vals[:k] / max(1e-12, total_var), 0.0, 1.0)
    comm = np.sum(L**2, axis=1)
    return {
        "loadings": L.astype(float),
        "eigenvalues": vals[:k].astype(float),
        "explained_var_ratio": evr.astype(float),
        "communalities": comm.astype(float),
    }


def tucker_congruence(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Коэффициент конгруэнтности Такера по столбцам двух матриц нагрузок (n_items×n_factors).
    Предполагается совпадение порядка факторов. Возвращает массив длины n_factors."""
    if a.shape != b.shape:
        raise ValueError("Shapes of loading matrices must match")
    n_factors = a.shape[1]
    phis = []
    for j in range(n_factors):
        av = a[:, j]
        bv = b[:, j]
        num = float(np.dot(av, bv))
        den = float(np.linalg.norm(av) * np.linalg.norm(bv)) + 1e-12
        # Инвариантность к произвольному знаку факторов
        phis.append(abs(num / den))
    return np.array(phis, dtype=float)


def check_invariance_by_congruence(groups: Dict[str, List[List[float]]], n_factors: int = 1, threshold: float = 0.90) -> Dict[str, float]:
    """Грубая проверка измерительной инвариантности: PCA-нагрузки по группам с варимакс-вращением и
    парные конгруэнтности Такера с оптимальным сопоставлением факторов (перестановкой столбцов).
    Возвращает минимальную/среднюю конгруэнтность и флаг pass_all.
    threshold — порог для конгруэнтности на каждом факторе.
    """
    def varimax(Phi, gamma: float = 1.0, q: int = 20, tol: float = 1e-6):
        Phi = np.array(Phi, dtype=float)
        p, k = Phi.shape
        if k == 1:
            return Phi
        R = np.eye(k)
        d = 0
        for _ in range(q):
            d_old = d
            Lambda = Phi @ R
            u, s, vh = np.linalg.svd(Phi.T @ (Lambda**3 - (gamma/p) * Lambda @ np.diag(np.sum(Lambda**2, axis=0))))
            R = u @ vh
            d = np.sum(s)
            if d_old != 0 and d/d_old < 1 + tol:
                break
        return Phi @ R

    keys = list(groups.keys())
    loadings = {}
    for k in keys:
        res = efa_pca(groups[k], n_factors=n_factors)
        L = res["loadings"]
        # Вращение Варимакс и нормировка столбцов
        try:
            L = varimax(L)
        except Exception:
            pass
        # Подавление NaN/Inf и безопасная нормировка столбцов
        L = np.nan_to_num(L, nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(L, axis=0)
        norms = np.where(norms < 1e-12, 1.0, norms)
        L = L / norms
        loadings[k] = L

    def congruence_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        if A.shape != B.shape:
            raise ValueError("Shapes of loading matrices must match")
        k = A.shape[1]
        C = np.zeros((k, k), dtype=float)
        for p in range(k):
            av = A[:, p]
            na = float(np.linalg.norm(av)) + 1e-12
            for q in range(k):
                bv = B[:, q]
                nb = float(np.linalg.norm(bv)) + 1e-12
                C[p, q] = abs(float(np.dot(av, bv)) / (na * nb))
        return C

    # Все пары с оптимальным матчингом факторов
    pair_congr = []
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            L1 = loadings[keys[i]]
            L2 = loadings[keys[j]]
            C = congruence_matrix(L1, L2)
            k = C.shape[0]
            phis = None
            # Пытаемся использовать Венгерский алгоритм, иначе перебор для малых k
            try:
                from scipy.optimize import linear_sum_assignment  # type: ignore
                row_ind, col_ind = linear_sum_assignment(-C)
                phis = C[row_ind, col_ind]
            except Exception:
                from itertools import permutations
                best = None
                for perm in permutations(range(k)):
                    val = float(np.sum([C[r, perm[r]] for r in range(k)]))
                    if best is None or val > best[0]:
                        best = (val, np.array([C[r, perm[r]] for r in range(k)], dtype=float))
                phis = best[1] if best is not None else np.diag(C)
            pair_congr.append(np.array(phis, dtype=float))

    if not pair_congr:
        raise ValueError("Need at least two groups for invariance check")
    pair_congr = np.vstack(pair_congr)
    min_congr = float(np.min(pair_congr))
    mean_congr = float(np.mean(pair_congr))
    pass_all = float((pair_congr >= threshold).all())
    return {"min_congruence": min_congr, "mean_congruence": mean_congr, "pass_all": pass_all}


def dif_linear(matrix: List[List[float]], groups: List[int], alpha: float = 0.05) -> pd.DataFrame:
    """Простая детекция DIF для поликоттомных пунктов через линейную регрессию.
    Для каждого пункта i оценивает модели:
      reduced:     y ~ 1 + total_without_i
      uniform:     y ~ 1 + total_without_i + group
      non-uniform: y ~ 1 + total_without_i + group + total_without_i:group
    Возвращает DataFrame с p-value для uniform/nonuniform и бинарными флагами при уровне alpha.
    """
    X = np.array(matrix, dtype=float)
    g = np.array(groups, dtype=float)
    if X.ndim != 2 or X.shape[0] < 5 or X.shape[1] < 2 or X.shape[0] != g.shape[0]:
        raise ValueError("Invalid inputs for DIF detection")
    n, k = X.shape
    total = X.sum(axis=1)
    rows = []
    from scipy.stats import f as fdist

    def ols_ssr(y, Xd):
        beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        resid = y - Xd @ beta
        return float(np.sum(resid**2)), Xd.shape[1]

    for i in range(k):
        y = X[:, i]
        tot_wo = total - y
        # Сборка дизайнов
        Xr = np.column_stack([np.ones(n), tot_wo])
        Xu = np.column_stack([np.ones(n), tot_wo, g])
        Xn = np.column_stack([np.ones(n), tot_wo, g, tot_wo * g])
        ssr_r, p_r = ols_ssr(y, Xr)
        ssr_u, p_u = ols_ssr(y, Xu)
        ssr_n, p_n = ols_ssr(y, Xn)
        # F-тесты nested
        df_u = n - p_u
        df_n = n - p_n
        F_uniform = ((ssr_r - ssr_u) / (p_u - p_r)) / (ssr_u / max(1e-12, df_u))
        F_nonunif = ((ssr_u - ssr_n) / (p_n - p_u)) / (ssr_n / max(1e-12, df_n))
        p_uniform = float(fdist.sf(F_uniform, p_u - p_r, df_u))
        p_nonunif = float(fdist.sf(F_nonunif, p_n - p_u, df_n))
        rows.append({
            "item": i,
            "p_uniform": p_uniform,
            "p_nonuniform": p_nonunif,
            "flag_uniform": float(p_uniform < alpha),
            "flag_nonuniform": float(p_nonunif < alpha),
        })
    return pd.DataFrame(rows)


def cfa_ls(matrix: List[List[float]], groups: List[int]) -> Dict[str, object]:
    """Упрощённая CFA по наименьшим квадратам на корреляционной матрице.
    Параметры:
    - matrix: ответы (n_resp × n_items)
    - groups: длины n_items, номер фактора для каждого пункта (0..k-1)
    Возвращает: {'loadings': L (p×k), 'SRMR': float, 'pseudo_CFI': float, 'communalities': h2}
    Примечание: метод приблизительный (на базе PCA по кластерам пунктов) — для быстрой диагностики.
    """
    X = np.array(matrix, dtype=float)
    if X.ndim != 2 or X.shape[0] < 3 or X.shape[1] < 2:
        raise ValueError("Incorrect input for CFA")
    p = X.shape[1]
    groups = np.array(groups, dtype=int)
    if groups.shape[0] != p:
        raise ValueError("groups length must equal number of items")
    k = int(groups.max()) + 1
    # Стандартизация и корреляционная матрица
    Xc = (X - X.mean(axis=0)) / (X.std(axis=0, ddof=1) + 1e-12)
    R = np.corrcoef(Xc, rowvar=False)
    # Оценка нагрузок по кластерам через 1-ю СК
    L = np.zeros((p, k), dtype=float)
    for j in range(k):
        idx = np.where(groups == j)[0]
        if idx.size >= 2:
            Rjj = R[np.ix_(idx, idx)]
            vals, vecs = np.linalg.eigh(Rjj)
            e = float(np.max(vals))
            v = vecs[:, int(np.argmax(vals))]
            load = v * np.sqrt(max(0.0, e))
            L[idx, j] = load
        elif idx.size == 1:
            # Однопунктовый фактор: оставляем малую нагрузку, чтобы избежать вырождения
            L[idx[0], j] = 0.3
    # Имплайд корреляции: R_hat = L L^T + diag(psi), psi = 1 - h2
    h2 = np.sum(L**2, axis=1)
    psi = np.clip(1.0 - h2, 1e-6, None)
    R_hat = L @ L.T + np.diag(psi)
    # Метрики подгонки
    off = ~np.eye(p, dtype=bool)
    resid = R[off] - R_hat[off]
    srmr = float(np.sqrt(np.mean(resid**2)))
    rss0 = float(np.sum(R[off]**2)) + 1e-12
    rssm = float(np.sum(resid**2))
    pseudo_cfi = float(np.clip(1.0 - (rssm / rss0), 0.0, 1.0))
    return {"loadings": L.astype(float), "SRMR": srmr, "pseudo_CFI": pseudo_cfi, "communalities": h2.astype(float)}


def irt_rasch_jml(matrix: List[List[int]], max_iter: int = 200, tol: float = 1e-3) -> Dict[str, object]:
    """Простая JML-оценка для Rasch (1PL) с центрированием параметров и стабилизацией шага.
    Возвращает: {'theta': np.array(N,), 'b': np.array(J,), 'converged': bool, 'n_iter': int}
    """
    X = np.array(matrix, dtype=float)
    if X.ndim != 2 or X.shape[0] < 5 or X.shape[1] < 2:
        raise ValueError("Incorrect input for Rasch JML")
    # Проверка на бинарность
    if not np.all((X == 0) | (X == 1)):
        raise ValueError("Rasch JML requires binary (0/1) responses")
    N, J = X.shape
    # Инициализация параметров по логитам сумм (устойчивая стартовая точка)
    row_sum = X.sum(axis=1)
    col_sum = X.sum(axis=0)
    def safe_logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))
    theta = safe_logit((row_sum + 0.5) / (J + 1.0))
    b = -safe_logit((col_sum + 0.5) / (N + 1.0))
    # Центрирование и клиппинг стартов
    theta -= np.mean(theta)
    b -= np.mean(b)
    theta = np.clip(theta, -4.0, 4.0)
    b = np.clip(b, -4.0, 4.0)

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    prev_ll = -np.inf
    converged = False
    lr = 1.0  # демпфирование шага Ньютона
    min_lr = 1e-5
    for it in range(1, max_iter + 1):
        theta0 = theta.copy()
        b0 = b.copy()

        # Векторизованный шаг Фишера + монотонный бэктрекинг по шагу (line-search)
        # Вычисляем направления при текущих параметрах
        P = sigmoid(theta[:, None] - b[None, :])
        grad_theta = np.sum(X - P, axis=1)
        hess_theta = np.sum(P * (1 - P), axis=1) + 1e-6
        delta_theta = np.clip(grad_theta / hess_theta, -3.0, 3.0)

        grad_b = -np.sum(X - P, axis=0)
        hess_b = np.sum(P * (1 - P), axis=0) + 1e-6
        delta_b = np.clip(grad_b / hess_b, -3.0, 3.0)

        # Бэктрекинг: подбираем lr_loc, чтобы ll не ухудшалось
        lr_loc = lr
        ll = prev_ll
        while True:
            theta_c = theta + lr_loc * delta_theta
            theta_c -= np.mean(theta_c)
            theta_c = np.clip(theta_c, -6.0, 6.0)

            b_c = b + lr_loc * delta_b
            b_c -= np.mean(b_c)
            b_c = np.clip(b_c, -6.0, 6.0)

            P_c = sigmoid(theta_c[:, None] - b_c[None, :])
            ll_c = float(np.sum(X * np.log(P_c + 1e-12) + (1 - X) * np.log(1 - P_c + 1e-12)))
            if ll_c + 1e-10 >= prev_ll:
                theta, b = theta_c, b_c
                ll = ll_c
                break
            lr_loc *= 0.5
            if lr_loc < 1e-6:
                # Слишком маленький шаг — остаёмся на месте
                theta, b = theta0, b0
                ll = prev_ll
                break
        lr = lr_loc

        # Критерии сходимости
        max_delta = max(float(np.max(np.abs(theta - theta0))), float(np.max(np.abs(b - b0))))
        # Градиенты для критерия стационарности (инфинити-норма)
        grad_theta = np.sum(X - P, axis=1)
        grad_b = -np.sum(X - P, axis=0)
        grad_inf = float(max(np.max(np.abs(grad_theta)), np.max(np.abs(grad_b))))
        rel_ll = float(abs(ll - prev_ll) / (abs(prev_ll) + 1e-9))
        if (rel_ll < max(1e-6, 0.2 * tol)) or (max_delta < max(2e-3, 8 * tol)) or (grad_inf < 0.1 * J):
            converged = True
            prev_ll = ll
            break

        prev_ll = ll
        # Плавное восстановление шага
        lr = min(1.0, lr * 1.05)

    return {"theta": theta.astype(float), "b": b.astype(float), "converged": bool(converged), "n_iter": int(it)}


def run_psychometrics_pipeline(
    df: pd.DataFrame,
    item_cols: List[str],
    qc_params: Dict[str, float] | None = None,
    cfa_groups: Optional[List[int]] = None,
    invariance_by: Optional[str] = None,
    dif_group_by: Optional[str] = None,
    irt: bool = False,
    efa_n_factors: Optional[int] = None,
) -> Dict[str, object]:
    """Универсальный пайплайн психометрики.
    Аргументы:
      - df: DataFrame с ответами и метаданными
      - item_cols: список колонок с пунктами
      - qc_params: словарь параметров QC:
          {'duration_col': str|None, 'normative_sec': float (120), 'cutoff_ratio': float (0.5),
           'max_allowed_run': int (8), 'min_completion': float (0.9), 'qc_min_score': float (0.5),
           'checks': List[Tuple[int, float]] ([]) }
      - cfa_groups: список длиной n_items с номерами факторов (0..k-1) для CFA, опционально
      - invariance_by: имя столбца с группой для проверки инвариантности (по конгруэнтности Такера)
      - dif_group_by: имя столбца с группой для DIF (линейный скрининг)
      - irt: если True, попытаться запустить Rasch JML для бинарных пунктов
      - efa_n_factors: если задано, использовать это число факторов в EFA/PCA; иначе k из cfa_groups или 1
    Возвращает словарь с ключами: 'qc_report', 'reliability', 'efa', опционально 'cfa', 'invariance', 'dif', 'irt'
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    if not item_cols:
        raise ValueError("item_cols must be a non-empty list of column names")
    for c in item_cols:
        if c not in df.columns:
            raise ValueError(f"item column not found: {c}")

    qp = dict(qc_params or {})
    duration_col = qp.get('duration_col', None)
    normative_sec = float(qp.get('normative_sec', 120.0))
    cutoff_ratio = float(qp.get('cutoff_ratio', 0.5))
    max_allowed_run = int(qp.get('max_allowed_run', 8))
    min_completion = float(qp.get('min_completion', 0.9))
    qc_min_score = float(qp.get('qc_min_score', 0.5))
    checks = qp.get('checks', []) or []

    # QC по строкам
    qc_rows: List[Dict[str, float]] = []
    for idx, row in df.iterrows():
        values = [row[c] for c in item_cols]
        dur = float(row[duration_col]) if (duration_col is not None and duration_col in df.columns and pd.notna(row[duration_col])) else normative_sec
        qc = qc_summary(values, duration_sec=dur, normative_sec=normative_sec, checks=checks,
                        max_allowed_run=max_allowed_run, min_completion=min_completion, cutoff_ratio=cutoff_ratio)
        qc_rows.append(qc)
    qc_df = pd.DataFrame(qc_rows).reset_index(drop=True)

    # Фильтрация по качеству
    mask_good = (qc_df['qc_score'] >= qc_min_score).values
    X_all = df[item_cols].values.astype(float)
    X_filt = X_all[mask_good] if mask_good.sum() >= 2 else X_all  # fallback, если слишком жёсткий порог

    # Импутация пропусков для устойчивости корреляций/EFA: заменяем NaN на медиану по колонке
    if np.isnan(X_filt).any():
        col_med = np.nanmedian(X_filt, axis=0)
        # Если вся колонка NaN, подставим центральное значение шкалы 4.0 как безопасный дефолт
        col_med = np.where(np.isnan(col_med), 4.0, col_med)
        nan_rows, nan_cols = np.where(np.isnan(X_filt))
        if nan_rows.size > 0:
            X_filt[nan_rows, nan_cols] = col_med[nan_cols]

    # Надёжность
    alpha = cronbach_alpha(X_filt.tolist())
    omega = mcdonald_omega_total(X_filt.tolist())
    reliability = {"alpha": float(alpha), "omega": float(omega), "n_used": int(X_filt.shape[0])}

    # EFA/PCA
    if efa_n_factors is None:
        if cfa_groups is not None and len(cfa_groups) == len(item_cols):
            efa_k = int(max(cfa_groups)) + 1
        else:
            efa_k = 1
    else:
        efa_k = int(efa_n_factors)
    efa = efa_pca(X_filt.tolist(), n_factors=max(1, efa_k))

    result: Dict[str, object] = {
        "qc_report": qc_df,
        "reliability": reliability,
        "efa": efa,
    }

    # CFA (если указано разбиение пунктов по факторам)
    if cfa_groups is not None:
        if len(cfa_groups) != len(item_cols):
            raise ValueError("cfa_groups length must equal number of item_cols")
        cfa_res = cfa_ls(X_filt.tolist(), cfa_groups)
        result["cfa"] = cfa_res

    # Инвариантность (по конгруэнтности Такера) между группами респондентов
    if invariance_by is not None and invariance_by in df.columns:
        groups_unique = [g for g in pd.unique(df[invariance_by]) if pd.notna(g)]
        if len(groups_unique) >= 2:
            group_mats: Dict[str, List[List[float]]] = {}
            for g in groups_unique:
                sub = df.loc[df[invariance_by] == g, item_cols].values.astype(float)
                if sub.shape[0] >= 5:
                    # Импутация NaN медианой колонок для устойчивой корреляции/EFA внутри группы
                    sub_imp = sub.copy()
                    if np.isnan(sub_imp).any():
                        med = np.nanmedian(sub_imp, axis=0)
                        med = np.where(np.isnan(med), 4.0, med)
                        rr, cc = np.where(np.isnan(sub_imp))
                        if rr.size > 0:
                            sub_imp[rr, cc] = med[cc]
                    group_mats[str(g)] = sub_imp.tolist()
            if len(group_mats) >= 2:
                inv = check_invariance_by_congruence(group_mats, n_factors=max(1, efa_k), threshold=float(qp.get('invariance_threshold', 0.9)))
                result["invariance"] = inv

    # DIF (линейный скрининг)
    if dif_group_by is not None and dif_group_by in df.columns:
        groups_vec_all = pd.Categorical(df[dif_group_by]).codes
        groups_vec = groups_vec_all[mask_good] if mask_good.sum() >= 2 else groups_vec_all
        if X_filt.shape[0] >= 5:
            dif_df = dif_linear(X_filt.tolist(), groups_vec.tolist(), alpha=float(qp.get('dif_alpha', 0.05)))
            result["dif"] = dif_df

    # IRT Rasch (если бинарные ответы)
    if irt:
        try:
            Xb = X_filt
            if not np.all((Xb == 0) | (Xb == 1)):
                # Попробуем бинаризовать по медиане (быстрый хак для диагностики)
                med = np.nanmedian(Xb, axis=0)
                Xb = (Xb >= med).astype(int)
            irt_res = irt_rasch_jml(Xb.tolist(), max_iter=int(qp.get('irt_max_iter', 200)), tol=float(qp.get('irt_tol', 1e-3)))
            result["irt"] = irt_res
        except Exception as e:
            result["irt"] = {"error": str(e)}

    return result


def score_teique(items: List[float], reverse_idx: List[int], scale_min: float, scale_max: float,
                 ref_mean: float, ref_sd: float) -> Dict[str, float]:
    """
    Скоринг TEIQue (Trait-EI):
    - Реверс по ключу
    - Среднее по фасетам/пунктам как Trait EI (упрощённая агрегация)
    - z-нормировка по референсу
    - Преобразование в 0-200
    """
    vals = np.array(reverse_likert(items, reverse_idx, scale_min, scale_max), dtype=float)
    trait_ei = float(np.mean(vals))
    z = compute_z(trait_ei, ref_mean, ref_sd)
    eq200 = z_to_0_200(z)
    return {"trait_ei_mean": trait_ei, "z_eq": z, "EQ_200": eq200}


def score_wleis(subscales: Dict[str, List[float]], reverse_idx: Dict[str, List[int]],
                scale_min: float, scale_max: float, ref_mean: float, ref_sd: float) -> Dict[str, float]:
    """
    Скоринг WLEIS: 4 субшкалы (SEA, OEA, UOE, ROE) → средние по субшкалам → общий индекс как среднее субшкал
    Далее: z-нормировка и преобразование в 0-200
    """
    sub_means: Dict[str, float] = {}
    all_means: List[float] = []
    for name, items in subscales.items():
        r_idx = reverse_idx.get(name, [])
        vals = np.array(reverse_likert(items, r_idx, scale_min, scale_max), dtype=float)
        m = float(np.mean(vals))
        sub_means[name] = m
        all_means.append(m)
    total_mean = float(np.mean(all_means))
    z = compute_z(total_mean, ref_mean, ref_sd)
    eq200 = z_to_0_200(z)
    return {**sub_means, "total_mean": total_mean, "z_eq": z, "EQ_200": eq200}


def score_esci(self_scores: Optional[List[float]] = None, others_scores: Optional[List[List[float]]] = None,
               ref_mean: float = 0.0, ref_sd: float = 1.0) -> Dict[str, float]:
    """
    Скоринг ESCI (360):
    - Усреднение self и усреднение по рейтёрам others (если присутствуют)
    - Комбинированный индекс как среднее из доступных компонентов
    - z-нормировка и преобразование в 0-200
    Примечание: надёжность по рейтёрам (ICC) и веса self/others задаются вне этой функции на уровне анализа.
    """
    comps: List[float] = []
    res: Dict[str, float] = {}
    if self_scores is not None and len(self_scores) > 0:
        self_mean = float(np.mean(self_scores))
        res["self_mean"] = self_mean
        comps.append(self_mean)
    if others_scores is not None and len(others_scores) > 0:
        others_means = [float(np.mean(r)) for r in others_scores if len(r) > 0]
        if len(others_means) > 0:
            others_mean = float(np.mean(others_means))
            res["others_mean"] = others_mean
            comps.append(others_mean)
    if len(comps) == 0:
        raise ValueError("At least one of self_scores or others_scores must be provided")
    combined = float(np.mean(comps))
    z = compute_z(combined, ref_mean, ref_sd)
    eq200 = z_to_0_200(z)
    res.update({"combined_mean": combined, "z_eq": z, "EQ_200": eq200})
    return res


def score_bias(subscales: Dict[str, List[float]], reverse_idx: Dict[str, List[int]],
               scale_min: float, scale_max: float, ref_mean: float, ref_sd: float, gamma_B: float = 2.0) -> Dict[str, float]:
    """
    Скоринг BIAS-Scale:
    - Реверс по ключам внутри каждой субшкалы
    - Среднее по каждой субшкале, затем общий B_raw как среднее субшкал
    - Нормирование: z_B по референсу и логистическое преобразование B = σ(γ_B·z_B)
    """
    sub_means: List[float] = []
    for name, items in subscales.items():
        r_idx = reverse_idx.get(name, [])
        vals = np.array(reverse_likert(items, r_idx, scale_min, scale_max), dtype=float)
        sub_means.append(float(np.mean(vals)))
    b_raw = float(np.mean(sub_means))
    z_b = compute_z(b_raw, ref_mean, ref_sd)
    b = logistic(gamma_B * z_b)
    return {"B_raw": b_raw, "z_B": z_b, "B": b}


# === Утилиты контроля качества данных (QC) ===

def longest_run(values: List[Optional[float]]) -> int:
    """Максимальная длина подряд идущих одинаковых значений (longstring)."""
    if values is None or len(values) == 0:
        return 0
    run = 1
    max_run = 1
    last = values[0]
    for v in values[1:]:
        if v == last:
            run += 1
        else:
            max_run = max(max_run, run)
            run = 1
            last = v
    max_run = max(max_run, run)
    return max_run


def detect_longstring(values: List[Optional[float]], max_allowed_run: int = 8) -> Dict[str, float]:
    """Детекция longstring: возвращает длину серии и флаг превышения порога."""
    lr = longest_run(values)
    flag = float(lr > max_allowed_run)
    return {"longest_run": float(lr), "flag_longstring": flag}


def detect_speeder(duration_sec: float, normative_sec: float, cutoff_ratio: float = 0.5) -> Dict[str, float]:
    """Флаг спидера: длительность < cutoff_ratio × нормативного времени."""
    if normative_sec <= 0:
        raise ValueError("normative_sec must be > 0")
    flag = float(duration_sec < cutoff_ratio * normative_sec)
    return {"duration_sec": float(duration_sec), "normative_sec": float(normative_sec), "flag_speeder": flag}


def completion_rate(values: List[Optional[float]]) -> float:
    """Доля ненулевых/непропущенных ответов (NaN/None считаем пропуском)."""
    arr = pd.Series(values)
    non_missing = arr.notna().sum()
    return float(non_missing / max(1, len(arr)))


def is_complete(values: List[Optional[float]], min_completion: float = 0.9) -> Dict[str, float]:
    """Проверка минимальной доли заполнения."""
    cr = completion_rate(values)
    flag = float(cr < min_completion)
    return {"completion_rate": cr, "flag_low_completion": flag}