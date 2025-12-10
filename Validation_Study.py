
"""
Validation_Study.py
Формальная валидация модели G (v0.3 Candidate).
Содержит 3 теста, запрашиваемых для финальной публикации:
1. Анализ распределения (Monte Carlo).
2. Визуализация фазового перехода (Heatmap D vs A).
3. Анализ чувствительности параметров (Alpha Sensitivity).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from Calc_G_v03_Candidate import StupidityCalculatorV3, PersonProfileV3

def generate_population(n=10000) -> List[PersonProfileV3]:
    """Генерирует N случайных агентов с реалистичными корреляциями"""
    # IQ нормальное распределение (100, 15), но нормированное к 0-200
    iq_raw = np.random.normal(100, 15, n)
    iq_raw = np.clip(iq_raw, 50, 160) # Обрезаем экстремумы для реализма
    
    # EQ коррелирует слабо с IQ (r=0.1)
    eq_raw = 0.1 * iq_raw + 0.9 * np.random.normal(100, 15, n)
    eq_raw = np.clip(eq_raw, 40, 160)
    
    # A (Attention) - 0-1. Бета-распределение (склонность к середине)
    a_raw = np.random.beta(5, 5, n) 
    
    # B_err, B_mot - случайные 0-1
    b_err = np.random.beta(2, 5, n) # Чаще низкие ошибки
    b_mot = np.random.beta(4, 4, n) # Распределены нормально вокруг 0.5
    
    profiles = []
    for i in range(n):
        p = PersonProfileV3(
            IQ=iq_raw[i],
            EQ=eq_raw[i],
            CQ=np.random.normal(100, 15),
            A=a_raw[i],
            B_err=b_err[i],
            B_mot=b_mot[i],
            D=np.random.beta(4, 2), # Среда склонна к шуму (mean ~ 0.66)
            S=np.random.beta(3, 3), # Соц. давление среднее
            R=np.random.random()
        )
        profiles.append(p)
    return profiles

def test_1_distribution(profiles):
    """Test 1: Monte Carlo Distribution Analysis"""
    calc = StupidityCalculatorV3()
    g_values = [calc.calculate_g(p) for p in profiles]
    
    # Plot Hist
    plt.figure(figsize=(10, 6))
    sns.histplot(g_values, kde=True, bins=50, color='teal')
    plt.axvline(1.0, color='red', linestyle='--', label='Singularity (G=1.0)')
    plt.xlabel('G-Factor')
    plt.title(f'Monte Carlo Distribution of G (N={len(profiles)})')
    plt.legend()
    plt.savefig('figure_dist_g.png')
    
    # Stats
    mean_g = np.mean(g_values)
    median_g = np.median(g_values)
    p95 = np.percentile(g_values, 95)
    
    return {
        "mean": mean_g, 
        "median": median_g, 
        "std": np.std(g_values),
        "p95": p95,
        "n_critical": sum(1 for x in g_values if x > 1.0) / len(g_values)
    }

def test_2_phase_transition():
    """Test 2: Heatmap D vs A (Phase Transition)"""
    # Grid search
    d_vals = np.linspace(0.01, 1.0, 50)
    a_vals = np.linspace(0.01, 1.0, 50)
    
    grid = np.zeros((50, 50))
    calc = StupidityCalculatorV3()
    
    # Fixed average person
    base_p = PersonProfileV3(IQ=110, EQ=100, CQ=100, A=0.5, B_err=0.3, B_mot=0.3, D=0.5, S=0.5, R=0.5)
    
    for i, d in enumerate(d_vals):
        for j, a in enumerate(a_vals):
            # Attention A varies (X axis), D varies (Y axis)
            base_p.A = a
            base_p.D = d
            grid[i, j] = calc.calculate_g(base_p)
            
    # Plot
    plt.figure(figsize=(8, 6))
    # Origin lower left
    plt.imshow(grid, origin='lower', extent=[0,1,0,1], aspect='auto', cmap='magma')
    plt.colorbar(label='G-Factor')
    plt.xlabel('Attention Control (A)')
    plt.ylabel('Digital Noise (D)')
    plt.title('Phase Transition: G(D, A)')
    plt.axhline(0.7, color='white', linestyle=':', label='Non-linear Threshold')
    plt.savefig('figure_heatmap_d_a.png')
    return "Heatmap generated"

def test_3_sensitivity(profiles):
    """Test 3: Sensitivity to Alpha Weights"""
    base_calc = StupidityCalculatorV3(0.4, 0.35, 0.25)
    base_g = np.mean([base_calc.calculate_g(p) for p in profiles])
    
    results = {}
    
    # Perturb Alpha 1
    c1 = StupidityCalculatorV3(0.44, 0.35, 0.25) # +10%
    g1 = np.mean([c1.calculate_g(p) for p in profiles])
    results['alpha1_sens'] = (g1 - base_g) / base_g
    
    # Perturb Alpha 2
    c2 = StupidityCalculatorV3(0.4, 0.385, 0.25) # +10%
    g2 = np.mean([c2.calculate_g(p) for p in profiles])
    results['alpha2_sens'] = (g2 - base_g) / base_g

    # --- Additional Scenarios for Paper ---
    calc = StupidityCalculatorV3()

    # 3. The Bureaucrat (High S, Low C)
    # S=0.9, C=20 (Low), EQ=100. D=0.3 (Low noise). I=100.
    p_bur = PersonProfileV3(IQ=100, EQ=100, CQ=20, A=0.5, 
                            B_err=0.2, B_mot=0.2, 
                            D=0.3, S=0.9, R=0.5)
    g_bur = calc.calculate_g(p_bur)
    print(f"Scenario 'Bureaucrat': G = {g_bur:.4f}")

    # 4. The Resilient Operator (High D, High A)
    # D=0.95, but A=0.9 (High Focus). S=0.5.
    p_res = PersonProfileV3(IQ=120, EQ=120, CQ=120, A=0.9,
                            B_err=0.1, B_mot=0.1,
                            D=0.95, S=0.5, R=0.9)
    g_res = calc.calculate_g(p_res)
    print(f"Scenario 'Resilient': G = {g_res:.4f}")

    results['bureaucrat_g'] = g_bur
    results['resilient_g'] = g_res
    
    return results

def main():
    print("Running Validation Study (v0.3)...")
    
    # 1. Generate Data
    pop = generate_population(10000)
    print(f"Generated {len(pop)} synthetic agents.")
    
    # 2. Dist Analysis
    res1 = test_1_distribution(pop)
    print("\n[Test 1] Distribution Stats:")
    for k, v in res1.items():
        print(f"  {k}: {v:.4f}")
        
    # 3. Phase Transition
    test_2_phase_transition()
    print("\n[Test 2] Phase transition map saved to 'figure_heatmap_d_a.png'")
    
    # 4. Sensitivity
    res3 = test_3_sensitivity(pop)
    print("\n[Test 3] Sensitivity (+10% param -> % change in G):")
    for k, v in res3.items():
        print(f"  {k}: {v*100:.2f}%")

    # Save summary
    with open('validation_results.txt', 'w') as f:
        f.write("Validation Results v0.3\n")
        f.write(str(res1) + "\n")
        f.write(str(res3) + "\n")

if __name__ == "__main__":
    main()
