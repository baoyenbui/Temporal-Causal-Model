import sys
import os
import nbformat
from nbformat import read
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.temporal import TemporalArchitecture
from src.cf import Layer3CounterfactualExplanation
from src.evaluation import run_full_evaluation

clean_main_df: pd.DataFrame

with open("src/preprocessing.ipynb", "r", encoding="utf-8") as f:
    notebook = read(f, as_version=4)

exec_globals = globals()
for i, cell in enumerate(notebook.cells):
    if cell.cell_type == "code":
        try:
            exec(cell.source, exec_globals)
        except Exception as e:
            print(f"preprocessing.ipynb cell {i} failed: {type(e).__name__}: {e}")
            raise


def main(mapping_path: str = None):
    arch = TemporalArchitecture()
    (
        layer1_df,
        agg_df,
        causal_graphs,
        reporting_key,
        filtered_graph,
        key_status,
        keyed_layer1_df,
    ) = arch.run(clean_main_df.copy(), mapping_path=mapping_path)

    if reporting_key is None or filtered_graph is None or agg_df is None:
        print("Temporal architecture did not produce usable graph. Abort CF.")
        return None

    print("\n--- Layer 3: Counterfactual Explanation ---")
    layer3 = Layer3CounterfactualExplanation(causal_graph=filtered_graph, agg_df=agg_df)
    actionable_vars = layer3.identify_actionable_variables()

    cf_target = 'avg_success'
    current_val = layer3.get_current_state(cf_target)

    print(f"\n=== PRIMARY: realistic improvement of '{cf_target}' ===")
    sim_scored = layer3.evaluate_forward_simulation(target=cf_target, direction='increase', max_hops=2)
    if sim_scored:
        for r in sim_scored:
            print(f"  Push '{r['source_variable']}': {r['current_source_value']:.3f} -> "
                  f"{r['max_feasible_source_value']:.3f} (delta={r['max_feasible_delta']:+.3f})")
            print(f"      => '{cf_target}': {r['current_target_value']:.3f} -> "
                  f"{r['achievable_target_value']:.3f} (change={r['achievable_change']:+.3f})")
            print(f"      forward_cf_score={r['forward_cf_score']:.3f} | "
                  f"efficiency={r['efficiency']:.3f} | effect_snr={r['effect_snr']:.3f}")
    else:
        print(f"  No actionable path for '{cf_target}'.")

    print(f"\n=== SECONDARY: move '{cf_target}' toward one-step target ===")
    target_step = layer3.get_max_step_delta(cf_target)
    if not np.isfinite(target_step) or target_step <= 0:
        target_step = float(agg_df[cf_target].std())
    cf_threshold = current_val + target_step
    cf_results = layer3.generate_counterfactual(
        target=cf_target, direction='increase', threshold=cf_threshold,
        current_value=current_val, max_hops=2, top_k=3
    )

    if cf_results:
        print(f"\nCandidates ({current_val:.3f} -> {cf_threshold:.3f}):")
        for r in cf_results:
            print(f"  '{r['source_variable']}': {r['current_source_value']:.3f} -> "
                  f"{r['proposed_source_value']:.3f} (delta={r['delta']:+.3f}) "
                  f"path=[{r['causal_path']}] lag={r['lag_days']}d status={r['status']}")
            print(f"    cf_score={r['cf_score']:.3f} | flip_success_rate={r['flip_success_rate']:.3f} | "
                  f"prob_improvement={r['prob_improvement']:.3f}")
    else:
        print(f"\nNo actionable path for '{cf_target}'.")

    print("\n--- Layer 3: 5-Student Counterfactual Test ---")
    student_cf_df = layer3.generate_student_counterfactuals(
        layer1_df=layer1_df, target=cf_target, direction='increase',
        n_users=5, threshold_quantile=0.60, realistic_step=True
    )
    print(student_cf_df.to_string(index=False))

    print("\n--- Layer 3: Multi-Variable Counterfactual ---")
    multi_var_results = layer3.generate_multi_variable_counterfactual(
        target=cf_target, direction='increase', threshold=cf_threshold,
        current_value=current_val, max_hops=2, max_vars=2, top_k=3
    )
    if multi_var_results:
        for r in multi_var_results:
            names = " + ".join(p['source_variable'] for p in r['pushes'])
            print(f"  [{r['status']}] {names}: {r['current_target_value']:.3f} -> "
                  f"{r['achievable_target_value']:.3f} (cf_score={r['cf_score']:.3f})")
    else:
        print(f"  No independent 2-variable combination for '{cf_target}'.")

    print("\n--- Layer 3: Multi-Step Counterfactual ---")
    multi_step_results = layer3.generate_multi_step_counterfactual(
        target=cf_target, direction='increase', threshold=cf_threshold,
        current_value=current_val, max_hops=2, n_steps=3, top_k=3
    )
    if multi_step_results:
        for r in multi_step_results:
            print(f"  [{r['status']}] '{r['source_variable']}' x{r['n_steps_used']}: "
                  f"{r['current_target_value']:.3f} -> {r['achievable_target_value']:.3f} "
                  f"(cf_score={r['cf_score']:.3f})")
    else:
        print(f"  No feasible multi-step plan for '{cf_target}'.")

    print("\n--- Evaluation ---")
    treatment_id, question_id, year_id = reporting_key
    reporting_entity_layer1_df = keyed_layer1_df[
        (keyed_layer1_df['TreatmentLessonConstructId'] == treatment_id) &
        (keyed_layer1_df['QuestionConstructId'] == question_id) &
        (keyed_layer1_df['Year'] == year_id)
    ].copy()
    print(f"Bootstrapping on {len(reporting_entity_layer1_df)} windows for key {reporting_key}")
    evaluation_results = run_full_evaluation(reporting_entity_layer1_df, agg_df, filtered_graph)

    return {
        "layer1_df": layer1_df,
        "agg_df": agg_df,
        "causal_graphs": causal_graphs,
        "cf_results": cf_results,
        "student_cf_df": student_cf_df,
        "evaluation_results": evaluation_results,
        "key_status": key_status,
        "filtered_graph": filtered_graph,
        "reporting_key": reporting_key,
    }


if __name__ == "__main__":
    main()