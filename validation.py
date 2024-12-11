# validation.py

def validate_model(training_results, baseline_results):
    """
    Validate the model's performance against baseline strategies.

    Parameters:
    - training_results: dict, results from the curriculum training process.
    - baseline_results: dict, results from baseline models or strategies.

    Returns:
    - validation_summary: dict, a summary of the validation comparison.
    """
    try:
        validation_summary = {
            "comparison": {},
            "observations": []
        }

        # Compare overall metrics
        for metric in training_results["overall_metrics"]:
            curriculum_metric = training_results["overall_metrics"][metric]
            baseline_metric = baseline_results["overall_metrics"].get(metric, None)

            if baseline_metric is not None:
                improvement = (curriculum_metric - baseline_metric) / baseline_metric * 100
                validation_summary["comparison"][metric] = {
                    "curriculum": curriculum_metric,
                    "baseline": baseline_metric,
                    "improvement (%)": improvement
                }

        # Analyze stage-wise progression
        for stage in training_results["stages"]:
            stage_id = stage["stage_id"]
            curriculum_loss = stage["metrics"]["loss"]
            curriculum_accuracy = stage["metrics"]["accuracy"]

            baseline_stage = next((s for s in baseline_results["stages"] if s["stage_id"] == stage_id), None)
            if baseline_stage:
                baseline_loss = baseline_stage["metrics"]["loss"]
                baseline_accuracy = baseline_stage["metrics"]["accuracy"]

                validation_summary["observations"].append({
                    "stage_id": stage_id,
                    "curriculum": {
                        "loss": curriculum_loss,
                        "accuracy": curriculum_accuracy
                    },
                    "baseline": {
                        "loss": baseline_loss,
                        "accuracy": baseline_accuracy
                    },
                    "improvement": {
                        "loss (%)": (baseline_loss - curriculum_loss) / baseline_loss * 100,
                        "accuracy (%)": (curriculum_accuracy - baseline_accuracy) / baseline_accuracy * 100
                    }
                })

        print("Validation completed successfully.")
        return validation_summary

    except Exception as e:
        print(f"Error during validation: {e}")
        raise
