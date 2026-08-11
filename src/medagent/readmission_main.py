from medagent.config import READMISSION_FEATURE_COLUMNS
import pandas as pd
from medagent.feature_engineering import (
    build_readmission_experiment_table,
)
from medagent.modeling import (
    evaluate_readmission_model,
    evaluate_readmission_threshold_grid,
    select_recall_first_threshold,
    split_readmission_train_validation_test,
    train_readmission_pipeline,
)
from medagent.persist import (
    build_readmission_bundle,
    save_readmission_bundle,
)


def run_readmission_recall_pipeline() -> None:
    """
    Train and evaluate the recall-first 30-day readmission model.

    Test data remains untouched until the final evaluation.
    """

    print("Step 1/7: Building the readmission feature table...")
    readmit_cohort = build_readmission_experiment_table()

    missing_columns = sorted(
        set(READMISSION_FEATURE_COLUMNS)
        - set(readmit_cohort.columns)
    )

    if missing_columns:
        raise ValueError(
            "Readmission features missing from cohort: "
            f"{missing_columns}"
        )

    print("Step 2/7: Creating patient-safe train/validation/test splits...")
    train_cohort, validation_cohort, test_cohort = (
        split_readmission_train_validation_test(readmit_cohort)
    )

    print("Step 3/7: Training the validation model...")
    validation_pipeline = train_readmission_pipeline(
        train_cohort,
        READMISSION_FEATURE_COLUMNS,
    )

    print("Step 4/7: Selecting a recall-first alert threshold...")
    threshold_results = evaluate_readmission_threshold_grid(
        validation_pipeline,
        validation_cohort,
        READMISSION_FEATURE_COLUMNS,
    )

    chosen_threshold = select_recall_first_threshold(
        threshold_results
    )

    chosen_row = threshold_results.loc[
        threshold_results["threshold"] == chosen_threshold
    ].iloc[0]

    print(
        f"Chosen threshold: {chosen_threshold:.2f} "
        f"| validation recall: {chosen_row['recall']:.3f} "
        f"| validation precision: {chosen_row['precision']:.3f}"
    )
    
    print("Step 5/7: Retraining on train + validation data...")
    final_training_cohort = pd.concat(
    [train_cohort, validation_cohort],
    ignore_index=True,
)

    final_pipeline = train_readmission_pipeline(
        final_training_cohort,
        READMISSION_FEATURE_COLUMNS,
    )

    print("Step 6/7: Evaluating once on untouched final test data...")
    final_metrics = evaluate_readmission_model(
        final_pipeline,
        test_cohort,
        READMISSION_FEATURE_COLUMNS,
        chosen_threshold,
    )

    split_subject_ids = {
        "train": train_cohort["subject_id"].unique().tolist(),
        "validation": validation_cohort["subject_id"]
        .unique()
        .tolist(),
        "test": test_cohort["subject_id"].unique().tolist(),
    }

    print("Step 7/7: Saving complete readmission model bundle...")
    bundle = build_readmission_bundle(
        pipeline=final_pipeline,
        selected_feature_columns=READMISSION_FEATURE_COLUMNS,
        chosen_threshold=chosen_threshold,
        final_metrics=final_metrics,
        split_subject_ids=split_subject_ids,
        threshold_results=threshold_results,
    )

    bundle_path = save_readmission_bundle(bundle)

    print("\n=== Final Readmission Model Results ===")
    print(f"Threshold: {final_metrics['threshold']:.2f}")
    print(f"Accuracy:  {final_metrics['accuracy']:.3f}")
    print(f"Precision: {final_metrics['precision']:.3f}")
    print(f"Recall:    {final_metrics['recall']:.3f}")
    print(f"F1:        {final_metrics['f1']:.3f}")
    print(f"ROC-AUC:   {final_metrics['roc_auc']:.3f}")
    print(f"PR-AUC:    {final_metrics['pr_auc']:.3f}")
    print(f"\nSaved model bundle to:\n{bundle_path}")


if __name__ == "__main__":
    run_readmission_recall_pipeline()