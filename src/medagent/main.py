from medagent.feature_engineering import build_experiment_table
from medagent.modeling import (
    add_extended_stay_target,
    evaluate_extended_stay_model,
    select_leakage_free_features,
    split_extended_stay_data,
    train_extended_stay_model,
)
from medagent.persist import (
    build_model_bundle,
    save_model_bundle,
)
from medagent.preprocessing import (
    fit_preprocessing_metadata,
    prepare_unencoded_features,
    transform_features,
)


def run_extended_stay_pipeline() -> None:
    """Build, train, evaluate, and save the extended-stay model bundle."""

    print("Step 1/7: Building the full feature table...")
    print("This scans large MIMIC files and may take a while.")
    df_experiment = build_experiment_table()

    print("Step 2/7: Applying fixed preprocessing rules...")
    df_prepared = prepare_unencoded_features(df_experiment)

    # Keep only rows with a known LOS target.
    df_prepared = df_prepared.dropna(subset=["los"]).copy()

    # Keep patient IDs separately for the patient-level split.
    patient_ids = df_experiment.loc[df_prepared.index, "subject_id"]

    print("Step 3/7: Creating the extended-stay target...")
    df_extended = add_extended_stay_target(df_prepared)

    print("Step 4/7: Splitting patients into train and test sets...")
    X_train, X_test, y_train, y_test = split_extended_stay_data(
        df_extended,
        patient_ids,
    )

    print("Step 5/7: Fitting preprocessing metadata on training data only...")
    preprocessing_metadata = fit_preprocessing_metadata(X_train)

    X_train = transform_features(X_train, preprocessing_metadata)
    X_test = transform_features(X_test, preprocessing_metadata)

    print("Step 6/7: Removing soft-leakage features and training XGBoost...")
    X_train = select_leakage_free_features(X_train)
    X_test = select_leakage_free_features(X_test)

    if not X_train.columns.equals(X_test.columns):
        raise ValueError(
            "Training and test feature columns do not match after preprocessing."
        )

    best_cols = X_train.columns.tolist()

    print(f"Training with {len(best_cols)} leakage-free features.")

    model = train_extended_stay_model(X_train, y_train)

    print("Step 7/7: Evaluating and saving the complete model bundle...")
    metrics = evaluate_extended_stay_model(model, X_test, y_test)

    metrics["n_train"] = int(len(X_train))
    metrics["n_test"] = int(len(X_test))
    metrics["n_features"] = int(len(best_cols))

    bundle = build_model_bundle(
        model=model,
        best_cols=best_cols,
        preprocessing_metadata=preprocessing_metadata,
        metrics=metrics,
    )

    bundle_path = save_model_bundle(bundle)

    print("\n=== Extended-Stay Model Results ===")
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1:        {metrics['f1']:.3f}")
    print(f"AUC-ROC:   {metrics['auc_roc']:.3f}")
    print(f"Features:  {metrics['n_features']}")
    print(f"\nSaved complete model bundle to:\n{bundle_path}")


if __name__ == "__main__":
    run_extended_stay_pipeline()
