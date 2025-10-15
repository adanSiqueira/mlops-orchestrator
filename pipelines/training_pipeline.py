from zenml import pipeline
from steps.ingesting_data import ingest_data_step
from steps.cleaning_data import clean_data_step
from steps.training_model import train_model_step
from steps.evaluating_model import evaluate_model_step

@pipeline(enable_cache=False)
def training_pipeline(data_path: str):
    """A simple training pipeline."""

    raw_data = ingest_data_step(data_path)
    X_train, X_test, y_train, y_test = clean_data_step(data=raw_data)
    model = train_model_step(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    R2, RMSE, MSE = evaluate_model_step(model=model, X_test=X_test, y_test=y_test)