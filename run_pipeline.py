from pipelines.training_pipeline import train_pipeline


# run_pipeline.py
if __name__ == "__main__":
    train_pipeline(data_path="data")          # folder not file

    print("pipeline execution completed.")


