uv run --env-file /home/rgautha1/projects/neural-dimensionality/.env jupyter nbconvert notebooks/*.ipynb --execute --inplace
uv run jupyter nbconvert notebooks/*.ipynb --clear-output --ClearMetadataPreprocessor.enabled=True --inplace
