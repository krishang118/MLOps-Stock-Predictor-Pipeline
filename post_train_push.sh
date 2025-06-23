#!/bin/bash
set -e
MODEL_FILES="models/xgboost_tuned_latest.pkl models/scaler.pkl models/feature_columns.pkl"
for file in $MODEL_FILES; do
    if [ -f "$file" ]; then
        git add "$file"
        echo "Added $file to git staging."
    else
        echo "Warning: $file not found, skipping."
    fi
done
git commit -m "Update"
git push
echo "Model files committed and pushed successfully." 
