#!/bin/bash
# post_train_push.sh
# Usage: bash post_train_push.sh
# Description: Adds, commits, and pushes updated model artifacts to the Git repo after retraining.

set -e

MODEL_FILES="models/xgboost_tuned_latest.pkl models/scaler.pkl models/feature_columns.pkl"

# Add model files to git
for file in $MODEL_FILES; do
    if [ -f "$file" ]; then
        git add "$file"
        echo "Added $file to git staging."
    else
        echo "Warning: $file not found, skipping."
    fi
done

git commit -m "Update model artifacts after retraining"
git push

echo "Model files committed and pushed successfully." 