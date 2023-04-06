#!/bin/bash --login

# Enable strict mode.
set -euo pipefail

cd /app/Web_app

# Temporarily disable strict mode and activate conda:
set +euo pipefail
conda activate peps

# Re-enable strict mode:
set -euo pipefail

# exec the final command:
exec streamlit run app.py 
