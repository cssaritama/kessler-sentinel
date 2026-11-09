"""Test client entrypoint for KESSLER-SENTINEL Conjunction Triage Service.

Aliases test_service.py for standard Zoomcamp evaluation conventions (python test.py).
"""

import sys
from test_service import DEFAULT_URL, run_verification

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    run_verification(url)
