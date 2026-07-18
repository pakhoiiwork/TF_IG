from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data_handler.bidmc_hanlder import BIDMC_Handler
from data_handler.capnobase_handler import CAPNOBASE_Handler


def summarize(records):
    if not records:
        return {"count": 0, "first_record": None}

    first = records[0]
    return {
        "count": len(records),
        "keys": list(first.keys()),
        "types": {key: type(value).__name__ for key, value in first.items()},
        "ppg_length": len(first["PPG"]),
    }


def main():
    bidmc_path = ROOT / "datasets" / "bidmc-ppg-and-respiration-dataset-1.0.0" / "bidmc_csv"
    capnobase_path = ROOT / "datasets" / "CapnoBase" / "data" / "csv"

    bidmc_handler = BIDMC_Handler(str(bidmc_path))
    capno_handler = CAPNOBASE_Handler(capnobase_path)

    bidmc_handler.get_raw_data_from_csv()
    capno_handler.get_raw_data_from_csv()

    bidmc_processed = bidmc_handler.process_data()
    capno_processed = capno_handler.process_data()

    bidmc_summary = summarize(bidmc_processed)
    capno_summary = summarize(capno_processed)

    same_keys = bidmc_summary["keys"] == capno_summary["keys"]
    same_key_set = set(bidmc_summary["keys"]) == set(capno_summary["keys"])

    print("BIDMC summary:", bidmc_summary)
    print("CapnoBase summary:", capno_summary)
    print("Same key order:", same_keys)
    print("Same key set:", same_key_set)


if __name__ == "__main__":
    main()