from data_handler.bidmc_hanlder import BIDMC_Handler
from data_handler.capnobase_handler import CAPNOBASE_Handler


DATASET_REGISTRY = {
    "bidmc": BIDMC_Handler,
    "capnobase": CAPNOBASE_Handler,
}


def create_handler(dataset_name, data_path):
    key = dataset_name.lower()
    try:
        handler_cls = DATASET_REGISTRY[key]
    except KeyError as exc:
        available = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. Available datasets: {available}"
        ) from exc
    return handler_cls(data_path)
