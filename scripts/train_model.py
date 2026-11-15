import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.supplylens.config import PRIORITY_CURRENCIES
from src.supplylens.multitask_training import train_joint_multicurrency_model


def main():
    parser = argparse.ArgumentParser(
        description="Train the MYR SupplyLens V2.2 joint multi-currency CNN-LSTM."
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=70)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-arima", action="store_true")
    parser.add_argument("--device", choices=["auto", "GPU", "CPU"], default="auto")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--calibration-stride", type=int, default=5)
    parser.add_argument(
        "--currencies", nargs="+", default=PRIORITY_CURRENCIES,
        help="Currencies trained jointly; default is all seven app currencies."
    )
    args = parser.parse_args()

    result = train_joint_multicurrency_model(
        currencies=args.currencies,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        force_download=args.force_download,
        include_arima=not args.skip_arima,
        calibration_stride=args.calibration_stride,
        device=args.device,
        require_gpu=args.require_gpu,
    )
    print("\\nJoint training complete:", result["model_path"])
    print("Currencies:", ", ".join(result["currencies"]))
    print("Epochs completed:", result["metadata"]["epochs_completed"])
    for currency, report in result["reports"].items():
        print(f"\\n{currency} — selected methods")
        print(report["selection_policy"].to_string(index=False))
        print(f"\\n{currency} — untouched final test")
        print(report["metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
