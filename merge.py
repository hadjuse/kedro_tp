from pathlib import Path
import pandas as pd


def merge_csvs(
    input_dir: str = "data/03_primary/primary_csv",
    output_file: str = "data/03_primary/primary.csv",
) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_file)

    # Crée uniquement le dossier parent (pas output_file comme dossier)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prend tous les CSV sauf le fichier de sortie lui-même
    csv_files = sorted(f for f in input_path.glob("*.csv") if f.resolve() != output_path.resolve())

    if not csv_files:
        raise FileNotFoundError(f"Aucun CSV trouvé dans {input_path}")

    frames = [pd.read_csv(f) for f in csv_files]
    merged = pd.concat(frames, ignore_index=True)

    merged.to_csv(output_path, index=False)
    print(f"Fusion terminée: {len(csv_files)} fichiers -> {output_path}")


if __name__ == "__main__":
    merge_csvs()
