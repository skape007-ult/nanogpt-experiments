"""Extract The Office dialogue from a Kaggle CSV into a tagged text file.

Usage:
    python data/prepare_office.py --csv path/to/The-Office-Lines-V4.csv --output input_texts/the_office.txt
"""
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Prepare The Office dialogue dataset")
    parser.add_argument('--csv', type=str, required=True,
                        help="Path to The-Office-Lines-V4.csv from Kaggle")
    parser.add_argument('--output', type=str, default='input_texts/the_office.txt',
                        help="Output text file path")
    args = parser.parse_args()

    print("Loading the Dunder Mifflin archives...")
    df = pd.read_csv(args.csv)
    df = df.dropna(subset=['speaker', 'line'])

    with open(args.output, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            speaker = str(row['speaker']).strip().upper().replace(" ", "_")
            speaker = ''.join(c for c in speaker if c.isalnum() or c == '_')
            line = str(row['line']).strip()
            f.write(f"<{speaker}> {line}\n")

    print(f"Extracted {len(df)} lines of dialogue to {args.output}")


if __name__ == '__main__':
    main()
