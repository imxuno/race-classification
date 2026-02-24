# scripts/plot_from_log.py
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log_csv', type=str, required=True)
    ap.add_argument('--out_dir', type=str, default=None)
    args = ap.parse_args()

    log_csv = Path(args.log_csv)
    df = pd.read_csv(log_csv)
    if 'total_east' not in df.columns:
        df['total_east'] = df.get('east_prob', 0.0) + df.get('se_prob', 0.0)

    out_dir = Path(args.out_dir) if args.out_dir else log_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) total_east 히스토그램
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(df['total_east'], bins=50)
    ax.set_xlabel('P(East Asian) + P(Southeast Asian)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of total_east')
    fig.tight_layout()
    fig.savefig(out_dir / 'hist_total_east.png', dpi=150)
    plt.close(fig)

    # 2) (선택) east_prob vs se_prob 산점도
    if {'east_prob','se_prob'}.issubset(df.columns):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(df['east_prob'], df['se_prob'], s=6)
        ax.set_xlabel('P(East Asian)')
        ax.set_ylabel('P(Southeast Asian)')
        ax.set_title('Scatter: east_prob vs se_prob')
        fig.tight_layout()
        fig.savefig(out_dir / 'scatter_east_vs_se.png', dpi=150)
        plt.close(fig)

    print(f"[OK] Saved plots under: {out_dir}")

if __name__ == "__main__":
    main()
