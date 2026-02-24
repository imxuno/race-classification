# -----------------------------
# 임계값 스윕 스크립트
# --
# - 입력: filter_log.csv (필수)
# - 선택 입력: gt_csv (정답 라벨, 있으면 Precision/Recall/F1/PRCurve까지 계산)
# - 출력: sweep_results.csv (+ 옵션에 따라 PNG 차트 2~3장)
# -----------------------------
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 유틸: 안전한 분수 ---
def _safe_div(n, d):
    return (n / d) if d else 0.0

def load_log(log_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(log_csv)
    # 호환성: total_east 없으면 계산
    if 'total_east' not in df.columns:
        df['total_east'] = df.get('east_prob', 0.0) + df.get('se_prob', 0.0)
    # 파일명만으로 매칭할 때 대비
    df['file'] = df['file'].astype(str)
    df['basename'] = df['file'].apply(lambda p: os.path.basename(p))
    return df

def load_gt(gt_csv: Path, join_on: str = 'file') -> pd.DataFrame:
    """
    gt_csv 포맷(두 가지 중 하나):
        1) binary: columns -> [file or basename, gt_asian (0/1)]
        2) race:   columns -> [file or basename, race(str in {'East Asian','Southeast Asian',...})]
            -> East Asian 또는 Southeast Asian 이면 양성(1)으로 변환
    """
    g = pd.read_csv(gt_csv)
    # 컬럼 표준화
    col = join_on
    if col not in g.columns:
        # 베이스네임으로 매칭 옵션일 때 자동 보정
        if join_on == 'file' and 'basename' in g.columns:
            col = 'basename'
        elif join_on == 'basename' and 'file' in g.columns:
            # file -> basename 파생
            g['basename'] = g['file'].astype(str).apply(lambda p: os.path.basename(p))
            col = 'basename'
        else:
            raise ValueError(f"gt_csv에 '{join_on}'(또는 호환 컬럼)가 없습니다. 가진 컬럼: {list(g.columns)}")

    g[col] = g[col].astype(str)
    # gt_asian 만들기
    if 'gt_asian' in g.columns:
        g['gt_asian'] = g['gt_asian'].astype(int).clip(0,1)
    elif 'race' in g.columns:
        g['gt_asian'] = g['race'].astype(str).isin(['East Asian','Southeast Asian']).astype(int)
    else:
        raise ValueError("gt_csv에는 'gt_asian' (0/1) 또는 'race' 컬럼이 필요합니다.")
    return g[[col, 'gt_asian']].rename(columns={col: join_on})

def sweep_thresholds(df: pd.DataFrame, thresholds: np.ndarray, gt: pd.DataFrame = None, join_on: str = 'file') -> pd.DataFrame:
    """
    thresholds: 예) np.arange(0.50, 0.91, 0.01)
    gt: (선택) ground truth df; join_on 기준으로 병합
    """
    base = df.copy()
    total = len(base)

    # GT 병합
    has_gt = gt is not None
    if has_gt:
        # join_on 기준으로 병합 준비
        if join_on not in base.columns:
            if join_on == 'basename':
                base['basename'] = base['file'].astype(str).apply(lambda p: os.path.basename(p))
            else:
                raise ValueError(f"로그에 '{join_on}' 컬럼이 없습니다.")
        merged = base.merge(gt, on=join_on, how='left')
        # 라벨 누락은 0(음성)으로 보정하거나, 제외할지 정책 결정
        # 여기서는 제외(유효 GT)로 계산하되, 표에 N_gt 포함
        has_label = merged['gt_asian'].notna()
        merged = merged.loc[has_label].copy()
        merged['gt_asian'] = merged['gt_asian'].astype(int)
    else:
        merged = base

    rows = []
    for t in thresholds:
        selected = merged['total_east'] > t
        n_selected = int(selected.sum())
        ratio_selected = _safe_div(n_selected, len(merged)) if len(merged) else 0.0

        row = {
            'threshold': round(float(t), 6),
            'N_total': int(total),
            'N_eval': int(len(merged)),
            'selected': n_selected,
            'selected_ratio': ratio_selected
        }

        if has_gt:
            y_true = merged['gt_asian'] == 1
            tp = int(((selected) & (y_true)).sum())
            fp = int(((selected) & (~y_true)).sum())
            fn = int(((~selected) & (y_true)).sum())
            tn = int(((~selected) & (~y_true)).sum())

            prec = _safe_div(tp, tp + fp)
            rec  = _safe_div(tp, tp + fn)
            f1   = _safe_div(2*prec*rec, prec+rec) if (prec+rec) else 0.0
            tpr  = _safe_div(tp, tp + fn)  # same as recall
            fpr  = _safe_div(fp, fp + tn)

            row.update({
                'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
                'precision': prec, 'recall': rec, 'f1': f1,
                'tpr': tpr, 'fpr': fpr,
                'prevalence': _safe_div(int(y_true.sum()), len(merged)) if len(merged) else 0.0
            })

        rows.append(row)

    return pd.DataFrame(rows)

def make_plots(sweep_df: pd.DataFrame, out_dir: Path, has_gt: bool):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 선택 개수/비율 vs 임계값
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(sweep_df['threshold'], sweep_df['selected'], marker='o')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Selected Count')
    ax.set_title('Selected Count vs Threshold')
    fig.tight_layout()
    fig.savefig(out_dir / 'counts_vs_threshold.png', dpi=150)
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(sweep_df['threshold'], sweep_df['selected_ratio'], marker='o')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Selected Ratio')
    ax.set_title('Selected Ratio vs Threshold')
    fig.tight_layout()
    fig.savefig(out_dir / 'selected_ratio_vs_threshold.png', dpi=150)
    plt.close(fig)

    if has_gt and {'precision','recall'}.issubset(sweep_df.columns):
        # 2) 정밀도 / 재현율 vs 임계값
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(sweep_df['threshold'], sweep_df['precision'], marker='o', label='Precision')
        ax.plot(sweep_df['threshold'], sweep_df['recall'], marker='o', label='Recall')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title('Precision & Recall vs Threshold')
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / 'precision_recall_vs_threshold.png', dpi=150)
        plt.close(fig)

        # 3) PR curve (Recall-x / Precision-y)
        # threshold를 내리면 보통 recall이 증가하므로, recall 기준 정렬
        pr = sweep_df[['recall','precision']].dropna().sort_values('recall')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(pr['recall'], pr['precision'], marker='o')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        fig.tight_layout()
        fig.savefig(out_dir / 'pr_curve.png', dpi=150)
        plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Sweep thresholds from filter_log.csv and (optionally) GT to produce tables/plots.")
    ap.add_argument('--log_csv', type=str, required=True, help='Path to filter_log.csv')
    ap.add_argument('--output_csv', type=str, default=None, help='Where to save sweep_results.csv (default: same dir)')
    ap.add_argument('--threshold_start', type=float, default=0.55)
    ap.add_argument('--threshold_end', type=float, default=0.75)
    ap.add_argument('--threshold_step', type=float, default=0.01)
    ap.add_argument('--gt_csv', type=str, default=None, help='(Optional) CSV with ground truth')
    ap.add_argument('--join_on', type=str, choices=['file','basename'], default='file', help='Join key for GT merge')
    ap.add_argument('--make_plots', action='store_true', help='Also save PNG plots next to output_csv')
    args = ap.parse_args()

    log_csv = Path(args.log_csv)
    df = load_log(log_csv)

    # thresholds 구성 (end 포함하도록 조정)
    thresholds = np.arange(args.threshold_start, args.threshold_end + 1e-9, args.threshold_step)

    gt_df = None
    if args.gt_csv:
        gt_df = load_gt(Path(args.gt_csv), join_on=args.join_on)

    sweep_df = sweep_thresholds(df, thresholds, gt=gt_df, join_on=args.join_on)

    # 저장 경로
    if args.output_csv:
        out_csv = Path(args.output_csv)
        out_dir = out_csv.parent
    else:
        out_dir = log_csv.parent
        out_csv = out_dir / 'sweep_results.csv'

    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(out_csv, index=False, encoding='utf-8-sig')

    if args.make_plots:
        make_plots(sweep_df, out_dir, has_gt=(gt_df is not None))

    print(f"[OK] Saved: {out_csv}")
    if args.make_plots:
        print(f"[OK] Plots saved under: {out_dir}")

if __name__ == "__main__":
    main()