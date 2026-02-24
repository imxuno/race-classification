# 인종 분류기

다양한 인종의 얼굴 데이터셋에서 동양인 얼굴만 추출하여 분류

이후, 홍조/모공/주름 3map 자동 레이블링 데이터셋으로 활용

## 환경 (conda)

```plaintext
- python: 3.10.18
```

### 실행 예:

```bash
cd src
python src\filter_asian.py
  --input_root "C:\..."
  --output_root "C:\..."
  --model_path "C:\..."
  --device "cuda:0" --batch_size 32 --threshold 0.7 --detector_ctx_id 0
```

정렬 끄고 로그만 남기기(복사X)
```bash
cd src
python filter_asian.py ... --no_align --no_copy
```

경계 샘플 저장(임계값 +-0.05)
```bash
cd src
python filter_asian.py ... --borderline_save --borderline_margin 0.05
```

임계값 스윕 (0.50~0.90, 0.01 간격) + 차트 생성
```bash
cd src
python scripts\sweep_thresholds.py ^
  --log_csv "D:\cGAN_datasets\filtered_CelebAMask-HQ\filter_log.csv" ^
  --threshold_start 0.50 --threshold_end 0.90 --threshold_step 0.01 ^
  --make_plots
```

GT가 있을 경우 (파일 열은 'file' 기준 매칭)
```bash
cd src
python scripts\sweep_thresholds.py ^
  --log_csv "D:\cGAN_datasets\filtered_CelebAMask-HQ\filter_log.csv" ^
  --gt_csv "D:\gt\celebamask_asian_gt.csv" ^
  --join_on file ^
  --make_plots
```

GT가 베이스네임만 있는 경우
```bash
cd src
python scripts\sweep_thresholds.py ^
  --log_csv "D:\cGAN_datasets\filtered_CelebAMask-HQ\filter_log.csv" ^
  --gt_csv "D:\gt\celebamask_asian_gt_basename.csv" ^
  --join_on basename ^
  --make_plots
```