import pandas as pd
df = pd.read_csv(r"D:\cGAN_datasets\filtered_asian\filtered_CelebAMask-HQ\filter_log.csv")

print("총 행수(=총 이미지):", len(df))
print("\n에러 유형 카운트:")
print(df['error'].fillna('').value_counts().head(10))

print("\n검출 성공/실패 분포(det_score>0 기준):")
print((df['det_score']>0).value_counts())

# 'no_face' 샘플 몇 개 확인용
no_face = df[df['error']=='no_face'].head(50)['file'].tolist()
print("\nno_face 예시 5개:")
for f in no_face[:5]:
    print(f)

# imread 실패가 있는지
print("\nimread_failed 개수:", (df['error']=='imread_failed').sum())
