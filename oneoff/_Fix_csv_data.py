import os
import pandas as pd

def convert_percent_column(df, col_name, str_na_to_minus_one=False):
    """ 
    %가 붙은 str 컬럼을 float으로 변환
    - 정수값은 100으로 나누고
    - 'NA'는 -1.0로 처리
    - 이미 0~1 사이의 값은 그대로 유지
    """
    if col_name not in df.columns:
        return df

    # 2. 문자열인 경우: '%' 제거 → 숫자로
    if pd.api.types.is_string_dtype(df[col_name]):
        df[col_name] = df[col_name].str.replace('%', '', regex=False)
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')

    # 3. 모든 값 numeric으로 변환
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')

    # 4. 값이 -1이면 그대로 유지, 나머지 중 1보다 큰 값만 100으로 나누기
    df[col_name] = df[col_name].apply(lambda x: x if x == -1 else (x / 100.0 if x > 0 else x))

    return df

def convert_percent_column_league(df, col_name, str_na_to_minus_one=False):
    
    # 1. NA 문자열을 -1로 변환
    if str_na_to_minus_one:
        df[col_name] = df[col_name].apply(lambda x: -1.0 if isinstance(x, str) and x.strip().upper() == "NA" else x)

    if pd.api.types.is_string_dtype(df[col_name]):
        df[col_name] = df[col_name].str.replace('%', '', regex=False)
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')

    # 3. 모든 값 numeric으로 변환
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')

    # 4. 값이 -1이면 그대로 유지, 나머지 중 1보다 큰 값만 100으로 나누기
    df[col_name] = df[col_name].apply(lambda x: x if x == -1 else (x / 100.0 if x > 1 else x))

    return df

def validate_and_print(df, file_path, expected_header, expected_types):
    print(f"\n✅ [{file_path}]")

    # 헤더 체크
    if list(df.columns) != expected_header:
        print(f"  ❌ 헤더 불일치: {df.columns.tolist()} → {expected_header}")
    else:
        print(f"  ✅ 헤더 확인 완료")

    # 타입 체크
    type_errors = []
    for col, expected_type in expected_types.items():
        if col not in df.columns:
            type_errors.append(f"{col} 없음")
            continue

        actual_dtype = df[col].dtype
        if expected_type == 'str' and not pd.api.types.is_string_dtype(df[col]):
            type_errors.append(f"{col}이 문자열이 아님 (dtype: {actual_dtype})")
        elif expected_type == 'int' and not pd.api.types.is_integer_dtype(df[col]):
            type_errors.append(f"{col}이 정수가 아님 (dtype: {actual_dtype})")
        elif expected_type == 'float' and not pd.api.types.is_float_dtype(df[col]):
            type_errors.append(f"{col}이 실수가 아님 (dtype: {actual_dtype})")

    if type_errors:
        print(f"  ❌ 타입 오류: {', '.join(type_errors)}")
    else:
        print(f"  ✅ 데이터 타입 확인 완료")

    # WinRate 및 PickBanRate 값 확인
    for col in ['WinRate', 'PickBanRate']:
        if col in df.columns:
            if df[col].max() > 1 or (df[col].min() < 0 and df[col].min() != -1):
                print(f"  ⚠️ {col} 범위 오류 (0~1 아님)")

def clean_gamer_csv(path):
    for team in os.listdir(path):
        team_path = os.path.join(path, team)
        if not os.path.isdir(team_path): continue

        for file in os.listdir(team_path):
            if not file.endswith('.csv'): continue
            file_path = os.path.join(team_path, file)

            try:
                df = pd.read_csv(file_path)
                df.columns = [col.strip() for col in df.columns]

                df = convert_percent_column(df, 'WinRate')
                df['Game'] = pd.to_numeric(df['Game'], errors='coerce').astype('Int64')

                df.to_csv(file_path, index=False)

                validate_and_print(
                    df,
                    file_path,
                    expected_header=['Champion', 'Game', 'WinRate'],
                    expected_types={'Champion': 'str', 'Game': 'int', 'WinRate': 'float'}
                )
            except Exception as e:
                print(f"  ❌ 오류 발생 [{file_path}]: {e}")

def clean_leauge_csv(path):
    for region in os.listdir(path):
        region_path = os.path.join(path, region)
        if not os.path.isdir(region_path): continue

        for file in os.listdir(region_path):
            if not file.endswith('.csv'): continue
            file_path = os.path.join(region_path, file)

            try:
                df = pd.read_csv(file_path, na_values=[], keep_default_na=False)
                df.columns = [col.strip() for col in df.columns]

                # PickBanRate 변환
                df = convert_percent_column_league(df, 'PickBanRate')

                # WinRate 변환 ("NA" → -1 처리 포함)
                df = convert_percent_column_league(df, 'WinRate', str_na_to_minus_one=True)

                df = df[['Champion', 'PickBanRate', 'WinRate']]
                df.to_csv(file_path, index=False)

                validate_and_print(
                    df,
                    file_path,
                    expected_header=['Champion', 'PickBanRate', 'WinRate'],
                    expected_types={'Champion': 'str', 'PickBanRate': 'float', 'WinRate': 'float'}
                )

            except Exception as e:
                print(f"  ❌ 오류 발생 [{file_path}]: {e}")

def clean_rank_csv(path):
    for region in os.listdir(path):
        region_path = os.path.join(path, region)
        if not os.path.isdir(region_path): continue

        for file in os.listdir(region_path):
            if not file.endswith('.csv'): continue
            file_path = os.path.join(region_path, file)

            try:
                df = pd.read_csv(file_path)
                df.columns = [col.strip() for col in df.columns]

                df = convert_percent_column(df, 'PickRate')
                df = convert_percent_column(df, 'BanRate')
                df = convert_percent_column(df, 'WinRate')

                df['PickBanRate'] = df['PickRate'] + df['BanRate']
                df = df[['Champion', 'PickBanRate', 'WinRate']]
                df.to_csv(file_path, index=False)

                validate_and_print(
                    df,
                    file_path,
                    expected_header=['Champion', 'PickBanRate', 'WinRate'],
                    expected_types={'Champion': 'str', 'PickBanRate': 'float', 'WinRate': 'float'}
                )
            except Exception as e:
                print(f"  ❌ 오류 발생 [{file_path}]: {e}")

# 실행
clean_gamer_csv('./data/Gamer')
clean_leauge_csv('./data/League')
clean_rank_csv('./data/Rank')
