import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_ROOT / "data" / "train.csv"
SENTENCE_PATH = PROJECT_ROOT / "data" / "Sentences_Oare_FirstWord_LinNum.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "train_truncated.csv"

def convert_fraction_to_decimal(match: re.Match) -> str:
    """
    マッチした '分子 / 分母' を小数に変換する。
    小数第4位までで四捨五入する。
    """
    numerator = int(match.group(1))
    denominator = int(match.group(2))
    
    if denominator == 0:
        return match.group(0)  # ゼロ除算の場合は置換せずそのまま返す
        
    # 計算と四捨五入（小数第4位まで）
    decimal_value = round(numerator / denominator, 4)
    
    # 文字列に変換（gフラグで余計な0を消す。ただし非常に小さい値は指数表記になる可能性あり）
    # 指数表記を避けたい場合は f"{decimal_value:.4f}".rstrip('0').rstrip('.') を使用
    return f"{decimal_value:g}"

def replace_fractions_in_text(text: str) -> str:
    """
    テキスト内の '数字 / 数字' パターンを小数表現に置換する関数。
    """
    # \d+ : 1つ以上の数字
    # \s* : 0個以上の空白
    pattern = r'(\d+)[\s]?/[\s]?(\d+)'
    
    return re.sub(pattern, convert_fraction_to_decimal, text)

def extract_patterns_from_text(text: str) -> list:
    """
    テキストから「数字 / 数字」のパターンをすべて抽出する関数。
    """
    if not isinstance(text, str):
        return []
    
    # パターン: 数字（1文字以上）[ ]/[ ]数字（1文字以上）
    # 空白が含まれていても対応できるよう \s* を入れています
    pattern = r'\d+[\s]?/[\s]?\d+'
    
    # 抽出結果のリストを返す（例: ['3 / 4', '11/12']）
    # strip() で前後の余計な空白を整える
    matches = re.findall(pattern, text)
    return [m.strip() for m in matches if m.strip() != "/"]

def get_all_matches_in_df(df: pd.DataFrame) -> list:
    """
    DataFrame内のすべての文字列カラムから、該当パターンを収集する関数。
    """
    all_found_patterns = []
    
    # 文字列型が含まれる可能性のあるカラムをループ
    for col in df.columns:
        # 各セルの値を文字列として処理し、パターンを抽出
        column_matches = df[col].astype(str).apply(extract_patterns_from_text)
        
        # 二次元リストを一次元に平坦化して追加
        for matches in column_matches:
            all_found_patterns.extend(matches)
            
    # 重複を除去してソート（必要に応じて）
    return sorted(list(set(all_found_patterns)))

def calculate_addition(match: re.Match) -> str:
    """
    マッチした '数値 + 数値' を計算して結果を返す。
    小数第4位までで四捨五入する。
    """
    # floatに変換して加算
    num1 = float(match.group(1))
    num2 = float(match.group(2))
    
    result_value = round(num1 + num2, 4)
    
    # gフォーマットで文字列化（不要な.0を消す）
    return f"{result_value:g}"

def replace_addition_with_sum(text: str) -> str:
    """
    文字列内の '数値 + 数値' のパターンを探して、加算結果に置換する。
    """
    # 数値（整数または小数）のパターン
    # \d+      : 1つ以上の数字
    # (?:\.\d+)? : 非キャプチャグループで、小数点と1つ以上の数字が続く（あってもなくても良い）
    num_pattern = r'(\d+(?:\.\d+)?)'
    
    # 数値 + 空白(任意) + 加算記号 + 空白(任意) + 数値
    pattern = rf'{num_pattern}\s*\+\s*{num_pattern}'
    
    return re.sub(pattern, calculate_addition, text)

def main():
    # 1. データの読み込み
    df = pd.read_csv(TRAIN_PATH)
    
    if df.empty:
        return

    # 2. パターンの抽出
    found_items = get_all_matches_in_df(df)

    # 3. 結果の出力
    print(f"見つかったパターン数: {len(found_items)}")
    for item in found_items:
        print(item)

if __name__ == "__main__":
    main()
    # --- テスト ---
    test_texts = [
        "3 / 4",      # -> 0.75
        "1 / 3",      # -> 0.3333
        "11 / 12",    # -> 0.9167
        "5 / 2",      # -> 2.5
        "割合は 7 / 8 です" # -> 割合は 0.875 です
    ]

    for t in test_texts:
        print(f"{t} -> {replace_fractions_in_text(t)}")
        
    # --- テスト ---
    test_cases = [
        "15.75 + 4.25",   # -> 20
        "10 +5",         # -> 15
        "3.14+2",       # -> 5.14
        "計算結果：1.234 + 5.6789" # -> 計算結果：6.9129
    ]

    for t in test_cases:
        print(f"{t}  =>  {replace_addition_with_sum(t)}")
