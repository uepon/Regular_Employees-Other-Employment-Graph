# プログラマー雇用形態推移グラフ作成プログラム

プログラマーの雇用形態（正社員 vs 非正規雇用）の推移を可視化し、統計情報を出力します。

## 実行方法

### 基本的な使い方

```bash
python3 programmer_employment_graph.py
```

デフォルトでは、プログラム内部のサンプルデータを使用します。

### CSVファイルを指定して実行

```bash
python3 programmer_employment_graph.py --input data.csv
```

### 出力ファイル名も指定

```bash
python3 programmer_employment_graph.py --input data.csv --output result.png
```

### ヘルプの表示

```bash
python3 programmer_employment_graph.py --help
```

## CSVファイルのフォーマット

以下の3列が必須です：

```csv
年,正社員,非正規雇用
2013,38,30
2014,40,32
2015,42,35
```

- `年`: 西暦年
- `正社員`: 正社員の人数（単位: 万人）
- `非正規雇用`: 非正規雇用の人数（単位: 万人）

オプションで非正規雇用の内訳を追加できます：
- `フリーランス・高スキル業務委託`
- `契約社員（元請・2次請）`
- `派遣（3次請以下）`
- `低賃金契約（3次請以下）`

## 出力ファイル

実行すると以下のファイルが生成されます：

1. `programmer_employment_pandas.png` - グラフ画像（300 DPI）
2. `programmer_employment_data.csv` - 統計データ（合計・比率などの計算列を含む）

## 必要なパッケージ

```bash
pip install pandas matplotlib
```

## データソース

実際のプログラマー雇用データは以下から取得できます：

- **e-Stat（政府統計の総合窓口）**: https://www.e-stat.go.jp/
  - 「労働力調査」→「詳細集計」→「職業別・雇用形態別」
- **総務省統計局**: https://www.stat.go.jp/data/roudou/
