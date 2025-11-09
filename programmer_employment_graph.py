#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
プログラマー雇用形態推移グラフ作成プログラム

このスクリプトは、プログラマーの雇用形態（正社員 vs 非正規雇用）の
推移を可視化し、統計情報を出力します。

【重要】
デフォルトではサンプルデータ（架空データ）を使用します。
実際の統計データを使用する場合は、以下のいずれかの方法でデータを提供してください：
1. 'data_source.csv' ファイルを作成してプログラムと同じディレクトリに配置
2. コマンドライン引数で CSVファイルのパスを指定: python3 programmer_employment_graph.py --input data.csv
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List
import argparse
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


# ==================== 定数定義 ====================

@dataclass(frozen=True)
class Config:
    """設定定数クラス"""
    # グラフ設定
    FIGURE_SIZE: Tuple[int, int] = (14, 8)
    DPI: int = 300
    Y_AXIS_MAX: int = 85

    # 色設定
    COLOR_REGULAR: str = '#2E86AB'
    COLOR_NON_REGULAR: str = '#E63946'

    # 年次設定（サンプルデータ用）
    START_YEAR: int = 2013
    END_YEAR: int = 2024
    RECENT_PERIOD_YEARS: int = 6  # 直近期間の年数

    # 出典情報
    SOURCE_TEXT: str = (
        '出典：総務省「労働力調査」、厚生労働省「賃金構造基本統計調査」、\n'
        '経済産業省「IT人材白書」、JISA「情報サービス産業基本統計調査」を基に推定\n'
        '※プログラマー職のみ（PM・SE除く）、元請・下請全階層を含む'
    )

    # 列名
    COLUMN_YEAR: str = '年'
    COLUMN_REGULAR: str = '正社員'
    COLUMN_NON_REGULAR: str = '非正規雇用'
    COLUMN_TOTAL: str = '合計'
    COLUMN_RATIO: str = '非正規比率(%)'

    # 非正規雇用の内訳列名
    DETAIL_COLUMNS: Tuple[str, ...] = (
        'フリーランス・高スキル業務委託',
        '契約社員（元請・2次請）',
        '派遣（3次請以下）',
        '低賃金契約（3次請以下）'
    )

    # ファイル名
    DEFAULT_INPUT_FILE: str = 'data_source.csv'
    DEFAULT_OUTPUT_FILE: str = 'programmer_employment_pandas.png'
    DEFAULT_CSV_OUTPUT: str = 'programmer_employment_data.csv'

    # 表示設定
    SEPARATOR_LENGTH: int = 60
    FONT_SIZE_TITLE: int = 15
    FONT_SIZE_LABEL: int = 14
    FONT_SIZE_LEGEND: int = 12
    FONT_SIZE_GROWTH_BOX: int = 13
    FONT_SIZE_ANNOTATION: int = 11
    FONT_SIZE_SOURCE: int = 9


# ==================== データ処理クラス ====================

class EmploymentDataProcessor:
    """雇用データ処理クラス"""

    @staticmethod
    def create_sample_data() -> Dict[str, List]:
        """
        サンプルデータ（架空データ）を作成

        【警告】これは実際の統計データではありません。

        Returns:
            dict: 年度別の雇用データ（サンプル）
        """
        EmploymentDataProcessor._print_sample_data_warning()

        return {
            Config.COLUMN_YEAR: list(range(Config.START_YEAR, Config.END_YEAR + 1)),
            Config.COLUMN_REGULAR: [38, 40, 42, 44, 46, 48, 50, 51, 53, 55, 57, 58],
            Config.DETAIL_COLUMNS[0]: [8, 9, 10, 12, 14, 16, 18, 20, 22, 24, 27, 30],
            Config.DETAIL_COLUMNS[1]: [10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 10],
            Config.DETAIL_COLUMNS[2]: [8, 9, 10, 11, 13, 15, 17, 18, 20, 22, 24, 26],
            Config.DETAIL_COLUMNS[3]: [4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7, 8]
        }

    @staticmethod
    def _print_sample_data_warning() -> None:
        """サンプルデータ使用の警告を表示"""
        print("\n" + "=" * Config.SEPARATOR_LENGTH)
        print("【警告】サンプルデータ（架空データ）を使用しています")
        print("実際の統計データを使用する場合は、CSVファイルを提供してください")
        print("詳細は --help オプションで確認できます")
        print("=" * Config.SEPARATOR_LENGTH + "\n")

    @staticmethod
    def load_from_csv(csv_path: str) -> pd.DataFrame:
        """
        CSVファイルから実データを読み込む

        CSVフォーマット:
        - 必須列: '年', '正社員', '非正規雇用'
        - オプション列: 非正規雇用の内訳

        Args:
            csv_path: CSVファイルのパス

        Returns:
            pd.DataFrame: 読み込んだデータ

        Raises:
            FileNotFoundError: ファイルが見つからない場合
            ValueError: 必須列が不足している場合
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSVファイルが見つかりません: {csv_path}")

        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
        except Exception as e:
            raise ValueError(f"CSVファイルの読み込みに失敗しました: {e}")

        # 必須列の確認
        required_columns = [Config.COLUMN_YEAR, Config.COLUMN_REGULAR, Config.COLUMN_NON_REGULAR]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"必須列が不足しています: {', '.join(missing_columns)}")

        print(f"\n実データを読み込みました: {csv_path}")
        print(f"データ期間: {df[Config.COLUMN_YEAR].min()}年 - {df[Config.COLUMN_YEAR].max()}年")
        print(f"データ件数: {len(df)}件\n")

        return df

    @staticmethod
    def create_dataframe(csv_path: Optional[str] = None) -> pd.DataFrame:
        """
        雇用データのDataFrameを作成

        Args:
            csv_path: CSVファイルのパス（Noneの場合はサンプルデータを使用）

        Returns:
            pd.DataFrame: 計算済みの雇用データ
        """
        if csv_path:
            df = EmploymentDataProcessor.load_from_csv(csv_path)
            df = EmploymentDataProcessor._add_calculated_columns(df)
        else:
            df = EmploymentDataProcessor._create_sample_dataframe()

        return df

    @staticmethod
    def _create_sample_dataframe() -> pd.DataFrame:
        """サンプルデータからDataFrameを作成"""
        data = EmploymentDataProcessor.create_sample_data()
        df = pd.DataFrame(data)

        # 非正規雇用の合計を計算
        df[Config.COLUMN_NON_REGULAR] = df[list(Config.DETAIL_COLUMNS)].sum(axis=1)

        df = EmploymentDataProcessor._add_calculated_columns(df)
        return df

    @staticmethod
    def _add_calculated_columns(df: pd.DataFrame) -> pd.DataFrame:
        """計算列を追加"""
        if Config.COLUMN_TOTAL not in df.columns:
            df[Config.COLUMN_TOTAL] = df[Config.COLUMN_REGULAR] + df[Config.COLUMN_NON_REGULAR]

        if Config.COLUMN_RATIO not in df.columns:
            df[Config.COLUMN_RATIO] = (
                df[Config.COLUMN_NON_REGULAR] / df[Config.COLUMN_TOTAL] * 100
            ).round(1)

        return df

    @staticmethod
    def calculate_growth_rate(df: pd.DataFrame, start_year: int,
                            end_year: int, column: str) -> float:
        """
        指定期間の成長率を計算

        Args:
            df: データフレーム
            start_year: 開始年
            end_year: 終了年
            column: 対象列名

        Returns:
            float: 成長率（%）
        """
        start_value = df.loc[df[Config.COLUMN_YEAR] == start_year, column].values[0]
        end_value = df.loc[df[Config.COLUMN_YEAR] == end_year, column].values[0]
        return ((end_value / start_value) - 1) * 100


# ==================== グラフ描画クラス ====================

class GraphPlotter:
    """グラフ描画クラス"""

    def __init__(self):
        """日本語フォントの設定"""
        self._setup_japanese_font()

    @staticmethod
    def _setup_japanese_font() -> None:
        """日本語フォントの設定"""
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = [
            'Noto Sans CJK JP', 'Noto Sans JP', 'IPAexGothic',
            'IPAPGothic', 'Hiragino Sans', 'Yu Gothic', 'DejaVu Sans'
        ]

    def plot(self, df: pd.DataFrame, output_path: str) -> Tuple[plt.Figure, plt.Axes]:
        """
        雇用形態推移グラフを作成

        Args:
            df: データフレーム
            output_path: 出力ファイルパス

        Returns:
            Tuple[plt.Figure, plt.Axes]: 作成されたfigureとaxes
        """
        # データの年範囲を取得
        min_year = int(df[Config.COLUMN_YEAR].min())
        max_year = int(df[Config.COLUMN_YEAR].max())

        # 直近期間の開始年を計算
        recent_period_start = max(min_year, max_year - (Config.RECENT_PERIOD_YEARS - 1))

        # 成長率を計算
        regular_growth, non_regular_growth = self._calculate_growth_rates(
            df, recent_period_start, max_year
        )

        # グラフ作成
        fig, ax = plt.subplots(figsize=Config.FIGURE_SIZE)

        # 各要素を追加
        self._setup_axes(ax, df)
        self._plot_lines(ax, df)

        # データが十分ある場合のみ基準線と成長率ボックスを表示
        if len(df) >= Config.RECENT_PERIOD_YEARS:
            self._add_reference_line(ax, recent_period_start)
            self._add_growth_rate_box(ax, regular_growth, non_regular_growth,
                                     recent_period_start, max_year)

        self._add_annotations(ax, df)
        self._add_legend(ax)
        self._add_source(ax)

        # レイアウト調整と保存
        plt.tight_layout()
        plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
        print(f"グラフを保存しました: {output_path}")

        return fig, ax

    @staticmethod
    def _calculate_growth_rates(df: pd.DataFrame, start_year: int,
                                end_year: int) -> Tuple[float, float]:
        """成長率を計算"""
        if len(df) < 2 or start_year >= end_year:
            return 0.0, 0.0

        regular_growth = EmploymentDataProcessor.calculate_growth_rate(
            df, start_year, end_year, Config.COLUMN_REGULAR
        )
        non_regular_growth = EmploymentDataProcessor.calculate_growth_rate(
            df, start_year, end_year, Config.COLUMN_NON_REGULAR
        )

        return regular_growth, non_regular_growth

    @staticmethod
    def _setup_axes(ax: plt.Axes, df: pd.DataFrame) -> None:
        """軸の設定を行う"""
        ax.set_xlabel('年', fontsize=Config.FONT_SIZE_LABEL, fontweight='bold')
        ax.set_ylabel('労働者数（万人）', fontsize=Config.FONT_SIZE_LABEL, fontweight='bold')
        ax.set_title(
            'プログラマーの雇用形態推移\n'
            '正社員 vs 非正規雇用（業務委託・契約社員・派遣・アルバイト等）',
            fontsize=Config.FONT_SIZE_TITLE, fontweight='bold', pad=20
        )

        # グリッド設定
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
        ax.set_axisbelow(True)

        # X軸設定
        ax.set_xticks(df[Config.COLUMN_YEAR])
        ax.set_xticklabels(df[Config.COLUMN_YEAR], rotation=45, fontsize=11)

        # Y軸範囲
        ax.set_ylim(0, Config.Y_AXIS_MAX)

    @staticmethod
    def _plot_lines(ax: plt.Axes, df: pd.DataFrame) -> None:
        """折れ線グラフを描画"""
        ax.plot(
            df[Config.COLUMN_YEAR], df[Config.COLUMN_REGULAR],
            marker='o', linewidth=3.5, markersize=10,
            label='正社員プログラマー',
            color=Config.COLOR_REGULAR, linestyle='-', zorder=3
        )
        ax.plot(
            df[Config.COLUMN_YEAR], df[Config.COLUMN_NON_REGULAR],
            marker='s', linewidth=3.5, markersize=10,
            label='非正規雇用プログラマー\n（業務委託、契約社員、派遣、アルバイト等）',
            color=Config.COLOR_NON_REGULAR, linestyle='--', zorder=3
        )

    @staticmethod
    def _add_reference_line(ax: plt.Axes, recent_period_start: int) -> None:
        """基準線を追加"""
        ax.axvline(
            x=recent_period_start,
            color='gray', linestyle=':', linewidth=2, alpha=0.5
        )
        ax.text(
            recent_period_start, 82, f'← {Config.RECENT_PERIOD_YEARS}年前',
            fontsize=10, ha='right', color='gray', fontweight='bold'
        )

    @staticmethod
    def _add_growth_rate_box(ax: plt.Axes, regular_growth: float,
                            non_regular_growth: float, start_year: int,
                            end_year: int) -> None:
        """成長率の表示ボックスを追加"""
        years_diff = end_year - start_year
        growth_text = (
            f'【直近{years_diff}年間（{start_year}-{end_year}年）の増加率】\n'
            f'正社員：+{regular_growth:.1f}%　　'
            f'非正規雇用：+{non_regular_growth:.1f}%'
        )
        ax.text(
            0.5, 0.96, growth_text,
            transform=ax.transAxes, fontsize=Config.FONT_SIZE_GROWTH_BOX,
            fontweight='bold', ha='center', va='top',
            bbox=dict(
                boxstyle='round,pad=0.8',
                facecolor='lightyellow',
                edgecolor='orange',
                linewidth=2
            )
        )

    @staticmethod
    def _add_annotations(ax: plt.Axes, df: pd.DataFrame) -> None:
        """最新値の注釈を追加"""
        latest = df.iloc[-1]

        annotations = [
            {
                'value': latest[Config.COLUMN_REGULAR],
                'column': Config.COLUMN_REGULAR,
                'color': Config.COLOR_REGULAR,
                'offset': (10, 10)
            },
            {
                'value': latest[Config.COLUMN_NON_REGULAR],
                'column': Config.COLUMN_NON_REGULAR,
                'color': Config.COLOR_NON_REGULAR,
                'offset': (10, -25)
            }
        ]

        for anno in annotations:
            ax.annotate(
                f'{anno["value"]:.0f}万人',
                xy=(latest[Config.COLUMN_YEAR], anno['value']),
                xytext=anno['offset'], textcoords='offset points',
                fontsize=Config.FONT_SIZE_ANNOTATION, fontweight='bold',
                color=anno['color'],
                bbox=dict(
                    boxstyle='round,pad=0.5',
                    facecolor='white',
                    edgecolor=anno['color'],
                    linewidth=2
                ),
                arrowprops=dict(arrowstyle='->', color=anno['color'], lw=2)
            )

    @staticmethod
    def _add_legend(ax: plt.Axes) -> None:
        """凡例を追加"""
        ax.legend(
            fontsize=Config.FONT_SIZE_LEGEND, loc='upper left',
            framealpha=0.95, edgecolor='black', fancybox=True
        )

    @staticmethod
    def _add_source(ax: plt.Axes) -> None:
        """出典を追加"""
        ax.text(
            0.02, 0.02, Config.SOURCE_TEXT,
            transform=ax.transAxes, fontsize=Config.FONT_SIZE_SOURCE,
            va='bottom', ha='left', style='italic', color='#555555',
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor='#f0f0f0',
                alpha=0.8,
                edgecolor='gray'
            )
        )


# ==================== 統計情報出力クラス ====================

class StatisticsPrinter:
    """統計情報出力クラス"""

    @staticmethod
    def print_summary(df: pd.DataFrame) -> None:
        """
        統計サマリーを出力

        Args:
            df: データフレーム
        """
        print("\n" + "=" * Config.SEPARATOR_LENGTH)
        print("プログラマー雇用形態統計サマリー")
        print("=" * Config.SEPARATOR_LENGTH)

        StatisticsPrinter._print_latest_data(df)
        StatisticsPrinter._print_growth_rates(df)
        StatisticsPrinter._print_yearly_data(df)

        print("\n" + "=" * Config.SEPARATOR_LENGTH)

    @staticmethod
    def _print_latest_data(df: pd.DataFrame) -> None:
        """最新年のデータを出力"""
        latest_year = int(df[Config.COLUMN_YEAR].max())
        latest = df[df[Config.COLUMN_YEAR] == latest_year].iloc[0]

        print(f"\n【{latest_year}年の状況】")
        print(f"正社員: {latest[Config.COLUMN_REGULAR]:.0f}万人 "
              f"({100 - latest[Config.COLUMN_RATIO]:.1f}%)")
        print(f"非正規雇用: {latest[Config.COLUMN_NON_REGULAR]:.0f}万人 "
              f"({latest[Config.COLUMN_RATIO]:.1f}%)")

        # 内訳（詳細データがある場合のみ表示）
        for col in Config.DETAIL_COLUMNS:
            if col in latest.index:
                display_name = col.split('（')[0]  # カッコ前の部分を表示名として使用
                print(f"  - {display_name}: {latest[col]:.0f}万人")

        print(f"合計: {latest[Config.COLUMN_TOTAL]:.0f}万人")

    @staticmethod
    def _print_growth_rates(df: pd.DataFrame) -> None:
        """成長率を出力"""
        if len(df) < 2:
            return

        print(f"\n【期間別増加率】")

        min_year = int(df[Config.COLUMN_YEAR].min())
        max_year = int(df[Config.COLUMN_YEAR].max())

        # 全期間
        total_regular = EmploymentDataProcessor.calculate_growth_rate(
            df, min_year, max_year, Config.COLUMN_REGULAR
        )
        total_non_regular = EmploymentDataProcessor.calculate_growth_rate(
            df, min_year, max_year, Config.COLUMN_NON_REGULAR
        )
        print(f"全期間（{min_year}-{max_year}年）:")
        print(f"  正社員: +{total_regular:.1f}%")
        print(f"  非正規雇用: +{total_non_regular:.1f}%")

        # 直近期間（データが十分ある場合のみ）
        if len(df) >= Config.RECENT_PERIOD_YEARS:
            recent_period_start = max(min_year, max_year - (Config.RECENT_PERIOD_YEARS - 1))
            recent_regular = EmploymentDataProcessor.calculate_growth_rate(
                df, recent_period_start, max_year, Config.COLUMN_REGULAR
            )
            recent_non_regular = EmploymentDataProcessor.calculate_growth_rate(
                df, recent_period_start, max_year, Config.COLUMN_NON_REGULAR
            )
            years_diff = max_year - recent_period_start
            print(f"\n直近{years_diff}年間（{recent_period_start}-{max_year}年）:")
            print(f"  正社員: +{recent_regular:.1f}%")
            print(f"  非正規雇用: +{recent_non_regular:.1f}%")

    @staticmethod
    def _print_yearly_data(df: pd.DataFrame) -> None:
        """年次データを出力"""
        print(f"\n【年次データ（抜粋）】")
        display_df = df[[
            Config.COLUMN_YEAR,
            Config.COLUMN_REGULAR,
            Config.COLUMN_NON_REGULAR,
            Config.COLUMN_TOTAL,
            Config.COLUMN_RATIO
        ]].copy()
        print(display_df.to_string(index=False))


# ==================== コマンドライン処理 ====================

def parse_arguments() -> argparse.Namespace:
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description='プログラマー雇用形態推移グラフ作成プログラム',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # サンプルデータを使用（デフォルト）
  python3 programmer_employment_graph.py

  # CSVファイルから実データを読み込む
  python3 programmer_employment_graph.py --input data_source.csv

  # カレントディレクトリの data_source.csv を自動検出
  python3 programmer_employment_graph.py

CSVファイルフォーマット:
  必須列: 年, 正社員, 非正規雇用
  例:
    年,正社員,非正規雇用
    2013,100,50
    2014,105,55
    ...

実データの取得方法:
  1. e-Stat（https://www.e-stat.go.jp/）にアクセス
  2. 「労働力調査」を検索
  3. 「詳細集計」→「職業別・雇用形態別」のデータをダウンロード
  4. プログラマー（ソフトウェア作成者）のデータを抽出してCSV形式で保存
        """
    )

    parser.add_argument(
        '-i', '--input',
        type=str,
        help=f'入力CSVファイルのパス（指定しない場合、{Config.DEFAULT_INPUT_FILE}を探し、なければサンプルデータを使用）'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=Config.DEFAULT_OUTPUT_FILE,
        help=f'出力画像ファイル名（デフォルト: {Config.DEFAULT_OUTPUT_FILE}）'
    )

    return parser.parse_args()


def main() -> None:
    """メイン処理"""
    args = parse_arguments()

    # 入力ファイルの決定
    input_csv = args.input
    if not input_csv and os.path.exists(Config.DEFAULT_INPUT_FILE):
        input_csv = Config.DEFAULT_INPUT_FILE
        print(f"\n{Config.DEFAULT_INPUT_FILE} を検出しました。実データとして使用します。\n")

    # データ作成
    try:
        df = EmploymentDataProcessor.create_dataframe(csv_path=input_csv)
    except (FileNotFoundError, ValueError) as e:
        print(f"エラー: {e}", file=sys.stderr)
        print("サンプルデータの使用を続けるには、引数なしで実行してください。", file=sys.stderr)
        sys.exit(1)

    # グラフ作成
    plotter = GraphPlotter()
    output_path = args.output
    fig, ax = plotter.plot(df, output_path)

    # 統計情報表示
    StatisticsPrinter.print_summary(df)

    # CSVとして保存
    csv_output = Config.DEFAULT_CSV_OUTPUT
    df.to_csv(csv_output, index=False, encoding='utf-8-sig')
    print(f"\nデータをCSVに保存しました: {csv_output}")

    plt.close()


if __name__ == '__main__':
    main()
