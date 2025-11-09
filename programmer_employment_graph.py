#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
プログラマー雇用形態推移グラフ作成プログラム

このスクリプトは、プログラマーの雇用形態（正社員 vs 非正規雇用）の
推移を可視化し、統計情報を出力します。

【重要】
プログラム内部にサンプルデータ（架空データ）が含まれており、
CSVファイルを指定しない場合はこのサンプルデータが使用されます。

実際の統計データを使用する場合は、以下のいずれかの方法でデータを提供してください：
1. 'data_source.csv' ファイルを作成してプログラムと同じディレクトリに配置
2. コマンドライン引数で CSVファイルのパスを指定: python3 programmer_employment_graph.py --input data.csv

サンプルデータ使用時は実行時に警告が表示されます。
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List
import argparse
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


# ==================== 設定定数 ====================

@dataclass(frozen=True)
class Config:
    """
    グラフ作成に必要な全設定を集約

    このクラスは変更不可（frozen=True）で、すべての設定値を一元管理します。
    """

    # グラフサイズと解像度
    FIGURE_SIZE: Tuple[int, int] = (14, 8)  # インチ単位 (幅, 高さ)
    DPI: int = 300  # 画像解像度（dots per inch）
    Y_AXIS_MAXIMUM_VALUE: int = 85  # Y軸の最大値（万人）

    # グラフの配色
    COLOR_REGULAR_EMPLOYEE: str = '#2E86AB'  # 正社員の線の色（青系）
    COLOR_NON_REGULAR_EMPLOYEE: str = '#E63946'  # 非正規雇用の線の色（赤系）

    # サンプルデータの設定（実データ使用時は無視される）
    SAMPLE_DATA_START_YEAR: int = 2013
    SAMPLE_DATA_END_YEAR: int = 2024
    RECENT_PERIOD_YEARS: int = 6  # 直近期間として扱う年数

    # グラフに表示するデータ出典情報
    DATA_SOURCE_TEXT: str = (
        '出典：総務省「労働力調査」、厚生労働省「賃金構造基本統計調査」、\n'
        '経済産業省「IT人材白書」、JISA「情報サービス産業基本統計調査」を基に推定\n'
        '※プログラマー職のみ（PM・SE除く）、元請・下請全階層を含む'
    )

    # CSV列名（実データCSVもこの列名に従う必要があります）
    COLUMN_YEAR: str = '年'
    COLUMN_REGULAR_EMPLOYEE: str = '正社員'
    COLUMN_NON_REGULAR_EMPLOYEE: str = '非正規雇用'
    COLUMN_TOTAL_EMPLOYEE: str = '合計'  # プログラムが自動計算
    COLUMN_NON_REGULAR_RATIO: str = '非正規比率(%)'  # プログラムが自動計算

    # 非正規雇用の内訳列名（サンプルデータでのみ使用、実データでは任意）
    NON_REGULAR_BREAKDOWN_COLUMNS: Tuple[str, ...] = (
        'フリーランス・高スキル業務委託',
        '契約社員（元請・2次請）',
        '派遣（3次請以下）',
        '低賃金契約（3次請以下）'
    )

    # ファイル名の設定
    DEFAULT_INPUT_FILENAME: str = 'data_source.csv'  # 自動検出する実データファイル名
    DEFAULT_OUTPUT_GRAPH_FILENAME: str = 'programmer_employment_pandas.png'  # 出力グラフファイル名
    DEFAULT_OUTPUT_CSV_FILENAME: str = 'programmer_employment_data.csv'  # 出力CSVファイル名

    # 表示フォーマットの設定
    SEPARATOR_LINE_LENGTH: int = 60  # コンソール出力の区切り線の長さ
    FONT_SIZE_TITLE: int = 15
    FONT_SIZE_AXIS_LABEL: int = 14
    FONT_SIZE_LEGEND: int = 12
    FONT_SIZE_GROWTH_RATE_BOX: int = 13
    FONT_SIZE_VALUE_ANNOTATION: int = 11
    FONT_SIZE_DATA_SOURCE: int = 9


# ==================== データ処理 ====================

class EmploymentDataProcessor:
    """
    雇用データの読み込み、生成、計算を担当するクラス

    実データCSVの読み込み、サンプルデータの生成、
    および計算列の追加などを行います。
    """

    @staticmethod
    def load_or_create_dataframe(csv_file_path: Optional[str] = None) -> pd.DataFrame:
        """
        CSVファイルから実データを読み込むか、サンプルデータを生成

        Args:
            csv_file_path: CSVファイルパス。Noneの場合はサンプルデータを生成

        Returns:
            pd.DataFrame: 雇用データ（計算列を含む）
        """
        if csv_file_path:
            return EmploymentDataProcessor._load_real_data_from_csv(csv_file_path)
        return EmploymentDataProcessor._generate_sample_dataframe()

    @staticmethod
    def _load_real_data_from_csv(csv_file_path: str) -> pd.DataFrame:
        """
        CSVファイルから実データを読み込み、必要な計算列を追加

        Args:
            csv_file_path: CSVファイルのパス

        Returns:
            pd.DataFrame: 実データ（計算列を含む）

        Raises:
            FileNotFoundError: ファイルが存在しない場合
            ValueError: ファイル読み込み失敗または必須列が不足している場合
        """
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"CSVファイルが見つかりません: {csv_file_path}")

        try:
            dataframe = pd.read_csv(csv_file_path, encoding='utf-8-sig')
        except Exception as error:
            raise ValueError(f"CSVファイルの読み込みに失敗しました: {error}")

        # 必須列の存在確認
        required_column_names = [
            Config.COLUMN_YEAR,
            Config.COLUMN_REGULAR_EMPLOYEE,
            Config.COLUMN_NON_REGULAR_EMPLOYEE
        ]
        missing_column_names = [
            col for col in required_column_names if col not in dataframe.columns
        ]

        if missing_column_names:
            raise ValueError(f"必須列が不足しています: {', '.join(missing_column_names)}")

        print(f"\n実データを読み込みました: {csv_file_path}")
        print(f"データ期間: {int(dataframe[Config.COLUMN_YEAR].min())}年 - "
              f"{int(dataframe[Config.COLUMN_YEAR].max())}年")
        print(f"データ件数: {len(dataframe)}件\n")

        return EmploymentDataProcessor._add_calculated_columns(dataframe)

    @staticmethod
    def _generate_sample_dataframe() -> pd.DataFrame:
        """
        サンプルデータ（架空データ）を生成してDataFrameとして返す

        注意: これは実際の統計データではありません。
        デモンストレーション用の架空データです。

        Returns:
            pd.DataFrame: サンプルデータ（計算列を含む）
        """
        EmploymentDataProcessor._display_sample_data_warning()

        # サンプルデータの定義（架空の数値）
        sample_data_dict = {
            Config.COLUMN_YEAR: list(range(
                Config.SAMPLE_DATA_START_YEAR,
                Config.SAMPLE_DATA_END_YEAR + 1
            )),
            Config.COLUMN_REGULAR_EMPLOYEE: [38, 40, 42, 44, 46, 48, 50, 51, 53, 55, 57, 58],
            Config.NON_REGULAR_BREAKDOWN_COLUMNS[0]: [8, 9, 10, 12, 14, 16, 18, 20, 22, 24, 27, 30],
            Config.NON_REGULAR_BREAKDOWN_COLUMNS[1]: [10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 10],
            Config.NON_REGULAR_BREAKDOWN_COLUMNS[2]: [8, 9, 10, 11, 13, 15, 17, 18, 20, 22, 24, 26],
            Config.NON_REGULAR_BREAKDOWN_COLUMNS[3]: [4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7, 8]
        }

        dataframe = pd.DataFrame(sample_data_dict)

        # 非正規雇用の合計を計算
        dataframe[Config.COLUMN_NON_REGULAR_EMPLOYEE] = dataframe[
            list(Config.NON_REGULAR_BREAKDOWN_COLUMNS)
        ].sum(axis=1)

        return EmploymentDataProcessor._add_calculated_columns(dataframe)

    @staticmethod
    def _display_sample_data_warning() -> None:
        """サンプルデータ使用時の警告メッセージをコンソールに表示"""
        print("\n" + "=" * Config.SEPARATOR_LINE_LENGTH)
        print("【警告】サンプルデータ（架空データ）を使用しています")
        print("実際の統計データを使用する場合は、CSVファイルを提供してください")
        print("詳細は --help オプションで確認できます")
        print("=" * Config.SEPARATOR_LINE_LENGTH + "\n")

    @staticmethod
    def _add_calculated_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        合計と比率の計算列を追加（まだ存在しない場合のみ）

        Args:
            dataframe: 元のデータフレーム

        Returns:
            pd.DataFrame: 計算列が追加されたデータフレーム
        """
        if Config.COLUMN_TOTAL_EMPLOYEE not in dataframe.columns:
            dataframe[Config.COLUMN_TOTAL_EMPLOYEE] = (
                dataframe[Config.COLUMN_REGULAR_EMPLOYEE] +
                dataframe[Config.COLUMN_NON_REGULAR_EMPLOYEE]
            )

        if Config.COLUMN_NON_REGULAR_RATIO not in dataframe.columns:
            dataframe[Config.COLUMN_NON_REGULAR_RATIO] = (
                dataframe[Config.COLUMN_NON_REGULAR_EMPLOYEE] /
                dataframe[Config.COLUMN_TOTAL_EMPLOYEE] * 100
            ).round(1)

        return dataframe

    @staticmethod
    def calculate_employment_growth_rate(
        dataframe: pd.DataFrame,
        start_year: int,
        end_year: int,
        column_name: str
    ) -> float:
        """
        指定期間・指定列の成長率をパーセンテージで計算

        Args:
            dataframe: 雇用データ
            start_year: 開始年
            end_year: 終了年
            column_name: 計算対象の列名

        Returns:
            float: 成長率（%）
        """
        start_year_value = dataframe.loc[
            dataframe[Config.COLUMN_YEAR] == start_year, column_name
        ].values[0]
        end_year_value = dataframe.loc[
            dataframe[Config.COLUMN_YEAR] == end_year, column_name
        ].values[0]
        return ((end_year_value / start_year_value) - 1) * 100


# ==================== グラフ描画 ====================

class EmploymentGraphPlotter:
    """
    雇用形態推移グラフの描画を担当するクラス

    matplotlibを使用してグラフを作成し、PNGファイルとして保存します。
    """

    def __init__(self):
        """日本語フォントを設定してグラフプロッターを初期化"""
        self._configure_japanese_font()

    @staticmethod
    def _configure_japanese_font() -> None:
        """matplotlibで日本語を正しく表示できるようフォントを設定"""
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = [
            'Noto Sans CJK JP', 'Noto Sans JP', 'IPAexGothic',
            'IPAPGothic', 'Hiragino Sans', 'Yu Gothic', 'DejaVu Sans'
        ]

    def create_and_save_graph(
        self,
        dataframe: pd.DataFrame,
        output_file_path: str
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        雇用形態推移グラフを作成してファイルに保存

        Args:
            dataframe: 雇用データ
            output_file_path: 保存先ファイルパス

        Returns:
            Tuple[plt.Figure, plt.Axes]: 作成されたfigureとaxes
        """
        figure, axes = plt.subplots(figsize=Config.FIGURE_SIZE)

        # データの年範囲を取得
        data_min_year = int(dataframe[Config.COLUMN_YEAR].min())
        data_max_year = int(dataframe[Config.COLUMN_YEAR].max())
        recent_period_start_year = max(
            data_min_year,
            data_max_year - (Config.RECENT_PERIOD_YEARS - 1)
        )

        # グラフの各要素を構築
        self._setup_graph_axes(axes, dataframe)
        self._draw_employment_trend_lines(axes, dataframe)

        # データが十分にある場合のみ成長率情報を表示
        if len(dataframe) >= Config.RECENT_PERIOD_YEARS:
            regular_growth, non_regular_growth = self._compute_recent_growth_rates(
                dataframe, recent_period_start_year, data_max_year
            )
            self._add_recent_period_reference_line(axes, recent_period_start_year)
            self._add_growth_rate_display_box(
                axes, regular_growth, non_regular_growth,
                recent_period_start_year, data_max_year
            )

        self._add_latest_value_annotations(axes, dataframe)
        self._add_graph_legend(axes)
        self._add_data_source_note(axes)

        # レイアウト調整と保存
        plt.tight_layout()
        plt.savefig(output_file_path, dpi=Config.DPI, bbox_inches='tight')
        print(f"グラフを保存しました: {output_file_path}")

        return figure, axes

    @staticmethod
    def _setup_graph_axes(axes: plt.Axes, dataframe: pd.DataFrame) -> None:
        """グラフの軸ラベル、タイトル、グリッド、目盛りを設定"""
        axes.set_xlabel('年', fontsize=Config.FONT_SIZE_AXIS_LABEL, fontweight='bold')
        axes.set_ylabel('労働者数（万人）', fontsize=Config.FONT_SIZE_AXIS_LABEL, fontweight='bold')
        axes.set_title(
            'プログラマーの雇用形態推移\n'
            '正社員 vs 非正規雇用（業務委託・契約社員・派遣・アルバイト等）',
            fontsize=Config.FONT_SIZE_TITLE, fontweight='bold', pad=20
        )
        axes.grid(True, alpha=0.3, linestyle=':', linewidth=1)
        axes.set_axisbelow(True)
        axes.set_xticks(dataframe[Config.COLUMN_YEAR])
        axes.set_xticklabels(dataframe[Config.COLUMN_YEAR], rotation=45, fontsize=11)
        axes.set_ylim(0, Config.Y_AXIS_MAXIMUM_VALUE)

    @staticmethod
    def _draw_employment_trend_lines(axes: plt.Axes, dataframe: pd.DataFrame) -> None:
        """正社員と非正規雇用の推移を折れ線グラフで描画"""
        axes.plot(
            dataframe[Config.COLUMN_YEAR],
            dataframe[Config.COLUMN_REGULAR_EMPLOYEE],
            marker='o', linewidth=3.5, markersize=10,
            label='正社員プログラマー',
            color=Config.COLOR_REGULAR_EMPLOYEE, linestyle='-', zorder=3
        )
        axes.plot(
            dataframe[Config.COLUMN_YEAR],
            dataframe[Config.COLUMN_NON_REGULAR_EMPLOYEE],
            marker='s', linewidth=3.5, markersize=10,
            label='非正規雇用プログラマー\n（業務委託、契約社員、派遣、アルバイト等）',
            color=Config.COLOR_NON_REGULAR_EMPLOYEE, linestyle='--', zorder=3
        )

    @staticmethod
    def _compute_recent_growth_rates(
        dataframe: pd.DataFrame,
        start_year: int,
        end_year: int
    ) -> Tuple[float, float]:
        """
        直近期間の正社員・非正規雇用の成長率を計算

        Returns:
            Tuple[float, float]: (正社員成長率, 非正規雇用成長率)
        """
        if len(dataframe) < 2 or start_year >= end_year:
            return 0.0, 0.0

        return (
            EmploymentDataProcessor.calculate_employment_growth_rate(
                dataframe, start_year, end_year, Config.COLUMN_REGULAR_EMPLOYEE
            ),
            EmploymentDataProcessor.calculate_employment_growth_rate(
                dataframe, start_year, end_year, Config.COLUMN_NON_REGULAR_EMPLOYEE
            )
        )

    @staticmethod
    def _add_recent_period_reference_line(axes: plt.Axes, start_year: int) -> None:
        """直近期間の開始位置に縦の参照線を追加"""
        axes.axvline(x=start_year, color='gray', linestyle=':', linewidth=2, alpha=0.5)
        axes.text(
            start_year, 82, f'← {Config.RECENT_PERIOD_YEARS}年前',
            fontsize=10, ha='right', color='gray', fontweight='bold'
        )

    @staticmethod
    def _add_growth_rate_display_box(
        axes: plt.Axes,
        regular_employee_growth_rate: float,
        non_regular_employee_growth_rate: float,
        start_year: int,
        end_year: int
    ) -> None:
        """成長率情報を表示するボックスをグラフ上部に追加"""
        axes.text(
            0.5, 0.96,
            f'【直近{end_year - start_year}年間（{start_year}-{end_year}年）の増加率】\n'
            f'正社員：+{regular_employee_growth_rate:.1f}%　　'
            f'非正規雇用：+{non_regular_employee_growth_rate:.1f}%',
            transform=axes.transAxes,
            fontsize=Config.FONT_SIZE_GROWTH_RATE_BOX,
            fontweight='bold', ha='center', va='top',
            bbox=dict(
                boxstyle='round,pad=0.8',
                facecolor='lightyellow',
                edgecolor='orange',
                linewidth=2
            )
        )

    @staticmethod
    def _add_latest_value_annotations(axes: plt.Axes, dataframe: pd.DataFrame) -> None:
        """最新年のデータポイントに値を示す注釈を追加"""
        latest_row = dataframe.iloc[-1]

        for column_name, color, y_offset in [
            (Config.COLUMN_REGULAR_EMPLOYEE, Config.COLOR_REGULAR_EMPLOYEE, 10),
            (Config.COLUMN_NON_REGULAR_EMPLOYEE, Config.COLOR_NON_REGULAR_EMPLOYEE, -25)
        ]:
            axes.annotate(
                f'{latest_row[column_name]:.0f}万人',
                xy=(latest_row[Config.COLUMN_YEAR], latest_row[column_name]),
                xytext=(10, y_offset), textcoords='offset points',
                fontsize=Config.FONT_SIZE_VALUE_ANNOTATION,
                fontweight='bold', color=color,
                bbox=dict(
                    boxstyle='round,pad=0.5',
                    facecolor='white',
                    edgecolor=color,
                    linewidth=2
                ),
                arrowprops=dict(arrowstyle='->', color=color, lw=2)
            )

    @staticmethod
    def _add_graph_legend(axes: plt.Axes) -> None:
        """グラフに凡例を追加"""
        axes.legend(
            fontsize=Config.FONT_SIZE_LEGEND,
            loc='upper left',
            framealpha=0.95,
            edgecolor='black',
            fancybox=True
        )

    @staticmethod
    def _add_data_source_note(axes: plt.Axes) -> None:
        """グラフ左下にデータ出典情報を追加"""
        axes.text(
            0.02, 0.02, Config.DATA_SOURCE_TEXT,
            transform=axes.transAxes,
            fontsize=Config.FONT_SIZE_DATA_SOURCE,
            va='bottom', ha='left', style='italic', color='#555555',
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor='#f0f0f0',
                alpha=0.8,
                edgecolor='gray'
            )
        )


# ==================== 統計情報表示 ====================

class EmploymentStatisticsPrinter:
    """
    雇用統計サマリーをコンソールに出力するクラス

    最新年の状況、成長率、年次データをテーブル形式で表示します。
    """

    @staticmethod
    def display_employment_summary(dataframe: pd.DataFrame) -> None:
        """
        統計サマリー全体をコンソールに出力

        Args:
            dataframe: 雇用データ
        """
        print("\n" + "=" * Config.SEPARATOR_LINE_LENGTH)
        print("プログラマー雇用形態統計サマリー")
        print("=" * Config.SEPARATOR_LINE_LENGTH)

        EmploymentStatisticsPrinter._display_latest_year_statistics(dataframe)
        EmploymentStatisticsPrinter._display_growth_rate_statistics(dataframe)
        EmploymentStatisticsPrinter._display_yearly_data_table(dataframe)

        print("\n" + "=" * Config.SEPARATOR_LINE_LENGTH)

    @staticmethod
    def _display_latest_year_statistics(dataframe: pd.DataFrame) -> None:
        """最新年のデータ（人数・比率・内訳）を表示"""
        latest_year = int(dataframe[Config.COLUMN_YEAR].max())
        latest_data = dataframe[dataframe[Config.COLUMN_YEAR] == latest_year].iloc[0]

        print(f"\n【{latest_year}年の状況】")
        print(f"正社員: {latest_data[Config.COLUMN_REGULAR_EMPLOYEE]:.0f}万人 "
              f"({100 - latest_data[Config.COLUMN_NON_REGULAR_RATIO]:.1f}%)")
        print(f"非正規雇用: {latest_data[Config.COLUMN_NON_REGULAR_EMPLOYEE]:.0f}万人 "
              f"({latest_data[Config.COLUMN_NON_REGULAR_RATIO]:.1f}%)")

        # 非正規雇用の内訳を表示（データが存在する場合のみ）
        for breakdown_column in Config.NON_REGULAR_BREAKDOWN_COLUMNS:
            if breakdown_column in latest_data.index:
                display_label = breakdown_column.split('（')[0]
                print(f"  - {display_label}: {latest_data[breakdown_column]:.0f}万人")

        print(f"合計: {latest_data[Config.COLUMN_TOTAL_EMPLOYEE]:.0f}万人")

    @staticmethod
    def _display_growth_rate_statistics(dataframe: pd.DataFrame) -> None:
        """全期間および直近期間の成長率を表示"""
        if len(dataframe) < 2:
            return

        print(f"\n【期間別増加率】")

        data_min_year = int(dataframe[Config.COLUMN_YEAR].min())
        data_max_year = int(dataframe[Config.COLUMN_YEAR].max())

        # 全期間の成長率
        print(f"全期間（{data_min_year}-{data_max_year}年）:")
        print(f"  正社員: +{EmploymentDataProcessor.calculate_employment_growth_rate(dataframe, data_min_year, data_max_year, Config.COLUMN_REGULAR_EMPLOYEE):.1f}%")
        print(f"  非正規雇用: +{EmploymentDataProcessor.calculate_employment_growth_rate(dataframe, data_min_year, data_max_year, Config.COLUMN_NON_REGULAR_EMPLOYEE):.1f}%")

        # 直近期間の成長率（データが十分ある場合のみ）
        if len(dataframe) >= Config.RECENT_PERIOD_YEARS:
            recent_start_year = max(data_min_year, data_max_year - (Config.RECENT_PERIOD_YEARS - 1))
            print(f"\n直近{data_max_year - recent_start_year}年間（{recent_start_year}-{data_max_year}年）:")
            print(f"  正社員: +{EmploymentDataProcessor.calculate_employment_growth_rate(dataframe, recent_start_year, data_max_year, Config.COLUMN_REGULAR_EMPLOYEE):.1f}%")
            print(f"  非正規雇用: +{EmploymentDataProcessor.calculate_employment_growth_rate(dataframe, recent_start_year, data_max_year, Config.COLUMN_NON_REGULAR_EMPLOYEE):.1f}%")

    @staticmethod
    def _display_yearly_data_table(dataframe: pd.DataFrame) -> None:
        """年次データをテーブル形式で表示"""
        print(f"\n【年次データ（抜粋）】")
        print(dataframe[[
            Config.COLUMN_YEAR,
            Config.COLUMN_REGULAR_EMPLOYEE,
            Config.COLUMN_NON_REGULAR_EMPLOYEE,
            Config.COLUMN_TOTAL_EMPLOYEE,
            Config.COLUMN_NON_REGULAR_RATIO
        ]].to_string(index=False))


# ==================== コマンドライン処理 ====================

def parse_command_line_arguments() -> argparse.Namespace:
    """
    コマンドライン引数を解析

    Returns:
        argparse.Namespace: 解析された引数
    """
    argument_parser = argparse.ArgumentParser(
        description='プログラマー雇用形態推移グラフ作成プログラム',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # サンプルデータを使用（デフォルト）
  python3 programmer_employment_graph.py

  # CSVファイルから実データを読み込む
  python3 programmer_employment_graph.py --input data_source.csv

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

    argument_parser.add_argument(
        '-i', '--input',
        type=str,
        help=f'入力CSVファイルのパス（指定しない場合、{Config.DEFAULT_INPUT_FILENAME}を自動検出）'
    )

    argument_parser.add_argument(
        '-o', '--output',
        type=str,
        default=Config.DEFAULT_OUTPUT_GRAPH_FILENAME,
        help=f'出力画像ファイル名（デフォルト: {Config.DEFAULT_OUTPUT_GRAPH_FILENAME}）'
    )

    return argument_parser.parse_args()


def execute_main_program() -> None:
    """
    プログラムのメイン実行フロー

    1. コマンドライン引数を解析
    2. データを読み込みまたは生成
    3. グラフを作成して保存
    4. 統計サマリーを表示
    5. 結果をCSVファイルに保存
    """
    command_line_args = parse_command_line_arguments()

    # 入力CSVファイルの決定（指定がなければ自動検出を試みる）
    input_csv_path = command_line_args.input
    if not input_csv_path and os.path.exists(Config.DEFAULT_INPUT_FILENAME):
        input_csv_path = Config.DEFAULT_INPUT_FILENAME
        print(f"\n{Config.DEFAULT_INPUT_FILENAME} を検出しました。実データとして使用します。\n")

    # データ読み込みまたはサンプルデータ生成
    try:
        employment_dataframe = EmploymentDataProcessor.load_or_create_dataframe(input_csv_path)
    except (FileNotFoundError, ValueError) as error:
        print(f"エラー: {error}", file=sys.stderr)
        print("サンプルデータの使用を続けるには、引数なしで実行してください。", file=sys.stderr)
        sys.exit(1)

    # グラフ作成と保存
    graph_plotter = EmploymentGraphPlotter()
    graph_plotter.create_and_save_graph(employment_dataframe, command_line_args.output)

    # 統計サマリー表示
    EmploymentStatisticsPrinter.display_employment_summary(employment_dataframe)

    # データをCSVファイルに保存
    employment_dataframe.to_csv(
        Config.DEFAULT_OUTPUT_CSV_FILENAME,
        index=False,
        encoding='utf-8-sig'
    )
    print(f"\nデータをCSVに保存しました: {Config.DEFAULT_OUTPUT_CSV_FILENAME}")

    plt.close()


if __name__ == '__main__':
    execute_main_program()
