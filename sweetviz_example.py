#!/usr/bin/env python
# coding: utf-8

# ライブラリの読み込み
import numpy as np
import pandas as pd
# グラフィックスのフォントはソースコード内で "Noto Sans CJK" に決め打ちされている
import sweetviz as sv
sv.config_parser.read('sweetviz_settings.ini')


# データの読み込み
df1 = pd.read_csv('https://raw.githubusercontent.com/ltl-manabi/py_sweetviz_example/main/pseudo_classification_data.csv')
df2 = pd.read_csv('https://raw.githubusercontent.com/ltl-manabi/py_sweetviz_example/main/weather_data_mod.csv')


# Sweetvizインスタンスの作成
my_report1 = sv.analyze(df1)
my_report2 = sv.analyze(df2)


# レポートの作成
my_report1.show_html(filepath='sweetviz_report01.html', open_browser=False)
my_report2.show_html(filepath='sweetviz_report02.html', open_browser=False)
my_report2.show_html(filepath='sweetviz_report02_vertical.html',
                     open_browser=False, layout='vertical')


# データの比較 (`compare_intra()` 関数、`compare()` 関数)


# データフレーム内の比較 (`compare_intra()` 関数)

# 月ラベルの付与
df2['月'] = np.where(df2['日時'] < '2021-05-01', 4, 5)

# データフレームから数値型の列のみ抽出
# compare_intra() 関数では、文字列型の列があるとエラーになる
df2 = df2.select_dtypes(include='number')

# Sweetvizインスタンスの作成
my_report3 = sv.compare_intra(df2, df2['月'] == 4, ['4月', '5月'])

# レポートの作成
my_report3.show_html(filepath='sweetviz_report03.html',
                     open_browser=False, layout='vertical')


# データフレーム間の比較 (`compare()` 関数)

# データフレームの分割
df1_train = df1.sample(frac=0.7, random_state=334)
df1_test = df1.drop(df1_train.index)

# Sweetvizインスタンスの作成
my_report4 = sv.compare([df1_train, 'Training'], [df1_test, 'Test'], 'target')

# レポートの作成
my_report4.show_html(filepath='sweetviz_report04.html',
                     open_browser=False, layout='vertical')
