from selected_technical_analysis_features import features
from os.path import join as path
import pandas as pd

import ta.volatility
import ta.momentum
import ta.trend
import ta.volume
import joblib


class TechnicalAnalysisFeatures:

    def __init__(self, label):

        self.grouped = joblib.load(path("..", "data", "grouped_by_date", label+".joblib"))
        self.daily = joblib.load(path("..", "data", "daily", label+".joblib"))
        self.raw = pd.read_csv(path("..", "data", "imputed_1min", label+".csv"))

        self.features = features

        self.daily_feature_names = [
            'sma_12day', 'sma_20day', 'ema_12day', 'ema_20day', 'dev_from_sma_12day', 'dev_from_sma_20day',
            'dev_from_ema_12day', 'dev_from_ema_20day', 'rsi_14day', 'macd_day', 'macd_signal_day',
            'macd_diff_day', 'bb_high_band_day', 'bb_low_band_day', 'bollinger_hband_indicator_day',
            'bollinger_lband_indicator_day', 'bollinger_pband_day', 'bollinger_wband_day', 'mfi_14day',
            'stochastic_osc_14day', 'stochastic_osc_14day_signal', 'obv_14day', 'chaikin_oscillator_20day'
        ]

        self.converted = False

    def _set_sma_minute_values(self):
        self.features["sma_12min"] = ta.trend.sma_indicator(close=self.raw["close"], window=12, fillna=True)
        self.features["sma_20min"] = ta.trend.sma_indicator(close=self.raw["close"], window=20, fillna=True)
        self.features["sma_50min"] = ta.trend.sma_indicator(close=self.raw["close"], window=50, fillna=True)
        self.features["dev_from_sma_12min"] = pd.Series(self.raw["close"]) - self.features["sma_12min"]
        self.features["dev_from_sma_20min"] = pd.Series(self.raw["close"]) - self.features["sma_20min"]
        self.features["dev_from_sma_50min"] = pd.Series(self.raw["close"]) - self.features["sma_50min"]

    def _set_sma_daily_values(self):
        self.features["sma_12day"] = ta.trend.sma_indicator(close=self.daily["close"], window=12, fillna=True)
        self.features["sma_20day"] = ta.trend.sma_indicator(close=self.daily["close"], window=20, fillna=True)
        self.features["sma_50day"] = ta.trend.sma_indicator(close=self.daily["close"], window=50, fillna=True)
        self.features["dev_from_sma_12day"] = pd.Series(self.daily["close"]) - self.features["sma_12day"]
        self.features["dev_from_sma_20day"] = pd.Series(self.daily["close"]) - self.features["sma_20day"]
        self.features["dev_from_sma_50day"] = pd.Series(self.daily["close"]) - self.features["sma_50day"]

    def set_sma_features(self):
        self._set_sma_minute_values()
        self._set_sma_daily_values()

    def _set_ema_minute_values(self):
        self.features["ema_12min"] = ta.trend.ema_indicator(close=self.raw["close"], window=12, fillna=True)
        self.features["ema_20min"] = ta.trend.ema_indicator(close=self.raw["close"], window=20, fillna=True)
        self.features["ema_50min"] = ta.trend.ema_indicator(close=self.raw["close"], window=50, fillna=True)
        self.features["dev_from_ema_12min"] = pd.Series(self.raw["close"]) - self.features["ema_12min"]
        self.features["dev_from_ema_20min"] = pd.Series(self.raw["close"]) - self.features["ema_20min"]
        self.features["dev_from_ema_50min"] = pd.Series(self.raw["close"]) - self.features["ema_50min"]

    def _set_ema_daily_values(self):
        self.features["ema_12day"] = ta.trend.ema_indicator(close=self.daily["close"], window=12, fillna=True)
        self.features["ema_20day"] = ta.trend.ema_indicator(close=self.daily["close"], window=20, fillna=True)
        self.features["ema_50day"] = ta.trend.ema_indicator(close=self.daily["close"], window=50, fillna=True)
        self.features["dev_from_ema_12day"] = pd.Series(self.daily["close"]) - self.features["ema_12day"]
        self.features["dev_from_ema_20day"] = pd.Series(self.daily["close"]) - self.features["ema_20day"]
        self.features["dev_from_ema_50day"] = pd.Series(self.daily["close"]) - self.features["ema_50day"]

    def set_ema_features(self):
        self._set_ema_minute_values()
        self._set_ema_daily_values()

    def set_diff_features(self):
        self.features["diff_sma50_sma20min"] = self.features["sma_50min"] - self.features["sma_20min"]
        self.features["diff_sma20_sma12min"] = self.features["sma_20min"] - self.features["sma_12min"]
        self.features["diff_sma50_sma12min"] = self.features["sma_50min"] - self.features["sma_12min"]

        self.features["diff_ema50_ema20min"] = self.features["ema_50min"] - self.features["ema_20min"]
        self.features["diff_ema20_ema12min"] = self.features["ema_20min"] - self.features["ema_12min"]
        self.features["diff_ema50_ema12min"] = self.features["ema_50min"] - self.features["ema_12min"]

        self.features["diff_ema50_sma50min"] = self.features["ema_50min"] - self.features["sma_50min"]
        self.features["diff_ema20_sma20min"] = self.features["ema_20min"] - self.features["sma_20min"]
        self.features["diff_ema12_sma12min"] = self.features["ema_12min"] - self.features["sma_12min"]

    def set_rsi_features(self):
        self.features["rsi_14min"] = ta.momentum.rsi(close=self.raw["close"], window=14, fillna=True)
        self.features["rsi_14day"] = ta.momentum.rsi(close=self.daily["close"], window=14, fillna=True)

    def set_macd_features(self):
        self.features["macd_min"] = ta.trend.macd(close=self.raw["close"], fillna=True)
        self.features["macd_signal_min"] = ta.trend.macd_signal(close=self.raw["close"], fillna=True)
        self.features["macd_diff_min"] = ta.trend.macd_diff(close=self.raw["close"], fillna=True)

        self.features["macd_day"] = ta.trend.macd(close=self.daily["close"], fillna=True)
        self.features["macd_signal_day"] = ta.trend.macd_signal(close=self.daily["close"], fillna=True)
        self.features["macd_diff_day"] = ta.trend.macd_diff(close=self.daily["close"], fillna=True)

    def set_bollinger_bands_features(self):

        bb = ta.volatility.BollingerBands(close=self.daily["close"], window=20, fillna=True)
        self.features["bb_high_band_day"] = bb.bollinger_hband()
        self.features["bb_low_band_day"] = bb.bollinger_lband()

        self.features["bollinger_hband_indicator_day"] = bb.bollinger_hband_indicator()
        self.features["bollinger_lband_indicator_day"] = bb.bollinger_lband_indicator()

        self.features["bollinger_pband_day"] = bb.bollinger_pband()
        self.features["bollinger_wband_day"] = bb.bollinger_wband()

    def set_mfi_features(self):
        self.features["mfi_14min"] = ta.volume.money_flow_index(
            high=self.raw["high"],
            low=self.raw["low"],
            close=self.raw["close"],
            volume=self.raw["volume"],
            fillna=True
        )
        self.features["mfi_14day"] = ta.volume.money_flow_index(
            high=self.daily["high"],
            low=self.daily["low"],
            close=self.daily["close"],
            volume=self.daily["volume"],
            fillna=True
        )

    def set_stochastic_oscillator_features(self):
        so_min = ta.momentum.StochasticOscillator(
            high=self.raw["high"],
            low=self.raw["low"],
            close=self.raw["close"],
            fillna=True
        )

        so_day = ta.momentum.StochasticOscillator(
            high=self.daily["high"],
            low=self.daily["low"],
            close=self.daily["close"],
            fillna=True
        )

        self.features["stochastic_osc_14min"] = so_min.stoch()
        self.features["stochastic_osc_14min_signal"] = so_min.stoch_signal()

        self.features["stochastic_osc_14day"] = so_day.stoch()
        self.features["stochastic_osc_14day_signal"] = so_day.stoch_signal()

    def set_obv_features(self):
        self.features["obv_14min"] = ta.volume.on_balance_volume(
            close=self.raw["close"],
            volume=self.raw["volume"],
            fillna=True
        )
        self.features["obv_14day"] = ta.volume.on_balance_volume(
            close=self.daily["close"],
            volume=self.daily["volume"],
            fillna=True
        )

    def set_chaikin_features(self):
        self.features["chaikin_oscillator_20min"] = ta.volume.chaikin_money_flow(
            high=self.raw["high"],
            low=self.raw["low"],
            close=self.raw["close"],
            volume=self.raw["volume"],
            fillna=True
        )
        self.features["chaikin_oscillator_20day"] = ta.volume.chaikin_money_flow(
            high=self.daily["high"],
            low=self.daily["low"],
            close=self.daily["close"],
            volume=self.daily["volume"],
            fillna=True
        )

    def set_ichimoku_features(self):
        ichimoku_min = ta.trend.IchimokuIndicator(high=self.raw["high"], low=self.raw["low"], fillna=True)
        self.features["ichimoku_a_52min"] = ichimoku_min.ichimoku_a()
        self.features["ichimoku_b_52min"] = ichimoku_min.ichimoku_b()
        self.features["ichimoku_base_52min"] = ichimoku_min.ichimoku_base_line()
        self.features["ichimoku_conversion_52min"] = ichimoku_min.ichimoku_conversion_line()

        ichimoku_day = ta.trend.IchimokuIndicator(high=self.daily["high"], low=self.daily["low"], fillna=True)
        self.features["ichimoku_a_52day"] = ichimoku_day.ichimoku_a()
        self.features["ichimoku_b_52day"] = ichimoku_day.ichimoku_b()
        self.features["ichimoku_base_52day"] = ichimoku_day.ichimoku_base_line()
        self.features["ichimoku_conversion_52day"] = ichimoku_day.ichimoku_conversion_line()

    def _convert_daily_feature_to_minutes(self, feature_name):
        multipliers = list()
        new_feature = list()

        feature_values = self.features[feature_name]

        for i in self.grouped.values():
            multipliers.append(len(i))

        for i in range(len(feature_values)):
            new_feature += [feature_values[i]] * multipliers[i]

        self.features[feature_name] = new_feature

    def convert_all_daily_features(self):
        if not self.converted:
            self.converted = True
            for feat in self.daily_feature_names:
                self._convert_daily_feature_to_minutes(feat)

    def run_all(self):
        self.set_sma_features()
        self.set_ema_features()
        self.set_diff_features()
        self.set_rsi_features()
        self.set_macd_features()
        self.set_bollinger_bands_features()
        self.set_mfi_features()
        self.set_stochastic_oscillator_features()
        self.set_obv_features()
        self.set_chaikin_features()
        self.set_ichimoku_features()
        self.convert_all_daily_features()

        return pd.DataFrame(data=self.features)
