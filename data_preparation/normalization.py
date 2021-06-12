from sklearn import preprocessing

columns_to_be_normalized = [
    'last_close', 'second_last_close', 'last_high', 'second_last_high', 'last_low', 'second_last_low',
    'deviation_from_5_min_avg', 'deviation_from_10_min_avg', 'deviation_from_15_min_avg', 'deviation_from_30_min_avg',
    'last_5_min_high', 'last_10_min_high', 'last_30_min_high', 'last_5_min_low', 'last_10_min_low', 'last_30_min_low',
    'last_5_min_interval', 'last_10_min_interval', 'last_30_min_interval', 'daily_avg_until_now',
    'deviation_from_daily_avg_until_now', 'previous_day_close', 'sma_12min', 'sma_12day', 'sma_20min', 'sma_20day',
    'sma_50min', 'ema_12min', 'ema_12day', 'ema_20min', 'ema_20day', 'ema_50min', 'dev_from_sma_12min',
    'dev_from_sma_12day', 'dev_from_sma_20min', 'dev_from_sma_50min', 'dev_from_sma_20day', 'dev_from_ema_12min',
    'dev_from_ema_12day', 'dev_from_ema_20min', 'dev_from_ema_50min', 'dev_from_ema_20day', 'diff_sma50_sma20min',
    'diff_sma20_sma12min', 'diff_sma50_sma12min', 'diff_ema50_ema20min', 'diff_ema20_ema12min', 'diff_ema50_ema12min',
    'diff_ema50_sma50min', 'diff_ema20_sma20min', 'diff_ema12_sma12min', 'rsi_14min', 'rsi_14day', 'macd_min',
    'macd_signal_min', 'macd_diff_min', 'macd_day', 'macd_signal_day', 'macd_diff_day', 'bb_high_band_min',
    'bb_low_band_min', 'bollinger_hband_indicator_min', 'bollinger_lband_indicator_min', 'bollinger_pband_min',
    'bollinger_wband_min', 'bb_high_band_day', 'bb_low_band_day', 'bollinger_hband_indicator_day',
    'bollinger_lband_indicator_day', 'bollinger_pband_day', 'bollinger_wband_day', 'mfi_14min', 'mfi_14day',
    'stochastic_osc_14min', 'stochastic_osc_14min_signal', 'stochastic_osc_14day', 'stochastic_osc_14day_signal',
    'obv_14min', 'obv_14day', 'chaikin_oscillator_20min', 'chaikin_oscillator_20day', 'ichimoku_a_52min',
    'ichimoku_b_52min', 'ichimoku_base_52min', 'ichimoku_conversion_52min', 'ichimoku_a_52day', 'ichimoku_b_52day',
    'ichimoku_base_52day', 'ichimoku_conversion_52day', 'sma_50day', 'dev_from_sma_50day', 'ema_50day',
    'dev_from_ema_50day'
]


def normalize_features(df):

    scaler = preprocessing.MinMaxScaler()
    df[columns_to_be_normalized] = scaler.fit_transform(df[columns_to_be_normalized])
    return df, scaler
