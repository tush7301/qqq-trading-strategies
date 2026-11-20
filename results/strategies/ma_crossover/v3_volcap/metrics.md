### ma_crossover - v3_volcap

Params: `{"short_window": 50, "long_window": 250, "ma_type": "sma", "band": 0.0, "allow_short": false, "reentry_window": 50, "vol_target": "train", "vol_window": 20, "vol_band": 0.1}`  
Costs: 5 bps one-way. Periods: train <= 2018-12-31, validation 2019-01-01 .. 2021-12-31.

|                           | Mean daily ret   | Std daily ret   | Ann. mean ret   | CAGR   | Ann. vol   |   Sharpe |   Sortino | Cum. ret   | Max DD   |   Calmar | Hit rate   | Exposure   |   # trades |   Turnover/yr |
|:--------------------------|:-----------------|:----------------|:----------------|:-------|:-----------|---------:|----------:|:-----------|:---------|---------:|:-----------|:-----------|-----------:|--------------:|
| v3_volcap · train         | 0.050%           | 0.933%          | 12.5%           | 12.1%  | 14.8%      |     0.76 |      1.06 | 393.9%     | -26.8%   |     0.45 | 61.5%      | 81.7%      |        114 |          2.71 |
| buy_and_hold · train      | 0.050%           | 1.274%          | 12.6%           | 11.1%  | 20.2%      |     0.56 |      0.8  | 337.9%     | -53.4%   |     0.21 | 55.2%      | 100.0%     |          1 |          0.07 |
| v3_volcap · validation    | 0.105%           | 1.123%          | 26.4%           | 28.1%  | 17.8%      |     1.44 |      1.98 | 110.6%     | -19.0%   |     1.48 | 60.3%      | 88.5%      |         45 |          2.06 |
| buy_and_hold · validation | 0.140%           | 1.569%          | 35.3%           | 37.9%  | 24.9%      |     1.38 |      1.96 | 162.8%     | -28.6%   |     1.33 | 59.9%      | 100.0%     |          0 |          0    |
