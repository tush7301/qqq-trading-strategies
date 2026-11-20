### ma_crossover - v2_reentry

Params: `{"short_window": 50, "long_window": 250, "ma_type": "sma", "band": 0.0, "allow_short": false, "reentry_window": 50, "vol_target": null, "vol_window": 20, "vol_band": 0.1}`  
Costs: 5 bps one-way. Periods: train <= 2018-12-31, validation 2019-01-01 .. 2021-12-31.

|                           | Mean daily ret   | Std daily ret   | Ann. mean ret   | CAGR   | Ann. vol   |   Sharpe |   Sortino | Cum. ret   | Max DD   |   Calmar | Hit rate   | Exposure   |   # trades |   Turnover/yr |
|:--------------------------|:-----------------|:----------------|:----------------|:-------|:-----------|---------:|----------:|:-----------|:---------|---------:|:-----------|:-----------|-----------:|--------------:|
| v2_reentry · train        | 0.053%           | 1.043%          | 13.2%           | 12.6%  | 16.5%      |     0.73 |      1.02 | 426.0%     | -30.6%   |     0.41 | 61.5%      | 85.9%      |         38 |          2.72 |
| buy_and_hold · train      | 0.050%           | 1.274%          | 12.6%           | 11.1%  | 20.2%      |     0.56 |      0.8  | 337.9%     | -53.4%   |     0.21 | 55.2%      | 100.0%     |          1 |          0.07 |
| v2_reentry · validation   | 0.133%           | 1.554%          | 33.5%           | 35.6%  | 24.7%      |     1.33 |      1.87 | 149.7%     | -28.6%   |     1.25 | 60.3%      | 98.8%      |          1 |          0.33 |
| buy_and_hold · validation | 0.140%           | 1.569%          | 35.3%           | 37.9%  | 24.9%      |     1.38 |      1.96 | 162.8%     | -28.6%   |     1.33 | 59.9%      | 100.0%     |          0 |          0    |
