### ma_crossover - v1

Params: `{"short_window": 50, "long_window": 200, "ma_type": "sma", "band": 0.0, "allow_short": false, "reentry_window": null, "vol_target": null, "vol_window": 20, "vol_band": 0.1}`  
Costs: 5 bps one-way. Periods: train <= 2018-12-31, validation 2019-01-01 .. 2021-12-31.

|                           | Mean daily ret   | Std daily ret   | Ann. mean ret   | CAGR   | Ann. vol   |   Sharpe |   Sortino | Cum. ret   | Max DD   |   Calmar | Hit rate   | Exposure   |   # trades |   Turnover/yr |
|:--------------------------|:-----------------|:----------------|:----------------|:-------|:-----------|---------:|----------:|:-----------|:---------|---------:|:-----------|:-----------|-----------:|--------------:|
| v1 · train                | 0.041%           | 0.929%          | 10.4%           | 9.8%   | 14.7%      |     0.62 |      0.86 | 268.1%     | -25.4%   |     0.38 | 66.6%      | 75.7%      |         18 |          1.29 |
| buy_and_hold · train      | 0.050%           | 1.274%          | 12.6%           | 11.1%  | 20.2%      |     0.56 |      0.8  | 337.9%     | -53.4%   |     0.21 | 55.2%      | 100.0%     |          1 |          0.07 |
| v1 · validation           | 0.109%           | 1.529%          | 27.6%           | 27.9%  | 24.3%      |     1.1  |      1.55 | 109.3%     | -28.6%   |     0.98 | 63.1%      | 90.4%      |          3 |          1    |
| buy_and_hold · validation | 0.140%           | 1.569%          | 35.3%           | 37.9%  | 24.9%      |     1.38 |      1.96 | 162.8%     | -28.6%   |     1.33 | 59.9%      | 100.0%     |          0 |          0    |
