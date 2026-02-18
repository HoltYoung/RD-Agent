# US Equity Factor Data

## daily_pv.h5

Daily price-volume data for ~600 US equities.
- Read with: `pd.read_hdf("daily_pv.h5", key="data")`
- Index: MultiIndex [datetime, instrument]
- Columns:
  - `$open`: Opening price
  - `$high`: High price
  - `$low`: Low price
  - `$close`: Closing price
  - `$volume`: Trading volume
  - `$factor`: Adjustment factor (adj_close / close)

## fundamentals.h5

XBRL financial statement data forward-filled to daily frequency.
- Read with: `pd.read_hdf("fundamentals.h5", key="data")`
- Index: MultiIndex [datetime, instrument]
- Balance sheet columns: prefixed with `xbrl_bs_`
- Income statement columns: prefixed with `xbrl_is_`
- Cash flow columns: prefixed with `xbrl_cf_`

## Usage in factor code

```python
import pandas as pd

# Load price data
df = pd.read_hdf("daily_pv.h5", key="data")
# df.index has levels: ['datetime', 'instrument']
# Access: df['$close'], df['$volume'], etc.

# Load fundamentals (if needed)
fund = pd.read_hdf("fundamentals.h5", key="data")
```
