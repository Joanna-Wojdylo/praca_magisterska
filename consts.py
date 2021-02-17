import os

SRC = 'src'
DATA = 'data'
RAW = 'raw'
RAW_DIR = os.path.join(DATA, RAW)

# data files
BITCOIN_FILE = 'BTC-USD.csv'
ETHEREUM_FILE = 'ETH-USD.csv'
LITECOIN_FILE = 'LTC-USD.csv'
RIPPLE_FILE = 'XRP-USD.csv'
TETHER_FILE = 'USDT-USD.csv'
CHAINLINK_FILE = 'LINK-USD.csv'
NEM_FILE = 'XEM-USD.csv'
STELLAR_FILE = 'XLM-USD.csv'


# column names
DEFAULT_PRICE_COLUMN_NAME = 'prices'
STANDARDIZED_PRICE_COLUMN_NAME = 'standardized prices'
