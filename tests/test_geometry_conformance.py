"""Board-geometry conformance for the Stocks plugin.

Runs the shared FiestaBoard conformance suite (src.plugins.geometry_conformance)
against this plugin: every board shape from a Note (15x3) up to the largest
note_array/FiestaPanel (120x24), including shapes that are narrower than a
Flagship but taller, and wider but shorter. See that module's docstring for
why both matter.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from src.plugins.geometry_conformance import assert_board_conformance

from plugins.stocks import StocksPlugin

MANIFEST = json.loads((Path(__file__).resolve().parent.parent / "manifest.json").read_text())

# Five symbols -- the configured maximum (validate_config rejects more) --
# so growth from a short board (which can only fit a couple of rows) to a
# tall one (which can fit all five) is a genuine content increase, not
# padding. GOOGL's large price and wide swing double as the stress case for
# the stocks.*.formatted max_length declared in the manifest: it is the
# widest single row the code will actually produce for this fixture.
_TICKER_DATA = {
    "AAPL": {"price": 178.50, "name": "Apple Inc.", "history": [176.00, 176.50, 177.00, 177.80, 178.50]},
    "GOOGL": {"price": 3421.55, "name": "Alphabet Inc.", "history": [2800.00, 2850.00, 2900.00, 3000.00, 3421.55]},
    "MSFT": {"price": 415.20, "name": "Microsoft Corporation", "history": [413.00, 413.50, 414.00, 414.50, 415.20]},
    "AMZN": {"price": 178.90, "name": "Amazon.com, Inc.", "history": [182.00, 181.00, 180.00, 179.50, 178.90]},
    "TSLA": {"price": 248.50, "name": "Tesla, Inc.", "history": [250.00, 249.50, 249.00, 248.80, 248.50]},
}


def _mock_ticker(symbol, *_args, **_kwargs):
    """Build one mock yfinance.Ticker keyed by symbol.

    A ``side_effect`` callable (rather than a fixed ``return_value``) so it
    keeps answering correctly across however many times the conformance
    suite re-fetches -- once per distinct board geometry it renders, times
    however many instances ``run_conformance`` constructs.
    """
    info = _TICKER_DATA[symbol]
    ticker = MagicMock()
    ticker.info = {"regularMarketPrice": info["price"], "longName": info["name"]}
    ticker.history.return_value = pd.DataFrame({"Close": info["history"]})
    return ticker


def make_plugin() -> StocksPlugin:
    """Return a fresh, configured StocksPlugin. Caller must stub the network."""
    plugin = StocksPlugin(manifest=MANIFEST)
    plugin.config = {
        "symbols": list(_TICKER_DATA.keys()),
        "time_window": "1 Day",
    }
    return plugin


def test_renders_on_every_board_shape():
    """The plugin must render within bounds on every board shape, and grow with height."""
    with patch("yfinance.Ticker", side_effect=_mock_ticker):
        assert_board_conformance(
            make_plugin,
            manifest=MANIFEST,
            strict_growth=True,
            require_note_array_preview=True,
        )
