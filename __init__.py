"""Stock Prices plugin for FiestaBoard.

Displays real-time stock prices using Yahoo Finance.
"""

from typing import Any, Dict, List, Optional
import logging

import requests

from src.devices import BoardContext
from src.plugins.base import (
    Option,
    OptionsRequest,
    OptionsResult,
    OptionsUnavailable,
    PluginBase,
    PluginResult,
)
from src.text_to_board import count_tiles, take_tiles

logger = logging.getLogger(__name__)

# Time window mapping
TIME_WINDOW_MAP = {
    "1 Day": "1d",
    "5 Days": "5d",
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y",
    "ALL": "max",
}

# Fallback board when a plugin is rendered outside a board-scoped call (unit
# tests, legacy callers). ``self.board`` is None there; treating that as a
# Flagship keeps behavior unchanged for every caller that never adopted the
# board-aware render path. This is the one place board dimensions are ever
# hardcoded -- it is the documented default, not a layout decision.
_DEFAULT_BOARD = BoardContext(device_type="flagship", rows=6, cols=22)

# The header consumes one row whenever there is more than one row available;
# a 1-row board (never actually issued -- the narrowest is a 3-row Note) would
# have nothing left for content, so the header is skipped there instead.
_HEADER_TEXT = "STOCKS"

# Symbol search for the settings picker (remote-options). Finnhub is used when
# the user supplied a key; Yahoo's public search endpoint is the keyless default.
OPTIONS_TIMEOUT_SECONDS = 10
FINNHUB_SEARCH_URL = "https://finnhub.io/api/v1/search"
YAHOO_SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"
_YAHOO_HEADERS = {"User-Agent": "Mozilla/5.0 (FiestaBoard stocks plugin)"}


class StocksPlugin(PluginBase):
    """Stock prices plugin.

    Fetches real-time stock data from Yahoo Finance via yfinance.
    """

    def __init__(self, manifest: Dict[str, Any]):
        """Initialize the stocks plugin."""
        super().__init__(manifest)
        self._cache: Optional[Dict[str, Any]] = None

    @property
    def plugin_id(self) -> str:
        return "stocks"

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate stocks configuration."""
        errors = []

        symbols = config.get("symbols", [])
        if not symbols:
            errors.append("At least one stock symbol is required")
        elif len(symbols) > 5:
            errors.append("Maximum 5 stock symbols allowed")

        time_window = config.get("time_window", "1 Day")
        if time_window not in TIME_WINDOW_MAP:
            errors.append(f"Invalid time window: {time_window}")

        return errors

    # ------------------------------------------------------------------
    # Settings picker (remote-options)
    # ------------------------------------------------------------------

    def get_options(self, request: OptionsRequest) -> OptionsResult:
        """Search ticker symbols so the user can pick them by company name."""
        if request.options_id != "symbols":
            raise NotImplementedError(request.options_id)

        query = (request.query or "").strip()
        if not query:
            # Nothing typed yet: no catalog to browse, the user must search.
            return OptionsResult(options=[], total=0)

        api_key = str(self.config.get("finnhub_api_key") or "").strip()
        try:
            if api_key:
                options = self._search_finnhub(query, api_key)
            else:
                options = self._search_yahoo(query)
        except (requests.RequestException, ValueError) as e:
            logger.warning("Stock symbol search failed: %s", e)
            raise OptionsUnavailable("Symbol search is unavailable right now. You can still type a symbol.")

        # Drop duplicate symbols, keeping the first (best-ranked) match.
        seen = set()
        unique: List[Option] = []
        for option in options:
            if option.value in seen:
                continue
            seen.add(option.value)
            unique.append(option)

        limit = request.limit if request.limit and request.limit > 0 else len(unique)
        page = unique[:limit]
        return OptionsResult(options=page, has_more=len(unique) > len(page), total=len(unique))

    @staticmethod
    def _search_finnhub(query: str, api_key: str) -> List[Option]:
        response = requests.get(
            FINNHUB_SEARCH_URL,
            params={"q": query, "token": api_key},
            timeout=OPTIONS_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        options = []
        for item in response.json().get("result") or []:
            symbol = str(item.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            name = str(item.get("description") or "").strip()
            options.append(
                Option(
                    value=symbol,
                    label=symbol,
                    description=name or None,
                    group=item.get("type") or None,
                )
            )
        return options

    @staticmethod
    def _search_yahoo(query: str) -> List[Option]:
        response = requests.get(
            YAHOO_SEARCH_URL,
            params={"q": query, "quotesCount": 20, "newsCount": 0},
            headers=_YAHOO_HEADERS,
            timeout=OPTIONS_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        options = []
        for item in response.json().get("quotes") or []:
            symbol = str(item.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            name = str(item.get("shortname") or item.get("longname") or "").strip()
            exchange = str(item.get("exchDisp") or item.get("exchange") or "").strip()
            description = " · ".join(part for part in (name, exchange) if part)
            options.append(
                Option(
                    value=symbol,
                    label=symbol,
                    description=description or None,
                    group=item.get("typeDisp") or item.get("quoteType") or None,
                )
            )
        return options

    def fetch_data(self) -> PluginResult:
        """Fetch stock data for all configured symbols."""
        try:
            import yfinance as yf
        except ImportError:
            return PluginResult(
                available=False,
                error="yfinance library not installed"
            )

        symbols = self.config.get("symbols", [])
        if not symbols:
            return PluginResult(
                available=False,
                error="No stock symbols configured"
            )

        time_window = self.config.get("time_window", "1 Day")
        period = TIME_WINDOW_MAP.get(time_window, "1d")

        try:
            stocks_data = []

            for symbol in symbols[:5]:
                stock_data = self._fetch_single_stock(symbol, period)
                if stock_data:
                    stocks_data.append(stock_data)

            if not stocks_data:
                return PluginResult(
                    available=False,
                    error="Failed to fetch any stock data"
                )

            # Align formatting across all stocks
            stocks_data = self._align_formatting(stocks_data)

            # Primary stock (first one)
            primary = stocks_data[0]

            data = {
                # Primary stock fields
                "symbol": primary["symbol"],
                "current_price": primary["current_price"],
                "previous_price": primary["previous_price"],
                "change_percent": primary["change_percent"],
                "change_direction": primary["change_direction"],
                "formatted": primary["formatted"],
                "company_name": primary["company_name"],
                # Aggregate
                "symbol_count": len(stocks_data),
                # Array of all stocks
                "stocks": stocks_data,
            }

            self._cache = data

            # ``formatted_lines`` is the live whole-board path (src/displays/
            # service.py). self.board is bound by PluginBase.get_data() for
            # the duration of this call; None here means "no board scoped"
            # (legacy callers, unit tests calling fetch_data() directly).
            board = self.board or _DEFAULT_BOARD
            lines = self._render_rows(stocks_data, board)

            return PluginResult(available=True, data=data, formatted_lines=lines)

        except Exception as e:
            logger.exception("Error fetching stock data")
            return PluginResult(available=False, error=str(e))

    def _fetch_single_stock(self, symbol: str, period: str) -> Optional[Dict]:
        """Fetch data for a single stock."""
        import yfinance as yf

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            current_price = info.get("regularMarketPrice") or info.get("currentPrice")
            if current_price is None:
                return None

            # Get historical data for comparison
            if period == "1d":
                hist = ticker.history(period="5d")
            else:
                hist = ticker.history(period=period)

            if hist.empty:
                return None

            # Calculate previous price
            if period == "1d" and len(hist) >= 2:
                previous_price = float(hist.iloc[-2]["Close"])
            else:
                previous_price = float(hist.iloc[0]["Close"])

            current_price = float(current_price)

            # Calculate change
            if previous_price > 0:
                change_percent = ((current_price - previous_price) / previous_price) * 100
            else:
                change_percent = 0.0

            change_direction = "up" if change_percent >= 0 else "down"

            # Determine color
            if change_percent > 0:
                color_tile = "{66}"  # green
            elif change_percent < 0:
                color_tile = "{63}"  # red
            else:
                color_tile = "{69}"  # white

            company_name = info.get("longName") or info.get("shortName") or symbol

            return {
                "symbol": symbol.upper(),
                "current_price": current_price,
                "previous_price": previous_price,
                "change_percent": round(change_percent, 2),
                "change_direction": change_direction,
                "color_tile": color_tile,
                "company_name": company_name,
                "formatted": "",  # Will be set in alignment step
            }

        except Exception as e:
            logger.error(f"Error fetching stock {symbol}: {e}")
            return None

    def _align_formatting(self, stocks: List[Dict]) -> List[Dict]:
        """Align price and percentage formatting across all stocks."""
        if not stocks:
            return stocks

        # Calculate max widths
        max_price_width = 0
        max_percent_width = 0

        for stock in stocks:
            price_str = f"${stock['current_price']:.2f}"
            percent_str = f"{'+' if stock['change_percent'] >= 0 else ''}{stock['change_percent']:.2f}%"
            max_price_width = max(max_price_width, len(price_str))
            max_percent_width = max(max_percent_width, len(percent_str))

        # Apply aligned formatting
        for stock in stocks:
            price_str = f"${stock['current_price']:.2f}".rjust(max_price_width)
            sign = "+" if stock['change_percent'] >= 0 else ""
            percent_str = f"{sign}{stock['change_percent']:.2f}%".rjust(max_percent_width)
            stock["formatted"] = f"{stock['symbol']}{stock['color_tile']} {price_str} {percent_str}"

        return stocks

    def _render_rows(self, stocks: List[Dict], board: BoardContext) -> List[str]:
        """Render *stocks* as board lines sized to *board*.

        Every dimension here is derived from ``board.rows``/``board.cols`` --
        a taller board shows more tickers (up to however many are
        configured), a wider one lets each ticker row carry more detail, and
        a narrower one abbreviates. Width is checked in tiles (``{66}`` is
        one tile, four characters), never characters.
        """
        rows, cols = board.rows, board.cols

        lines: List[str] = []
        header_rows = 1 if rows > 1 else 0
        if header_rows:
            lines.append(_HEADER_TEXT.center(cols))

        max_stock_rows = max(rows - header_rows, 0)
        for stock in stocks[:max_stock_rows]:
            lines.append(self._format_row(stock, cols))

        # Pad to a full board fill (never truncate a row we already built --
        # only ever add blank trailing rows), matching the "single page"
        # contract of a complete board frame.
        while len(lines) < rows:
            lines.append("")

        return lines[:rows]

    @staticmethod
    def _format_row(stock: Dict[str, Any], cols: int) -> str:
        """Format one stock as a single row that fits within *cols* tiles.

        Tries progressively more compact representations -- the richest adds
        the company name when a wide board (a note_array) has room to spare,
        down to a bare symbol + percent -- and returns the first one that
        fits. If even the most compact form doesn't fit (an unusually long
        symbol on the narrowest board), it is hard-truncated tile-safely
        rather than allowed to overflow.
        """
        symbol = stock["symbol"]
        color = stock["color_tile"]
        price = stock["current_price"]
        percent = stock["change_percent"]
        sign = "+" if percent >= 0 else ""
        company = stock.get("company_name") or ""

        base = f"{symbol}{color} ${price:.2f} {sign}{percent:.2f}%"
        candidates = []
        if company:
            candidates.append(f"{base}  {company}")
        candidates.append(base)
        candidates.append(f"{symbol}{color} {price:.2f} {sign}{percent:.1f}%")
        candidates.append(f"{symbol[:4]}{color} {price:.0f} {sign}{percent:.0f}%")

        for candidate in candidates:
            if count_tiles(candidate) <= cols:
                return candidate

        head, _ = take_tiles(candidates[-1], cols)
        return head

    def get_formatted_display(self) -> Optional[List[str]]:
        """Return the board-fitted stocks display for the currently bound board."""
        if not self._cache:
            result = self.fetch_data()
            if not result.available:
                return None
            # fetch_data() always sets self._cache before returning a result
            # with available=True, so self._cache is guaranteed truthy here.

        stocks = self._cache.get("stocks", [])
        board = self.board or _DEFAULT_BOARD
        return self._render_rows(stocks, board)


# Export the plugin class
Plugin = StocksPlugin
