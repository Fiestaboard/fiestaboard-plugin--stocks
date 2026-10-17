"""Tests for the stocks settings picker (remote-options ``get_options``)."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

from src.plugins.base import OptionsRequest, OptionsUnavailable

from plugins.stocks import StocksPlugin

MANIFEST = json.loads((Path(__file__).parent.parent / "manifest.json").read_text())


def _plugin(config=None):
    plugin = StocksPlugin(MANIFEST)
    plugin.config = config or {}
    return plugin


def _response(payload):
    response = Mock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


def test_manifest_symbols_field_uses_remote_options():
    field = MANIFEST["settings_schema"]["properties"]["symbols"]
    assert field["ui:widget"] == "remote-options"
    opts = field["ui:options"]
    assert opts["options_id"] == "symbols"
    assert opts["multiple"] is True
    assert opts["server_search"] is True
    assert opts["allow_custom"] is True


def test_unknown_options_id_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        _plugin().get_options(OptionsRequest(options_id="nope", query="a"))


def test_empty_query_returns_no_options_without_network():
    with patch("plugins.stocks.requests.get") as get:
        result = _plugin().get_options(OptionsRequest(options_id="symbols", query="  "))
    get.assert_not_called()
    assert result.options == []


def test_yahoo_search_without_key():
    payload = {
        "quotes": [
            {"symbol": "aapl", "shortname": "Apple Inc.", "exchDisp": "NASDAQ", "typeDisp": "Equity"},
            {"symbol": "AAPL", "shortname": "Apple dup"},
            {"symbol": "", "shortname": "no symbol"},
            {"symbol": "APLE", "longname": "Apple Hospitality REIT"},
        ]
    }
    with patch("plugins.stocks.requests.get", return_value=_response(payload)) as get:
        result = _plugin().get_options(OptionsRequest(options_id="symbols", query="apple"))
    assert "yahoo" in get.call_args.args[0]
    assert get.call_args.kwargs["timeout"]
    assert [o.value for o in result.options] == ["AAPL", "APLE"]
    assert result.options[0].description == "Apple Inc. · NASDAQ"
    assert result.options[0].group == "Equity"
    assert result.total == 2


def test_finnhub_search_with_key():
    payload = {"result": [{"symbol": "MSFT", "description": "MICROSOFT CORP", "type": "Common Stock"}]}
    with patch("plugins.stocks.requests.get", return_value=_response(payload)) as get:
        result = _plugin({"finnhub_api_key": "k"}).get_options(
            OptionsRequest(options_id="symbols", query="micro")
        )
    assert "finnhub" in get.call_args.args[0]
    assert get.call_args.kwargs["params"]["token"] == "k"
    assert result.options[0].value == "MSFT"
    assert result.options[0].description == "MICROSOFT CORP"


def test_limit_pages_results():
    payload = {"quotes": [{"symbol": f"S{i}"} for i in range(5)]}
    with patch("plugins.stocks.requests.get", return_value=_response(payload)):
        result = _plugin().get_options(OptionsRequest(options_id="symbols", query="s", limit=2))
    assert len(result.options) == 2
    assert result.has_more is True
    assert result.total == 5


def test_network_failure_raises_options_unavailable():
    with patch("plugins.stocks.requests.get", side_effect=requests.ConnectionError("down")):
        with pytest.raises(OptionsUnavailable):
            _plugin().get_options(OptionsRequest(options_id="symbols", query="x"))
