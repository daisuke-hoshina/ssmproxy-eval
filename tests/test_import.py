import importlib


def test_import_ssmproxy_package():
    module = importlib.import_module("ssmproxy")
    assert module.__version__ == "0.1.0"
