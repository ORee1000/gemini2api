import json
from decimal import Decimal
import pytest
from app.utils.prompt import _readable_json_arguments

@pytest.mark.parametrize("value", [r'{"x":"\u4e2d", "x":"\u6587", "n":1.2345678901234567890123456789}',json.dumps({"code":r'print("\u4e2d")',"emoji":"😀"}), '["text",-0,1e999]'])
def test_argument_values_preserved(value):
    out=_readable_json_arguments(value)
    parse=lambda s:json.loads(s,parse_float=Decimal,object_pairs_hook=list)
    assert parse(out)==parse(value)
    assert _readable_json_arguments(out)==out

@pytest.mark.parametrize("value", ["not json",r'{"x":"\ud800"}',r'{"broken":'])
def test_invalid_arguments_pass_through(value):
    assert _readable_json_arguments(value)==value

def test_cjk_history_no_longer_inflates_sixfold():
    value=json.dumps({"log":"设备状态正常。"*30000})
    out=_readable_json_arguments(value)
    assert len(value)>1_000_000>len(out)
    assert json.loads(value)==json.loads(out)
