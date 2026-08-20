import json

from ablation_test.scripts.run_newgen_2_0_benchmark import schema_for_provider


def test_claude_schema_removes_unsupported_constraints_recursively():
    schema = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "minItems": 14,
                "maxItems": 14,
                "items": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "integer", "minimum": 0, "maximum": 4},
                        "reason": {"type": "string", "maxLength": 500},
                    },
                },
            },
        },
        "required": ["rows"],
    }

    cleaned = schema_for_provider(schema, "claude")
    encoded = json.dumps(cleaned)

    for unsupported in ("minItems", "maxItems", "minimum", "maximum", "maxLength"):
        assert unsupported not in encoded
    assert cleaned["required"] == ["rows"]
    assert cleaned["properties"]["rows"]["items"]["properties"]["score"]["type"] == "integer"


def test_non_claude_schema_is_unchanged():
    schema = {"type": "array", "minItems": 14, "maxItems": 14}

    assert schema_for_provider(schema, "openai") is schema
    assert schema_for_provider(schema, "qwen") is schema
