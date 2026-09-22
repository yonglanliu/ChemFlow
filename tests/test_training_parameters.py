import json
import tomllib

from chemflow.desktop.training_parameters import (
    MODEL_PARAMETER_SCHEMAS,
    serialize_configuration,
)


def test_every_desktop_training_schema_serializes():
    for label, schema in MODEL_PARAMETER_SCHEMAS.items():
        values = {parameter.path: parameter.default for parameter in schema.parameters}
        content = serialize_configuration(label, values)

        parsed = json.loads(content) if schema.extension == ".json" else tomllib.loads(content)

        assert parsed


def test_ml_schema_builds_model_list():
    schema = MODEL_PARAMETER_SCHEMAS["Conventional ML"]
    values = {parameter.path: parameter.default for parameter in schema.parameters}

    parsed = json.loads(serialize_configuration("Conventional ML", values))

    assert parsed["models"] == [
        {"model_name": "LightGBM", "estimator": "LGBMRegressor"}
    ]


def test_empty_optional_values_are_omitted_from_toml():
    schema = MODEL_PARAMETER_SCHEMAS["Graphormer"]
    values = {parameter.path: parameter.default for parameter in schema.parameters}

    parsed = tomllib.loads(serialize_configuration("Graphormer", values))

    assert "test_dataset_path" not in parsed["DatasetConfig"]
    assert "task_types" not in parsed["DatasetConfig"]
    assert "resume_checkpoint" not in parsed["TrainingConfig"]
