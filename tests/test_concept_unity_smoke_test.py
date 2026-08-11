import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np


# def test_main_loads_dotenv_before_using_preparation_helpers(monkeypatch) -> None:
#     script_path = Path(__file__).parents[1] / "scripts" / "concept_unity_smoke_test.py"
#     spec = importlib.util.spec_from_file_location("concept_unity_smoke_test", script_path)
#     assert spec is not None
#     assert spec.loader is not None
#     concept_unity_smoke_test = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(concept_unity_smoke_test)
#     assert isinstance(concept_unity_smoke_test, ModuleType)
#     dotenv_loaded = False

#     def load_dotenv() -> None:
#         nonlocal dotenv_loaded
#         dotenv_loaded = True

#     def generate_statements(*args, **kwargs) -> list[str]:
#         assert dotenv_loaded
#         return ["statement"] * concept_unity_smoke_test.STATEMENT_COUNT

#     monkeypatch.setattr(concept_unity_smoke_test, "load_dotenv", load_dotenv, raising=False)
#     monkeypatch.setattr(concept_unity_smoke_test, "generate_statements", generate_statements)
#     monkeypatch.setattr(
#         concept_unity_smoke_test,
#         "calculate_embeddings",
#         lambda *args, **kwargs: np.ones((concept_unity_smoke_test.STATEMENT_COUNT, 2)),
#     )
#     monkeypatch.setattr(concept_unity_smoke_test, "concept_unity", lambda embeddings: 1.0)

#     concept_unity_smoke_test.main()
