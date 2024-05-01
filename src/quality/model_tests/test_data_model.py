# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import unittest
import gc
import os
import shutil
from datetime import datetime
from typing import Any
from src.configuration import configuration as cfg
from src.model.model_control.model_database import ModelDatabase, sqlalchemy_utility


TESTING_DB_PATH = f"{cfg.PATHS.TEST_PATH}/DataModelTest/backend.db"


class DataModelTest(unittest.TestCase):
    """
    Test case class for testing the data model.
    """

    def test_01_infrastructure(self) -> None:
        """
        Method for testing basic infrastructure.
        """
        self.assertTrue(os.path.exists(TESTING_DB_PATH))
        self.assertTrue(all(hasattr(self.database, attribute) for attribute in [
                        "base", "engine", "model", "session_factory", "primary_keys"]))
        self.assertEqual(len(
            [c for c in self.database.base.metadata.tables[f"{self.schema}model"].columns]), len(self.model_columns))
        self.assertEqual(len(
            [c for c in self.database.base.metadata.tables[f"{self.schema}modelversion"].columns]), len(self.modelversion_columns))
        self.assertEqual(len(
            [c for c in self.database.base.metadata.tables[f"{self.schema}modelinstance"].columns]), len(self.modelinstance_columns))
        self.assertEqual(len(
            [c for c in self.database.base.metadata.tables[f"{self.schema}asset"].columns]), len(self.asset_columns))
        self.assertEqual(len(
            [c for c in self.database.base.metadata.tables[f"{self.schema}log"].columns]), len(self.log_columns))

    def test_02_key_constraints(self) -> None:
        """
        Method for testing key constraints.
        """
        self.assertEqual(
            list(self.database.base.metadata.tables[f"{self.schema}model"].primary_key.columns)[0].name, "id")
        self.assertTrue(
            isinstance(list(self.database.base.metadata.tables[f"{self.schema}model"].primary_key.columns)[0].type, sqlalchemy_utility.Integer))
        self.assertEqual(list(self.database.base.metadata.tables[f"{self.schema}modelversion"].foreign_keys)[
            0].column.table.name, f"{self.schema}model")
        self.assertEqual(list(self.database.base.metadata.tables[f"{self.schema}modelversion"].foreign_keys)[
            0].target_fullname, f"{self.schema}model.id")
        for foreign_key in list(self.database.base.metadata.tables[f"{self.schema}modelinstance"].foreign_keys):
            self.assertTrue(foreign_key.column.table.name in [
                            f"{self.schema}model", f"{self.schema}modelversion"])
            self.assertTrue(foreign_key.target_fullname in [
                            f"{self.schema}model.id", f"{self.schema}modelversion.id"])
        self.assertEqual(
            list(self.database.base.metadata.tables[f"{self.schema}modelinstance"].primary_key.columns)[0].name, "uuid")
        self.assertTrue(
            isinstance(list(self.database.base.metadata.tables[f"{self.schema}modelinstance"].primary_key.columns)[0].type, sqlalchemy_utility.Uuid))
        self.assertEqual(
            list(self.database.base.metadata.tables[f"{self.schema}log"].primary_key.columns)[0].name, "id")
        self.assertTrue(
            isinstance(list(self.database.base.metadata.tables[f"{self.schema}log"].primary_key.columns)[0].type, sqlalchemy_utility.Integer))

    def test_03_model_key_representation(self) -> None:
        """
        Method for testing model representation.
        """
        primary_keys = {
            object_class: self.database.model[object_class].__mapper__.primary_key[0].name for object_class in self.database.model}
        self.assertEqual(primary_keys["model"], "id")
        self.assertEqual(primary_keys["modelinstance"], "uuid")
        self.assertEqual(primary_keys["log"], "id")

    def test_04_model_object_interaction(self) -> None:
        """
        Method for testing model representation.
        """
        model = self.database.model["model"](
            **self.example_model_data
        )
        self.assertTrue(all(hasattr(model, attribute)
                        for attribute in self.model_columns))

        instance = self.database.model["modelinstance"](
            **self.example_modelinstance_data
        )
        self.assertTrue(all(hasattr(instance, attribute)
                        for attribute in self.modelinstance_columns))

        log = self.database.model["log"](
            **self.example_log_data
        )
        self.assertTrue(all(hasattr(log, attribute)
                        for attribute in self.log_columns))

        model_id = None
        with self.database.session_factory() as session:
            session.add(model)
            session.commit()
            model_id = model.id
        self.assertFalse(model_id is None)

        with self.database.session_factory() as session:
            resulting_model = session.query(self.database.model["model"]).filter(
                getattr(self.database.model["model"], "id") == model_id).first()
        self.assertFalse(resulting_model is None)

        self.assertTrue(isinstance(resulting_model.created,
                        datetime))
        with self.database.session_factory() as session:
            resulting_model = session.query(self.database.model["model"]).filter(
                getattr(self.database.model["model"], "id") == model_id).first()
            resulting_model.path = "my_new_path"
            session.commit()
            session.refresh(resulting_model)
        self.assertEqual(resulting_model.path, "my_new_path")

        instance.model_id = model_id
        with self.database.session_factory() as session:
            session.add(instance)
            session.commit()
            session.refresh(instance)
            resulting_model = session.query(self.database.model["model"]).filter(
                getattr(self.database.model["model"], "id") == model_id).first()
            self.assertEqual(instance.model_id, resulting_model.id)
            self.assertTrue(isinstance(
                instance.model, self.database.model["model"]))

    @classmethod
    def setUpClass(cls):
        """
        Class method for setting up test case.
        """
        if not os.path.exists(f"{cfg.PATHS.TEST_PATH}/DataModelTest"):
            os.makedirs(f"{cfg.PATHS.TEST_PATH}/DataModelTest")
        cls.schema = "test."
        cls.database = ModelDatabase(
            database_uri=f"sqlite:///{TESTING_DB_PATH}",
            schema=cls.schema,
            verbose=True)

        cls.example_model_data = {
            "path": "TheBloke_vicuna-7B-v1.3-GGML",
            "task": "text-generation",
            "architecture": "llama"
        }
        cls.example_modelversion_data = {
            "path": "vicuna-7b-v1.3.ggmlv3.q4_0.bin",
            "format": "ggml",
            "quantization": "q4_0"
        }
        cls.example_modelinstance_data = {
            "backend": "llamacpp",
            "loader": "_default",
            "loader_kwargs": {
                "n_ctx": 2048,
                "verbose": True
            }
        }
        cls.example_log_data = {"request":
                                {"my_request_key": "my_request_value"}}
        cls.model_columns = ["id", "path", "name", "task", "type", "architecture",
                             "url", "source", "meta_data", "created", "updated", "inactive"]
        cls.modelversion_columns = ["id", "path", "name", "basemodel", "type", "format", "url", "source", "sha256",
                                    "meta_data", "created", "updated", "inactive", "model_id"]
        cls.modelinstance_columns = ["uuid", "backend", "loader", "loader_kwargs", "gateway",
                                     "meta_data", "created", "updated", "inactive", "model_id", "modelversion_id"]
        cls.asset_columns = ["id", "path", "type", "url", "sha256",
                             "meta_data", "created", "updated", "inactive", "model_id", "modelversion_id"]
        cls.scraping_fails_columns = ["id", "url", "source", "fetched_data", "normalized_data",
                                      "exception_data", "created", "inactive"]
        cls.log_columns = ["id", "request",
                           "response", "requested", "responded"]

    @classmethod
    def tearDownClass(cls):
        """
        Class method for setting tearing down test case.
        """
        del cls.database
        del cls.schema
        del cls.example_model_data
        del cls.example_modelinstance_data
        del cls.example_log_data
        del cls.model_columns
        del cls.modelversion_columns
        del cls.modelinstance_columns
        del cls.asset_columns
        del cls.log_columns
        if os.path.exists(cfg.PATHS.TEST_PATH):
            shutil.rmtree(cfg.PATHS.TEST_PATH, ignore_errors=True)
        gc.collect()

    @classmethod
    def setup_class(cls):
        """
        Alternative class method for setting up test case.
        """
        cls.setUpClass()

    @classmethod
    def teardown_class(cls):
        """
        Alternative class for setting tearing down test case.
        """
        cls.tearDownClass()


if __name__ == '__main__':
    unittest.main()
