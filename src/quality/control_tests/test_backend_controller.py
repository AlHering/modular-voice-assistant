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
from src.control.backend_controller import BackendController, UUID
from src.model.language_model_control.llm_pool import LLMPool
from src.configuration import configuration as cfg


TESTING_PATH = cfg.PATHS.TEST_PATH


class BackendControllerTest(unittest.TestCase):
    """
    Test case class for testing the backend controller.
    """

    def test_01_initiation_process(self):
        """
        Method for testing controller instantiation.
        """
        self.assertTrue(os.path.exists(TESTING_PATH))
        self.assertTrue(all(getattr(self.controller, attribute) is not None for attribute in [
                        "base", "engine", "model", "session_factory", "primary_keys", "_cache", "llm_pool"]))
        self.assertTrue(isinstance(self.controller.llm_pool, LLMPool))

    def test_02_object_handling(self):
        """
        Method for testing object handling.
        """
        model_id = self.controller.post_object(
            "model", **self.example_model_data)
        self.assertTrue(isinstance(model_id, int))
        self.assertEqual(model_id, 1)
        registered_models = self.controller.get_objects("model")
        self.assertEqual(len(registered_models), 1)
        self.assertEqual(registered_models[0].id, model_id)
        self.assertEqual(
            registered_models[0].type, self.example_model_data["type"])
        self.controller.patch_object(
            "model", model_id, **self.example_model_patch)
        patched_model_object = self.controller.get_object(
            "model", model_id)
        for key in self.example_model_patch:
            self.assertEqual(getattr(patched_model_object, key),
                             self.example_model_patch[key])

        instance_uuid = self.controller.post_object(
            "instance", **self.example_instance_data)
        registered_instances = self.controller.get_objects("instance")
        self.assertTrue(isinstance(instance_uuid, UUID))
        self.assertEqual(len(registered_instances), 1)
        self.assertEqual(registered_instances[0].uuid, instance_uuid)

        self.controller.delete_object("model", model_id)
        self.controller.delete_object("instance", instance_uuid)
        self.assertEqual(len(self.controller.get_objects("model")), 0)
        self.assertEqual(len(self.controller.get_objects("instance")), 0)

    def test_03_llm_handling(self):
        """
        Method for testing LLM handling.
        """
        model_id = self.controller.post_object(
            "model", **self.example_model_data)
        self.example_instance_data["model_id"] = model_id
        instance_uuid = str(self.controller.post_object(
            "instance", **self.example_instance_data))
        self.assertTrue(len(self.controller.get_objects("model")) > 0)
        self.assertTrue(len(self.controller.get_objects("instance")) > 0)

        self.controller.load_instance(str(instance_uuid))
        response = self.controller.forward_generate(
            instance_uuid, "You are an assistant. Give me a list of fruit trees.")
        print(response)
        self.assertTrue(all(hasattr(response, attribute)
                        for attribute in ["generations", "llm_output", "run"]))
        generation_batches = response.generations
        self.assertTrue(isinstance(generation_batches, list))
        self.assertTrue(len(generation_batches) > 0)
        self.assertTrue(isinstance(generation_batches[0], list))
        self.assertTrue(len(generation_batches[0]) > 0)
        self.assertTrue(isinstance(generation_batches[0][0].text, str))

        self.controller.unload_instance(instance_uuid)
        self.assertFalse(self.controller.llm_pool.is_running(instance_uuid))

        self.controller.delete_object("model", model_id)
        self.controller.delete_object("instance", UUID(instance_uuid))
        self.assertEqual(len(self.controller.get_objects("model")), 0)
        self.assertEqual(len(self.controller.get_objects("instance")), 0)

    @classmethod
    def setUpClass(cls):
        """
        Class method for setting up test case.
        """
        if not os.path.exists(TESTING_PATH):
            os.makedirs(TESTING_PATH)
        cls.controller = BackendController(
            working_directory=TESTING_PATH)
        cls.controller.llm_pool.generation_timeout = 360.0
        cls.example_model_data = {"path": "TheBloke_vicuna-7B-v1.3-GGML",
                                  "type": "llamacpp"}
        cls.example_model_patch = {"type": "exllama"}
        cls.example_instance_data = {
            "type": "llamacpp",
            "loader": "_default",
            "model_version": "vicuna-7b-v1.3.ggmlv3.q4_0.bin",
            "loader_kwargs": {
                "n_ctx": 2048,
                "verbose": True
            },
            "model_id": 1
        }

    @classmethod
    def tearDownClass(cls):
        """
        Class method for setting tearing down test case.
        """
        del cls.controller
        del cls.example_model_data
        del cls.example_model_patch
        del cls.example_instance_data
        if os.path.exists(TESTING_PATH):
            shutil.rmtree(TESTING_PATH, ignore_errors=True)
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
        gc.collect()


if __name__ == '__main__':
    unittest.main()
