import os

import torch

from silc import Client

CLUSTER = True


def test_set_model(mock_model):
    model = mock_model.create_torch_cnn()
    c = Client(None, CLUSTER)
    c.set_model("simple_cnn", model, "TORCH", "CPU")
    returned_model = c.get_model("simple_cnn")
    assert model == returned_model


def test_set_model_from_file(mock_model):
    try:
        mock_model.create_torch_cnn(filepath="./torch_cnn.pt")
        c = Client(None, CLUSTER)
        c.set_model_from_file("file_cnn", "./torch_cnn.pt", "TORCH", "CPU")
        returned_model = c.get_model("file_cnn")
        with open("./torch_cnn.pt", "rb") as f:
            model = f.read()
        assert model == returned_model
    finally:
        os.remove("torch_cnn.pt")


def test_torch_inference(mock_model):
    # get model and set into database
    model = mock_model.create_torch_cnn()
    c = Client(None, CLUSTER)
    c.set_model("torch_cnn", model, "TORCH")

    # setup input tensor
    data = torch.rand(1, 1, 3, 3).numpy()
    c.put_tensor("torch_cnn_input", data)

    # run model and get output
    c.run_model("torch_cnn", inputs=["torch_cnn_input"], outputs=["torch_cnn_output"])
    out_data = c.get_tensor("torch_cnn_output")
    assert out_data.shape == (1, 1, 1, 1)
