import torch

from yamlu.pytorch import isin


# noinspection PyArgumentList
def test_isin():
    element = torch.LongTensor([0, 1, 3, 2, 1, 2])
    test_elements = torch.LongTensor([0, 1])
    res = isin(element, test_elements)
    assert res.tolist() == [1, 1, 0, 0, 1, 0]

    res = isin(element.to(torch.int), test_elements.to(torch.int))
    assert res.tolist() == [1, 1, 0, 0, 1, 0]

    res = isin(element, [0, 1])
    assert res.tolist() == [1, 1, 0, 0, 1, 0]
