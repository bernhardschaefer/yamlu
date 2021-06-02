import torch

from yamlu.pytorch import isin, indices_to_mask


# noinspection PyArgumentList
def test_indices_to_mask():
    indices = torch.LongTensor([0, 2, 3])
    mask_length = 5
    mask = indices_to_mask(indices, mask_length)
    assert torch.equal(mask, torch.BoolTensor([True, False, True, True, False]))


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
