import unittest
import numpy as np
import torch as torch
import torch.nn as nn
import model.base as base
from numpy.testing import assert_allclose, assert_equal


class TestNormalizer1D(unittest.TestCase):

    def test_normalize_ones(self):
        x = torch.ones((3, 4, 5))
        norml = base.Normalizer1D([2.0, 2.0, 2.0, 2.0], [0.5, 0.5, 0.5, 0.5])
        x_norml= norml.normalize(x)
        x_unorml = norml.unnormalize(x_norml)

        assert_allclose(x, x_unorml)
        assert_allclose(x_norml, 0.25*torch.ones((3, 4, 5)))
        assert_equal(x.shape, x_norml.shape)
        assert_equal(x_unorml.shape, x_norml.shape)

    def test_range(self):
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        norml = base.Normalizer1D(2.0, 0.5)
        x_norml = norml.normalize(x)
        x_unorml = norml.unnormalize(x_norml)

        assert_allclose(x, x_unorml)
        assert_allclose(x_norml, (x - 0.5) / 2)
        assert_equal(x.shape, x_norml.shape)
        assert_equal(x_unorml.shape, x_norml.shape)

    def test_range_2signals(self):
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]]])
        norml = base.Normalizer1D([2.0, 1.0], [0.5, 1.0])
        x_norml = norml.normalize(x)
        x_unorml = norml.unnormalize(x_norml)

        assert_allclose(x, x_unorml)
        assert_allclose(x_norml, torch.tensor([[[0.25, 0.75, 1.25, 1.75], [1.0, 2.0, 3.0, 4.0]]]))
        assert_equal(x.shape, x_norml.shape)
        assert_equal(x_unorml.shape, x_norml.shape)


class TestCausalConv(unittest.TestCase):

    def test_requested_output_eq_same(self):
        for in_shape in [(3, 2, 5), (5, 2, 3), (1, 2, 20)]:
            for out_dim in [1, 2, 3, 4]:
                for kernel_size in [1, 2, 3, 4, 5, 6]:
                    x = torch.ones(in_shape)
                    conv = base.CausalConv(in_shape[1], out_dim, kernel_size)
                    conv.set_requested_output('same')
                    y = conv(x)
                    assert_equal(y.shape, (in_shape[0], out_dim, in_shape[2]))

    def test_requested_output_eq_same_2(self):
        in_dim = 2
        out_dim = 3
        n_batches = 5
        for subsampl in range(1, 4):
            for kernel_size in range(1, 8):
                conv = base.CausalConv(in_dim, out_dim, kernel_size, subsampl=subsampl)
                conv.set_mode('dilation')
                for in_length in range(1, 20, 2):
                    conv.set_requested_output('same')
                    x = torch.ones((n_batches, in_dim, in_length))
                    y = conv(x)
                    assert_equal(y.shape, (n_batches, out_dim, in_length))

    def test_requested_output_is_none_1(self):
        in_dim = 3
        for out_shape in [(3, 2, 1), (5, 2, 3), (1, 2, 20), (1, 3, 5)]:
            for kernel_size in [1, 2, 3, 4, 5, 6]:
                conv = base.CausalConv(in_dim, out_shape[1], kernel_size)
                conv.set_requested_output(None)
                input_len = conv.get_requested_input(out_shape[2])
                x = torch.ones((out_shape[0], in_dim, input_len))
                y = conv(x)
                assert_equal(y.shape, out_shape)

    def test_requested_output_is_none_2(self):
        in_dim = 2
        out_dim = 3
        n_batches = 5
        for subsampl in range(1, 4):
            for kernel_size in range(1, 8):
                conv = base.CausalConv(in_dim, out_dim, kernel_size, subsampl=subsampl)
                conv.set_mode('dilation')
                conv.set_requested_output(None)
                for out_length in range(1, 20, 2):
                    in_length = conv.get_requested_input(out_length)
                    x = torch.ones((n_batches, in_dim, in_length))
                    y = conv(x)
                    assert_equal(y.shape, (n_batches, out_dim, out_length))

    def test_requested_output_is_none_stride(self):
        in_dim = 2
        out_dim = 3
        n_batches = 5
        for subsampl in range(1, 4):
            for kernel_size in range(1, 8):
                conv = base.CausalConv(in_dim, out_dim, kernel_size, subsampl=subsampl)
                conv.set_mode('stride')
                conv.set_requested_output(None)
                for out_length in range(1, 20, 2):
                    in_length = conv.get_requested_input(out_length)
                    x = torch.ones((n_batches, in_dim, in_length))
                    y = conv(x)
                    assert_equal(y.shape, (n_batches, out_dim, out_length))

    def test_requested_output_is_number(self):
        in_dim = 2
        out_dim = 3
        n_batches = 5
        for subsampl in range(1, 4):
            for kernel_size in range(1, 8):
                conv = base.CausalConv(in_dim, out_dim, kernel_size, subsampl=subsampl)
                conv.set_mode('dilation')
                for in_length in range(1, 20, 2):
                    start_len = max(in_length - subsampl*(kernel_size - 1), 1)
                    for out_length in range(start_len, 20, 2):
                        conv.set_requested_output(out_length)
                        x = torch.ones((n_batches, in_dim, in_length))
                        y = conv(x)
                        assert_equal(y.shape, (n_batches, out_dim, out_length))

    def test_requested_output_is_number_stride(self):
        in_dim = 2
        out_dim = 3
        n_batches = 5
        for subsampl in range(1, 4):
            for kernel_size in range(1, 8):
                conv = base.CausalConv(in_dim, out_dim, kernel_size, subsampl=subsampl)
                conv.set_mode('stride')
                for in_length in range(1, 20, 2):
                    start_len = max(int(np.ceil(in_length/subsampl)) - kernel_size+1, 1)
                    for out_length in range(start_len, 20, 2):
                        conv.set_requested_output(out_length)
                        x = torch.ones((n_batches, in_dim, in_length))
                        y = conv(x)
                        assert_equal(y.shape, (n_batches, out_dim, out_length))

    def test_requested_output_is_number_stride_eqto_dilation(self):
        in_dim = 2
        out_dim = 3
        n_batches = 5
        torch.manual_seed(1)
        for subsampl in range(2, 4):
            for kernel_size in range(2, 8):
                conv = base.CausalConv(in_dim, out_dim, kernel_size, subsampl=subsampl)
                conv.set_requested_output(None)
                for out_length in range(1, 10):
                    conv.set_mode('stride')
                    in_length_stride = conv.get_requested_input(out_length)
                    conv.set_mode('dilation')
                    in_length_dilation = conv.get_requested_input(subsampl*(out_length-1) + 1)

                    assert_equal(in_length_stride, in_length_dilation)
                    in_length = in_length_stride
                    x = 10*torch.randn((n_batches, in_dim, in_length))

                    conv.set_mode('dilation')
                    y_dilation = conv(x)
                    conv.set_mode('stride')
                    y_strides = conv(x)
                    assert_allclose(y_dilation[:, :, (in_length-1)%subsampl::subsampl].detach(), y_strides.detach(),
                                    atol=1e-5)
