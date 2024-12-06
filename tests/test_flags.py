# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from fastforward.flags import (
    export_mode,
    get_export_mode,
    get_strict_quantization,
    set_export_mode,
    set_strict_quantization,
    strict_quantization,
)


def test_strict_quantization_flag():
    with strict_quantization(True):
        assert get_strict_quantization()
        with strict_quantization(False):
            assert not get_strict_quantization()
            set_strict_quantization(True)
            assert get_strict_quantization()
            set_strict_quantization(False)
            assert not get_strict_quantization()
            with strict_quantization(True):
                assert get_strict_quantization()
            assert not get_strict_quantization()
        assert get_strict_quantization()
    assert get_strict_quantization()


def test_export_mode_flag():
    with export_mode(True):
        assert get_export_mode()
        with export_mode(False):
            assert not get_export_mode()
            set_export_mode(True)
            assert get_export_mode()
            set_export_mode(False)
            assert not get_export_mode()
            with export_mode(True):
                assert get_export_mode()
            assert not get_export_mode()
        assert get_export_mode()
    assert not get_export_mode()
