from sms_rl.drivers.dolphin import MemoryFlagSpec, MemoryValueSpec, _read_scalar


class DummyMemoryModule:
    def read_byte(self, address: int) -> int:
        return address + 1

    def read_word(self, address: int) -> int:
        return address + 2

    def read_float(self, address: int) -> float:
        return address + 3.5

    def read_double(self, address: int) -> float:
        return address + 4.5


def test_read_scalar_dispatches_by_type() -> None:
    memory = DummyMemoryModule()

    assert _read_scalar(memory, "byte", 10) == 11
    assert _read_scalar(memory, "word", 10) == 12
    assert _read_scalar(memory, "float", 10) == 13.5
    assert _read_scalar(memory, "double", 10) == 14.5


def test_memory_specs_store_pointer_configuration() -> None:
    value = MemoryValueSpec(base_address=0x1234, value_type="float", pointer_offsets=(0x8,))
    flag = MemoryFlagSpec(base_address=0x9999, value_type="byte", expected_value=7)

    assert value.base_address == 0x1234
    assert value.pointer_offsets == (0x8,)
    assert flag.expected_value == 7
