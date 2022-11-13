from dataclasses import dataclass
from typing import Generic, TypeVar


Variant = TypeVar("Variant")
Data = TypeVar("Data")


@dataclass
class Enumerant(Generic[Variant, Data]):
    variant: Variant
    data: Data


def get_variant(enumerant: Enumerant[Variant, Data]) -> Variant:
    return enumerant.variant


def get_data(enumerant: Enumerant[Variant, Data]) -> Data:
    return enumerant.data
