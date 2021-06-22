"""Calculations for cross-section properties."""
from dataclasses import dataclass
from typing import Collection

import numpy as np


@dataclass(frozen=True)
class GenericSection:
    """Generic section with properties area, centre of gravity, moments of inertia."""

    A: float
    y_c: float
    z_c: float
    I_yy: float
    I_zz: float
    I_yz: float


@dataclass(frozen=True)
class MaterialisedGenericSection(GenericSection):
    """Generic section with properties area, centre of gravity, moments of inertia, and material stiffness."""

    E: float


class RingSection:
    """Circular hollow section with properties area, centre of gravity, Moments of Inertia."""

    def __init__(
        self,
        outer_diameter: float,
        wall_thickness: float,
        y_c: float = 0.0,
        z_c: float = 0.0,
    ):
        """Create a circular hollow section based on outer diameter and wall thickness."""
        inner_diameter = outer_diameter - 2 * wall_thickness
        r_out = outer_diameter / 2
        r_in = inner_diameter / 2
        self.A: float = np.pi * (r_out ** 2 - r_in ** 2)
        self.y_c = y_c
        self.z_c = z_c
        self.I_yy = np.pi / 4 * (r_out ** 4 - r_in ** 4)
        self.I_zz = self.I_yy
        self.I_yz = self.I_yy


def combine_sections(sections: Collection[GenericSection]) -> GenericSection:
    """Calculate the sectional properties of combined sections.

    Args:
        sections (Iterable[GenericSection]): list or tuple of GenericSections

    Returns:
        GenericSection: Combined section
    """
    A = sum([GS.A for GS in sections])
    y_c = sum([GS.A * GS.y_c for GS in sections]) / A
    z_c = sum([GS.A * GS.z_c for GS in sections]) / A
    I_yy_simple_sum = sum([GS.I_yy for GS in sections])
    I_zz_simple_sum = sum([GS.I_zz for GS in sections])
    I_yz_simple_sum = sum([GS.I_yz for GS in sections])
    I_yy_steiner = sum([GS.A * (z_c - GS.z_c) ** 2 for GS in sections])
    I_zz_steiner = sum([GS.A * (y_c - GS.y_c) ** 2 for GS in sections])
    I_yz_steiner = sum([GS.A * (y_c - GS.y_c) * (z_c - GS.z_c) for GS in sections])
    return GenericSection(
        A,
        y_c,
        z_c,
        I_yy_simple_sum + I_yy_steiner,
        I_zz_simple_sum + I_zz_steiner,
        I_yz_simple_sum + I_yz_steiner,
    )


# TODO: Overload for Collection[MaterialisedGenericSection]
def idealised_section(
    sections: Collection[GenericSection],
    youngs_moduli: Collection[float],
    reference_modulus: float,
) -> MaterialisedGenericSection:
    """Calculate an idealised section on basis of a reference Young's modulus.

    Args:
        sections (Collection[GenericSection]): [description]
        youngs_moduli (Collection[float]): [description]
        reference_modulus (float) : [description]

    Returns:
        MaterialisedGenericSection: [description]
    """
    assert len(sections) == len(
        youngs_moduli
    ), "sections and youngs_moduli provided have to have the same length."

    n_sections = len(sections)

    n_s = np.zeros(n_sections)
    A_s = np.zeros(n_sections)
    y_c = np.zeros(n_sections)
    z_c = np.zeros(n_sections)
    I_yy = np.zeros(n_sections)
    I_zz = np.zeros(n_sections)
    I_yz = np.zeros(n_sections)

    for i, (section, youngs_modulus) in enumerate(zip(sections, youngs_moduli)):
        n_s[i] = youngs_modulus / reference_modulus
        A_s[i] = section.A
        y_c[i] = section.y_c
        z_c[i] = section.z_c
        I_yy[i] = section.I_yy
        I_zz[i] = section.I_zz
        I_yz[i] = section.I_yz
    # calculate idealised area
    A_i = A_s * n_s
    A_i_sum = np.sum(A_i)

    # calculate idealised centres of gravity
    y_c_A_i = y_c * A_i
    z_c_A_i = z_c * A_i

    y_c_i = np.sum(y_c_A_i) / A_i_sum
    z_c_i = np.sum(z_c_A_i) / A_i_sum

    I_yy_steiner = A_s * (z_c_i - z_c) ** 2
    I_zz_steiner = A_s * (y_c_i - y_c) ** 2
    I_yz_steiner = A_s * (y_c_i - y_c) * (z_c_i - z_c)
    I_yy_i = np.sum(n_s * (I_yy_steiner + I_yy))
    I_zz_i = np.sum(n_s * (I_zz_steiner + I_zz))
    I_yz_i = np.sum(n_s * (I_yz_steiner + I_yz))

    return MaterialisedGenericSection(
        A_i_sum, y_c_i, z_c_i, I_yy_i, I_zz_i, I_yz_i, reference_modulus
    )
