#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal tyre model enum for the vehicle simulator tyre dispatcher."""

from enum import Enum


class TyreModelType(Enum):
    """Tyre model backends selectable via core.tyre_model.set_active_tyre_model."""

    PACEJKA_STANDARD = "pacejka_standard"
    PACEJKA_NEW = "pacejka_new"
    PACEJKA_TTC = "pacejka_ttc"
