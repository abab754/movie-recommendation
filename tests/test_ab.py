"""Tests for A/B variant assignment."""

from api.services.ab_service import get_variant


def test_even_user_gets_svd():
    assert get_variant(2) == "svd"
    assert get_variant(100) == "svd"
    assert get_variant(0) == "svd"


def test_odd_user_gets_coldstart():
    assert get_variant(1) == "coldstart"
    assert get_variant(999) == "coldstart"


def test_assignment_is_deterministic():
    for user_id in range(50):
        assert get_variant(user_id) == get_variant(user_id)


def test_split_is_roughly_half():
    variants = [get_variant(uid) for uid in range(1000)]
    svd_count = variants.count("svd")
    assert svd_count == 500
