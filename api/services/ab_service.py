"""A/B testing variant assignment."""


def get_variant(user_id: int) -> str:
    """Assign user to variant: even user_ids get 'svd', odd get 'coldstart'."""
    return "svd" if user_id % 2 == 0 else "coldstart"
