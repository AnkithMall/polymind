from polymind.core.model.types import RankedModel


def group_models(
    models: list[RankedModel],
) -> list[list[RankedModel]]:
    """
    Group model variants belonging to the same Hugging Face
    repository.

    A repository may contain multiple GGUF quantizations,
    for example:

        model-Q4_K_M.gguf
        model-Q5_K_M.gguf
        model-Q6_K.gguf

    These should be displayed as one model with multiple
    variants.
    """

    groups: dict[str, list[RankedModel]] = {}

    for model in models:
        key = model.repo_id

        groups.setdefault(
            key,
            [],
        ).append(model)

    # Keep the best-ranked variant first inside each repository.
    for group in groups.values():
        group.sort(
            key=lambda model: model.score,
            reverse=True,
        )

    # Rank repositories by their best variant.
    result = list(groups.values())

    result.sort(
        key=lambda group: group[0].score,
        reverse=True,
    )

    return result
