from __future__ import annotations

from behavioral_records.controlled_outer import (
    build_llm_requests,
    compile_population,
    fixed_connected_edges,
    initial_population,
    mutate_population,
    select_llm_parents,
)


def test_outer_population_compiles_without_objective() -> None:
    population = initial_population("C1", 0, 12)
    compiled = compile_population("C1", population, outer_seed=0, generation=0)
    assert len(compiled["evidence"]) == 12
    assert len(set(row["semantic_violation_rate"] for row in compiled["hidden_oracle"].values())) > 1
    requests, private = build_llm_requests(
        scenario="C1",
        outer_seed=0,
        generation=0,
        arm="B2-FullTrace-LLM-PBT",
        population=population,
        evidence=compiled["evidence"],
        requirement="A fixed natural-language requirement.",
    )
    assert len(fixed_connected_edges(12)) == 24
    assert len(requests) == len(private) == 48
    assert all("mode" not in row["prompt"] for row in requests)


def test_outer_selection_and_mutation_are_operational() -> None:
    population = initial_population("C2", 1, 12)
    compiled = compile_population("C2", population, outer_seed=1, generation=0)
    requests, private = build_llm_requests(
        scenario="C2",
        outer_seed=1,
        generation=0,
        arm="B1-LLM-PBT",
        population=population,
        evidence=compiled["evidence"],
        requirement="A requirement.",
    )
    responses = [
        {
            "request_id": row["request_id"],
            "response": {
                "winner": "ABSTAIN",
                "confidence": 0.5,
                "evidence_refs": ["insufficient"],
                "reason": "test",
            },
        }
        for row in requests
    ]
    selection = select_llm_parents(
        population=population,
        private_rows=private,
        responses=responses,
        parent_count=3,
    )
    assert len(set(selection["selected_policy_ids"])) == 3
    assert selection["scheduled_graph_connected"]
    children = mutate_population(
        scenario="C2",
        population=population,
        selected_policy_ids=selection["selected_policy_ids"],
        outer_seed=1,
        next_generation=1,
    )
    assert len(children) == 12
    assert all(row["generation"] == 1 for row in children)


def test_pilot_sized_population_supports_four_parent_mutation() -> None:
    population = initial_population("C1", 2, 16)
    assert len(population) == 16
    children = mutate_population(
        scenario="C1",
        population=population,
        selected_policy_ids=[row["policy_id"] for row in population[:4]],
        outer_seed=2,
        next_generation=1,
    )
    assert len(children) == 16
    assert len({row["policy_id"] for row in children}) == 16
