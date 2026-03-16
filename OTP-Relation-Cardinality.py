# ============================================================
#   Cardinality Computation (Bidirectional)
# ============================================================
def compute_relation_cardinalities(relations):
    """
    relations = { (src_type, qualifier, tgt_type) : [(src_id, tgt_id), ...] }
    Returns mapping (A, B) -> "1 : *" etc, and also reverse (B, A) correctly.
    """
    cardinality_map = {}

    for (src_type, qualifier, tgt_type), pairs in relations.items():

        fromA = defaultdict(set)
        fromB = defaultdict(set)

        for a, b in pairs:
            fromA[a].add(b)
            fromB[b].add(a)

        maxA = max((len(v) for v in fromA.values()), default=0)
        maxB = max((len(v) for v in fromB.values()), default=0)

        # Forward cardinality
        if maxA <= 1 and maxB <= 1:
            forward = "1 : 1"
            reverse = "1 : 1"

        elif maxA > 1 and maxB <= 1:
            forward = "1 : *"     # A → B (one A has many B)
            reverse = "* : 1"     # B → A (many B belong to one A)

        elif maxA <= 1 and maxB > 1:
            forward = "* : 1"     # A → B (many A map to one B)
            reverse = "1 : *"     # B → A (one B maps to many A)

        else:
            forward = "* : *"
            reverse = "* : *"

        # Store correct forward
        cardinality_map[(src_type, tgt_type)] = forward

        # Store correct reverse
        cardinality_map[(tgt_type, src_type)] = reverse

    return cardinality_map
