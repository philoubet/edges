import logging
from collections import defaultdict
from functools import cache, lru_cache
from typing import Optional

from .utils import make_hashable, get_shares


def preprocess_cfs(cf_list, by="consumer"):
    """
    Group CFs by location from either 'consumer', 'supplier', or both.

    :param cf_list: List of characterization factors (CFs)
    :param by: One of 'consumer', 'supplier', or 'both'
    :return: defaultdict of location -> list of CFs
    """
    assert by in {
        "consumer",
        "supplier",
        "both",
    }, "'by' must be 'consumer', 'supplier', or 'both'"

    lookup = defaultdict(list)

    for cf in cf_list:
        consumer_loc = cf.get("consumer", {}).get("location")
        supplier_loc = cf.get("supplier", {}).get("location")

        if by == "consumer":
            if consumer_loc:
                lookup[consumer_loc].append(cf)

        elif by == "supplier":
            if supplier_loc:
                lookup[supplier_loc].append(cf)

        elif by == "both":
            if consumer_loc:
                lookup[consumer_loc].append(cf)
            elif supplier_loc:
                lookup[supplier_loc].append(cf)

    return lookup


def process_cf_list(
    cf_list: list,
    filtered_supplier: dict,
    filtered_consumer: dict,
    supplier_info: dict,
    consumer_info: dict,
    candidate_suppliers: list,
    candidate_consumers: list,
    share: float,
) -> list:
    results = []
    best_score = -1
    best_cf = None

    for cf in cf_list:
        supplier_cf = cf.get("supplier", {})
        consumer_cf = cf.get("consumer", {})
        operator = supplier_cf.get("operator", "equals")

        supplier_match = True
        for k in filtered_supplier:
            if k == "classifications":
                continue
            val = filtered_supplier.get(k, "")
            expected = supplier_cf.get(k, "")

            if k == "location" and candidate_suppliers:
                match = any(
                    match_operator(loc_val, expected, operator)
                    for loc_val in candidate_suppliers
                )
            else:
                match = match_operator(val, expected, operator)

            if not match:
                supplier_match = False
                break

        if not supplier_match:
            continue

        consumer_match = True
        for k in filtered_consumer:
            if k == "classifications":
                continue

            if k not in consumer_cf:
                continue  # Missing field in CF means match anything

            val = filtered_consumer.get(k, "")
            expected = consumer_cf.get(k, "")

            if k == "location" and candidate_consumers:
                match = any(
                    match_operator(loc_val, expected, operator)
                    for loc_val in candidate_consumers
                )
            else:
                match = match_operator(val, expected, operator)

            if not match:
                consumer_match = False
                break

        if not consumer_match:
            continue

        match_score = 0
        cf_class = supplier_cf.get("classifications")
        ds_class = supplier_info.get("classifications")
        if cf_class and ds_class and matches_classifications(cf_class, ds_class):
            match_score += 1

        cf_cons_class = consumer_cf.get("classifications")
        ds_cons_class = consumer_info.get("classifications")
        if (
            cf_cons_class
            and ds_cons_class
            and matches_classifications(cf_cons_class, ds_cons_class)
        ):
            match_score += 1

        if match_score > best_score:
            best_score = match_score
            best_cf = cf

    if best_cf:
        results.append((best_cf, share))

    return results


def matches_classifications(cf_classifications, dataset_classifications):
    """Match CF classification codes to dataset classifications."""
    if isinstance(cf_classifications, dict):
        cf_classifications = [
            (scheme, code)
            for scheme, codes in cf_classifications.items()
            for code in codes
        ]
    elif isinstance(cf_classifications, (list, tuple)):
        if all(
            isinstance(x, tuple) and isinstance(x[1], (list, tuple))
            for x in cf_classifications
        ):
            # Convert from tuple of tuples like (('cpc', ('01.1',)),) -> [('cpc', '01.1')]
            cf_classifications = [
                (scheme, code) for scheme, codes in cf_classifications for code in codes
            ]

    dataset_codes = [
        (scheme, code.split(":")[0].strip()) for scheme, code in dataset_classifications
    ]

    for scheme, code in dataset_codes:
        if any(
            code.startswith(cf_code)
            and scheme.lower().strip() == cf_scheme.lower().strip()
            for cf_scheme, cf_code in cf_classifications
        ):
            return True
    return False


@cache
def match_operator(value: str, target: str, operator: str) -> bool:
    """
    Implements matching for three operator types:
      - "equals": value == target
      - "startswith": value starts with target (if both are strings)
      - "contains": target is contained in value (if both are strings)

    :param value: The flow's value.
    :param target: The lookup's candidate value.
    :param operator: The operator type ("equals", "startswith", "contains").
    :return: True if the condition is met, False otherwise.
    """
    if operator == "equals":
        return value == target
    elif operator == "startswith":
        if isinstance(value, str):
            return value.startswith(target)
        if isinstance(value, tuple):
            return value[0].startswith(target)
    elif operator == "contains":
        return target in value
    return False


def normalize_classification_entries(cf_list: list[dict]) -> list[dict]:

    for cf in cf_list:
        supplier = cf.get("supplier", {})
        classifications = supplier.get("classifications")
        if isinstance(classifications, dict):
            # Normalize from dict
            supplier["classifications"] = tuple(
                (scheme, val)
                for scheme, values in sorted(classifications.items())
                for val in values
            )
        elif isinstance(classifications, list):
            # Already list of (scheme, code), just ensure it's a tuple
            supplier["classifications"] = tuple(classifications)
        elif isinstance(classifications, tuple):
            # Handle legacy format like: (('cpc', ('01.1',)),)
            new_classifications = []
            for scheme, maybe_codes in classifications:
                if isinstance(maybe_codes, (tuple, list)):
                    for code in maybe_codes:
                        new_classifications.append((scheme, code))
                else:
                    new_classifications.append((scheme, maybe_codes))
            supplier["classifications"] = tuple(new_classifications)
    return cf_list


def build_cf_index(
    raw_cfs: list[dict], required_supplier_fields: set
) -> dict[str, dict[tuple, list[dict]]]:
    """
    Build a nested CF index:
        cf_index[consumer_location][supplier_signature] → list of CFs
    """
    index = defaultdict(lambda: defaultdict(list))

    for cf in raw_cfs:
        consumer_loc = cf.get("consumer", {}).get("location", "__ANY__")

        supplier = cf.get("supplier", {})
        sig = tuple(
            sorted((k, supplier[k]) for k in required_supplier_fields if k in supplier)
        )

        index[consumer_loc][sig].append(cf)

    return index


@cache
def cached_match_with_index(flow_to_match_hashable, required_fields_tuple):
    flow_to_match = dict(flow_to_match_hashable)
    required_fields = set(required_fields_tuple)
    return match_with_index(
        flow_to_match,
        cached_match_with_index.index,
        cached_match_with_index.lookup_mapping,
        required_fields,
        cached_match_with_index.reversed_lookup,
    )


def preprocess_flows(flows_list: list, mandatory_fields: set) -> dict:
    """
    Preprocess flows into a lookup dictionary.
    Each flow is keyed by a tuple of selected metadata fields.
    If no fields are present, falls back to using its position as key.

    :param flows_list: List of flows (dicts with metadata + 'position')
    :param mandatory_fields: Fields that must be included in the lookup key.
    :return: A dictionary mapping keys -> list of flow positions
    """
    lookup = {}

    for flow in flows_list:

        def make_value_hashable(v):
            if isinstance(v, list):
                return tuple(v)
            if isinstance(v, dict):
                return tuple(
                    sorted((k, make_value_hashable(val)) for k, val in v.items())
                )
            return v

        # Build a hashable key from mandatory fields (if any are present)
        key_elements = [
            (k, make_value_hashable(flow[k]))
            for k in mandatory_fields
            if k in flow and flow[k] is not None
        ]

        if key_elements:
            key = tuple(sorted(key_elements))
        else:
            # Fallback: use the position as a unique key
            key = (("position", flow["position"]),)

        lookup.setdefault(key, []).append(flow["position"])

    return lookup


def build_index(lookup: dict, required_fields: set) -> dict:
    """
    Build an inverted index from the lookup dictionary.
    The index maps each required field to a dict, whose keys are the values
    from the lookup entries and whose values are lists of tuples:
    (lookup_key, positions), where lookup_key is the original key from lookup.

    :param lookup: The original lookup dictionary.
    :param required_fields: The fields to index.
    :return: A dictionary index.
    """
    index = {field: {} for field in required_fields}
    for key, positions in lookup.items():
        # Each key is assumed to be an iterable of (field, value) pairs.
        for k, v in key:
            if k in required_fields:
                index[k].setdefault(v, []).append((key, positions))
    return index


def match_with_index(
    flow_to_match: dict,
    index: dict,
    lookup_mapping: dict,
    required_fields: set,
    reversed_lookup: dict,
) -> list:
    """
    Match a flow against the lookup using the inverted index.
    Supports "equals", "startswith", and "contains" operators.

    :param flow_to_match: The flow to match.
    :param index: The inverted index produced by build_index().
    :param lookup_mapping: The original lookup dictionary mapping keys to positions.
    :param required_fields: The required fields for matching.
    :return: A list of matching positions.
    """
    operator_value = flow_to_match.get("operator", "equals")
    candidate_keys = None

    if not required_fields:
        # Match all flows if no fields to match
        return [pos for positions in lookup_mapping.values() for pos in positions]

    for field in required_fields:
        match_target = flow_to_match.get(field)
        field_index = index.get(field, {})
        field_candidates = set()

        if operator_value == "equals":
            # Fast direct lookup.
            for candidate in field_index.get(match_target, []):
                candidate_key, _ = candidate
                field_candidates.add(candidate_key)
        else:
            # For "startswith" or "contains", we iterate over all candidate values.
            for candidate_value, candidate_list in field_index.items():
                if match_operator(
                    value=candidate_value, target=match_target, operator=operator_value
                ):
                    for candidate in candidate_list:
                        candidate_key, _ = candidate
                        field_candidates.add(candidate_key)

        # Initialize or intersect candidate sets.
        if candidate_keys is None:
            candidate_keys = field_candidates
        else:
            candidate_keys &= field_candidates

        # Early exit if no candidates remain.
        if not candidate_keys:
            return []

    # Gather positions from the original lookup mapping for all candidate keys.
    matches = []
    for key in candidate_keys:
        matches.extend(lookup_mapping.get(key, []))

    if "classifications" in flow_to_match:
        cf_classifications_by_scheme = flow_to_match["classifications"]

        if isinstance(cf_classifications_by_scheme, tuple):
            cf_classifications_by_scheme = dict(cf_classifications_by_scheme)

        classified_matches = []

        for pos in matches:
            flow = reversed_lookup.get(pos)
            flow = dict(flow)
            if not flow:
                continue
            try:
                dataset_classifications = flow.get("classifications", [])
            except:
                print(flow)
                raise

            if dataset_classifications:
                for scheme, cf_codes in cf_classifications_by_scheme.items():
                    relevant_codes = [
                        code.split(":")[0].strip()
                        for s, code in dataset_classifications
                        if s.lower() == scheme.lower()
                    ]
                    if any(
                        code.startswith(prefix)
                        for prefix in cf_codes
                        for code in relevant_codes
                    ):
                        classified_matches.append(pos)
                        break

        if classified_matches:
            return classified_matches

    return matches


def compute_cf_memoized_factory(
    cf_index, required_supplier_fields, required_consumer_fields, weights, logger
):
    @lru_cache(maxsize=None)
    def compute_cf(s_key, c_key, supplier_candidates, consumer_candidates):
        return compute_average_cf(
            candidate_suppliers=list(supplier_candidates),
            candidate_consumers=list(consumer_candidates),
            supplier_info=dict(s_key),
            consumer_info=dict(c_key),
            weight=weights,
            cf_index=cf_index,
            required_supplier_fields=required_supplier_fields,
            required_consumer_fields=required_consumer_fields,
            logger=logger,
        )

    return compute_cf


def normalize_signature_data(info_dict, required_fields):
    filtered = {k: info_dict[k] for k in required_fields if k in info_dict}

    # Normalize classifications
    if "classifications" in filtered:
        c = filtered["classifications"]
        if isinstance(c, dict):
            # From dict of lists -> tuple of (scheme, code)
            filtered["classifications"] = tuple(
                (scheme, code) for scheme, codes in c.items() for code in codes
            )
        elif isinstance(c, list):
            # Ensure it's a list of 2-tuples
            filtered["classifications"] = tuple(
                (scheme, code) for scheme, code in c if isinstance(scheme, str)
            )
        elif isinstance(c, tuple):
            # Possibly already normalized — validate structure
            if all(isinstance(e, tuple) and len(e) == 2 for e in c):
                filtered["classifications"] = c
            else:
                # Convert from legacy format
                new_classifications = []
                for scheme, maybe_codes in c:
                    if isinstance(maybe_codes, (tuple, list)):
                        for code in maybe_codes:
                            new_classifications.append((scheme, code))
                    else:
                        new_classifications.append((scheme, maybe_codes))
                filtered["classifications"] = tuple(new_classifications)

    return filtered


def group_edges_by_signature(
    edge_list, required_supplier_fields, required_consumer_fields, geo, weights
):
    grouped = defaultdict(list)

    for (
        supplier_idx,
        consumer_idx,
        supplier_info,
        consumer_info,
        candidate_locs,
    ) in edge_list:
        s_filtered = normalize_signature_data(supplier_info, required_supplier_fields)
        c_filtered = normalize_signature_data(consumer_info, required_consumer_fields)

        s_key = make_hashable(s_filtered)
        c_key = make_hashable(c_filtered)

        # Determine candidate location grouping key
        supplier_sub = (
            geo.resolve(supplier_info.get("location"), containing=True)
            if "location" in required_supplier_fields
            else []
        )
        consumer_sub = (
            geo.resolve(consumer_info.get("location"), containing=True)
            if "location" in required_consumer_fields
            else []
        )

        loc_key = (
            tuple(sorted(set([g for g in supplier_sub if g in weights]))),
            tuple(sorted(set([g for g in consumer_sub if g in weights]))),
        )

        grouped[(s_key, c_key, loc_key)].append((supplier_idx, consumer_idx))

    return grouped


def compute_average_cf(
    candidate_suppliers: list,
    candidate_consumers: list,
    supplier_info: dict,
    consumer_info: dict,
    weight: dict,
    cf_index: dict,
    required_supplier_fields: set = None,
    required_consumer_fields: set = None,
    logger=None,
) -> tuple[str | float, Optional[dict]]:
    """
    Compute the weighted average characterization factor (CF) for a given supplier-consumer pair.
    Supports disaggregated regional matching on both supplier and consumer sides.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if not candidate_suppliers and not candidate_consumers:
        return 0, None

    valid_candidates = [
        (loc, weight[loc])
        for loc in set(candidate_suppliers + candidate_consumers)
        if loc in weight and weight[loc] > 0
    ]
    if not valid_candidates:
        return 0, None

    cand_locs, shares = get_shares(tuple(valid_candidates))
    if not shares.any():
        return 0, None

    filtered_supplier = {
        k: supplier_info[k] for k in required_supplier_fields if k in supplier_info
    }
    filtered_consumer = {
        k: consumer_info[k] for k in required_consumer_fields if k in consumer_info
    }

    if "classifications" in filtered_supplier:
        c = filtered_supplier["classifications"]
        if isinstance(c, dict):
            filtered_supplier["classifications"] = tuple(
                (scheme, tuple(sorted(vals))) for scheme, vals in c.items()
            )
        elif isinstance(c, (list, tuple)):
            filtered_supplier["classifications"] = tuple(sorted(c))

    supplier_sig = make_hashable(filtered_supplier)
    matched_cfs = []

    for candidate_location, share in zip(cand_locs, shares):
        loc_cfs_dict = cf_index.get(candidate_location)

        if loc_cfs_dict is None:
            loc_cfs_dict = cf_index.get("__ANY__")

        if not loc_cfs_dict:
            continue

        loc_cfs = loc_cfs_dict.get(supplier_sig, [])

        matched_cfs.extend(
            process_cf_list(
                loc_cfs,
                filtered_supplier,
                filtered_consumer,
                supplier_info,
                consumer_info,
                candidate_suppliers,
                candidate_consumers,
                share,
            )
        )

    if not matched_cfs:
        return 0, None

    expressions = [f"({share:.6f} * ({cf['value']}))" for cf, share in matched_cfs]
    expr = " + ".join(expressions)

    return (expr, matched_cfs[0][0]) if len(matched_cfs) == 1 else (expr, None)
