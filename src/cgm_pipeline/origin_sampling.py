"""Outcome-blind proportional sampling of an eligible training identity index."""
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
import hashlib
from cgm_pipeline.formal_training import digest


def sample_origins(rows, *, scope, fraction, seed):
    if not 0 < fraction <= 1 or seed < 0:
        raise ValueError('invalid explicit sampling policy')
    checked = scope.check_rows(rows, len(rows), training=True)
    groups = defaultdict(list)
    for row in checked:
        groups[(row['dataset'], row['participant_id'])].append(row)
    count = len(checked)
    total = int((Decimal(str(fraction))*count).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
    if total < len(groups):
        raise ValueError('sample budget is smaller than the number of eligible participants')
    def key(value):
        return hashlib.sha256((str(seed)+'|'+digest(scope.__dict__)+'|'+digest(value)).encode()).hexdigest()
    quotas = {g: total*len(v)/count for g, v in groups.items()}
    # Hamilton's largest-remainder rule defines the unconstrained proportional
    # allocation. The participant-preservation floor is then applied explicitly.
    allocations = {g:int(quotas[g]) for g in groups}
    remainder = total-sum(allocations.values())
    for g in sorted(groups, key=lambda g:(-(quotas[g]-allocations[g]), key(g))):
        if remainder == 0: break
        allocations[g] += 1
        remainder -= 1
    hamilton = dict(allocations)
    allocations = {g:max(1, allocations[g]) for g in groups}
    forced = sum(allocations[g]-hamilton[g] for g in groups)
    # The floor can consume the fixed budget. Remove excess from the most
    # over-allocated strata first; otherwise add to the most under-allocated.
    while sum(allocations.values()) != total:
        direction = 1 if sum(allocations.values()) < total else -1
        eligible = [g for g in groups if (allocations[g] < len(groups[g]) if direction == 1 else allocations[g] > 1)]
        if not eligible:
            raise ValueError('participant minimum cannot be reconciled with sample budget')
        if direction > 0:
            g = min(eligible, key=lambda g:(allocations[g]-quotas[g], key(g)))
        else:
            g = max(eligible, key=lambda g:(allocations[g]-quotas[g], key(g)))
        allocations[g] += direction
    wanted = set()
    strata = []
    for g in sorted(groups):
        selected = sorted(groups[g], key=lambda r:key((g,r['origin_id'])))[:allocations[g]]
        wanted.update(r['origin_id'] for r in selected)
        strata.append(dict(dataset=g[0], participant_id=g[1], eligible=len(groups[g]),
                           sampled=allocations[g], quota=quotas[g]))
    selected = [r for r in checked if r['origin_id'] in wanted]
    return dict(schema_version=1, seed=seed, requested_fraction=fraction, eligible=count,
                sampled=total, participants=len(groups), minimum_allocation=1,
                quota_below_one=sum(q<1 for q in quotas.values()),
                hamilton_zero_strata=sum(v==0 for v in hamilton.values()),
                minimum_forced_initial_additions=forced,
                final_allocations_changed_from_hamilton=sum(allocations[g]!=hamilton[g] for g in groups),
                minimum_allocation_net_origins=sum(allocations.values())-sum(hamilton.values()),
                maximum_absolute_quota_error=max(abs(allocations[g]-quotas[g]) for g in groups),
                allocation_total_variation=.5*sum(abs(allocations[g]/total-len(groups[g])/count) for g in groups),
                full_index_sha256=digest(checked), sampled_index_sha256=digest(selected),
                rule='scope-wide round-half-up count; proportional participant/dataset quotas; minimum one; greatest deficit/excess correction; SHA256 ties and within-stratum origin ranking',
                strata=strata, origin_rows=selected)
