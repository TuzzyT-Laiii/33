import numba


@numba.njit(cache=True)
def accumulate(t, cpu, cid, machine, request, limit, good, slots_per_container,
               seen_slots, counts, request_hours, limit_hours, one_core_hours, support_hours):
    kept=0
    duplicate=0
    factor=10/3600
    for row in range(len(t)):
        if not good[row]:
            continue
        container=cid[row]
        slot=container*slots_per_container+int(t[row])//10
        byte=slot//8
        bit=1 << (slot%8)
        if seen_slots[byte]&bit:
            duplicate+=1
            continue
        seen_slots[byte] |= bit
        hour=int(t[row])//3600
        host=machine[row]
        value=cpu[row]
        counts[container,hour]+=1
        request_hours[host,hour]+=value*request[row]/10000*factor
        limit_hours[host,hour]+=value*limit[row]/10000*factor
        one_core_hours[host,hour]+=value/100*factor
        support_hours[host,hour]+=factor
        kept+=1
    return kept,duplicate
