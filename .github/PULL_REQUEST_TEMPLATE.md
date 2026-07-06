## Summary

-

## What Changed

-

## Validation

- [ ] Built locally
- [ ] Relevant tests/examples run

## Lifetime / Ownership Audit (nanobind)

If this PR touches bindings, complete this section using `docs/binding-lifetime.md`.

- [ ] No new non-owning stored dependencies were introduced
- [ ] Any new stored non-owning constructor/method dependencies use `nb::keep_alive`
- [ ] Any new returned ndarray/view into internal memory has explicit owner (`nb::find(self)` or capsule)
- [ ] Any new returned references/spans/iterators have explicit safe lifetime semantics (copy, owner tie, or documented policy)
- [ ] Any new span-from-container APIs avoid temporary-conversion lifetime hazards (`.noconvert()`, bound container, or copy)

### If bindings changed, list concrete lifetime-sensitive spots

- `path/to/file.cpp:line` — what changed, why safe
-

## Notes

-
