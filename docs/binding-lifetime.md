# Nanobind Lifetime and Ownership Guide

This document defines the binding conventions we use in `pyoperon` to avoid dangling pointers, dangling views, and accidental ownership bugs.

## Core rules

- If C++ stores a non-owning pointer/reference/span to an argument beyond the call, bind with `nb::keep_alive`.
- If Python receives a view into object-owned memory, attach an explicit owner (usually `nb::find(self)`).
- If an API stores a span/view of container input, do not allow implicit conversion from Python temporaries.
- Prefer explicit return-value policy for references/pointers when ownership is not obvious.
- Use `nb::gil_scoped_release` only around long-running C++ work that does not call Python.

## 1) Stored constructor dependencies

When a constructor takes pointers/references and the object stores them non-owning, use `nb::keep_alive<1, N>()` for each stored argument.

```cpp
nb::class_<Foo>(m, "Foo")
    .def(nb::init<A const*, B const*>(),
         nb::keep_alive<1, 2>(),
         nb::keep_alive<1, 3>());
```

Notes:

- `1` is `self` for constructor bindings.
- `N` is the 1-based Python argument index for the dependency.
- Add one `keep_alive` per independently stored dependency.

## 2) Methods that store non-owning inputs

If a method stores a pointer/reference/span to its argument, also tie lifetime with `keep_alive`.

```cpp
.def("SetDependency", &Foo::SetDependency, nb::keep_alive<1, 2>())
```

If input is a container and C++ stores a span into it:

- Prefer bound container types (for example `IndividualCollection`) instead of generic STL conversion from Python lists.
- Add `.noconvert()` where needed to avoid temporary materialization.
- Keep `nb::keep_alive<1, 2>()` so owner outlives the consumer.

## 3) Returning ndarray/views into internal memory

Any `nb::ndarray` that points to memory owned by a C++ object must carry that owner.

```cpp
return nb::ndarray<Scalar const, nb::numpy>(
    ptr, ndim, shape,
    nb::find(self),
    strides
);
```

Do not use `nb::handle()` (null owner) for internal memory views.

Use a `nb::capsule` owner only when memory is heap-allocated specifically for the return value.

## 4) Returning references, spans, and iterators

For non-owning returns, make ownership semantics explicit:

- Return a copy when practical and cheap enough.
- Otherwise tie returned object lifetime to parent (`reference_internal` and/or `keep_alive<0, 1>()` for iterators/views).

Prefer copies for small data or when lifetime semantics are hard to communicate safely.

## 5) GIL usage

Use `nb::call_guard<nb::gil_scoped_release>()` only for long-running C++ operations that do not invoke Python callbacks.

If C++ may call a Python callable, do not release GIL around that call path unless GIL is reacquired correctly.

## 6) Return policy guidance

When binding raw pointers/references:

- Use `nb::rv_policy::reference_internal` when return is owned by `self`.
- Use `nb::rv_policy::reference` when lifetime is managed elsewhere and documented.
- Avoid relying on implicit defaults for non-trivial ownership.

## 7) Review checklist for PRs

- [ ] Any new `nb::init<...*>` or pointer/ref `__init__` audited for `keep_alive`.
- [ ] Any method that stores dependency pointers/refs/spans has `keep_alive`.
- [ ] Any `nb::ndarray` view into object internals sets a valid owner.
- [ ] Any returned iterator/span/reference has explicit safe lifetime semantics.
- [ ] Any container-to-span API avoids temporary conversion pitfalls.
- [ ] Any `gil_scoped_release` usage is safe with callback behavior.
- [ ] Ownership and lifetime intent are obvious from binding code.

## 8) Preferred defaults

When uncertain, prefer in this order:

1. Return copy (simplest and safest).
2. Return view with explicit owner.
3. Return non-owning reference/view only with explicit lifetime tie and clear rationale.

This keeps bindings predictable for Python users and robust against refcount/GC timing.
