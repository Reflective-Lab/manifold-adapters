> **Archived 2026-07-02** — active milestone tracking moved to Linear (Reflective team).
> This file is kept for historical context only. Do not add new items here.

---
source: mixed
---
# Milestones

> See `~/dev/reflective/stack/bedrock-platform/EPIC.md` for the coarse-grained outcomes these milestones advance.

## Shipped: v1.1.1 — Converge 3.9.1 alignment — 2026-05-17

**Tracks:** Converge 3.9.1

- [x] Bump converge-core / converge-experience / converge-pack /
      converge-provider / converge-storage to 3.9.1.
- [x] First clean `just release-check` run including all five gates.
- [x] Publish to crates.io (v1.1.0 was the prior published version).
- [x] Tag v1.1.1.

## Shipped: v1.1.0 — Adapter family migration — 2026-05-07

- [x] Object-storage, experience-store, and vector adapters live in Manifold.
- [x] LLM adapter family moved from Converge staging and compiling behind `llm-all`.
- [x] Remove LLM adapter definitions, model catalog, live chat examples, and
      live LLM endpoint probes from Converge staging.
- [x] Move search, fetch, feed, embedding, reranking, vector, and OpenAPI/GraphQL
      tool adapters.

## Shipped: v1.0.0 — Adapter Foundation — 2026-05

- [x] Workspace package version is `1.0.0`.
- [x] Tag v1.0.0.

## Open: pull-driven
**Epic:** E9

- [ ] Downstream proof that products register Manifold handles through
      `converge_provider::ChatBackendRegistry`.
- [ ] Baseten backend (`crates/manifold/src/llm/baseten.rs`) for hosting
      stock open models and Reflective-branded fine-tunes. Design captured
      in `kb/Architecture/Baseten Integration.md`. ~30 min implementation
      when picked up; deferred from 2026-05-23 in favor of completing the
      cost-aware routing phase first.
- [ ] Per-domain quality scoring in `ModelMetadata`
      (`domain_quality: HashMap<String, f64>`) + `AgentRequirements.domain`
      so a Reflective-branded fine-tune can be picked preferentially for
      its trained domain without overstating general-purpose quality.
      Deferred until there's a concrete second Reflective fine-tune to
      compare against.
- [ ] `AgentBackend` trait + first implementation
      (`PerplexityDeepResearchBackend`) for long-running server-side
      multi-step agents. Design captured in
      `kb/Architecture/Long-Running Agent Backend.md`. Result type
      (`AgentArtifact`) shape-mirrors `converge_pack::gate::SolverReport`
      so artifacts can be wrapped as Converge evidence at the runtime
      layer. Deferred from 2026-05-23.
- [ ] Streaming chat backends — `StreamingChatBackend` trait is now
      defined and `OpenRouterBackend` implements it (gated on the
      `streaming` feature). Outstanding: implement for `AnthropicBackend`,
      `OpenAiBackend`, `GeminiBackend`, etc. when downstream UI cases
      need real-time token rendering.
