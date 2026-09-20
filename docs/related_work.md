# Related work and spec-pin notes

Logged 2026-09-18 before implementation. Searches: arXiv, ACL-adjacent papers, Alignment Forum, A2A spec v1.0.0.

## Conjunction check

Query: {cross-lingual, multilingual} × {collusion, oversight evasion, monitorability, agent protocol}.

**No paper covers Study D** (agents freely select the language their overseer is weakest in, residual to fluency). Studies A–C also remain uncovered as an A2A-layer benchmark. Proceed.

## Closest papers (not this project)

### Cross-lingual safety (single-agent jailbreaks)

- Yong, Menghini, Bach (2023). *Low-Resource Languages Jailbreak GPT-4*. arXiv:2310.02446. Safety training is English-centric; translating AdvBench into low-resource languages bypasses GPT-4 ~79% vs <1% in English.
- Deng et al. (2023). *Multilingual Jailbreak Challenges in Large Language Models*. arXiv:2310.06474. Unintentional + intentional multilingual jailbreaks; low-resource languages ~3× unsafe rate.

These are user→model jailbreaks, not agent-card routing, A2A transcripts, or capability-normalised monitor slopes.

### Secret collusion / steganography

- Motwani et al. (2024). *Secret Collusion among AI Agents*. NeurIPS 2024 / arXiv:2402.07510. CASE framework. Collusion via steganographic schemes.
- Roger-family and follow-ups on encoded CoT / early steganographic capability (e.g. arXiv:2507.02737).
- *Tool Use Enables Undetectable Steganography in Multi-Agent LLM Systems* (arXiv:2606.28425). Tool-enabled stegosystems.
- *Audit the Whisper* (arXiv:2510.04303). Detecting steganographic collusion.

Gap vs this work: steganography requires a scheme. Speaking Hindi to a weaker Hindi monitor does not.

### Multilingual monitorability (not A2A)

- *The Fragility of Chain-of-Thought Monitoring Across Typologically Diverse Languages* (arXiv:2605.27901). CoT unfaithfulness ~96% across 13 languages; 100% in low-resource languages. Single-agent CoT, not protocol language choice, not CCA/OG.

### Agent protocols

- *Security Threat Modeling for Emerging AI-Agent Protocols* (MCP, A2A, Agora, ANP), arXiv:2602.11327. Collusion is listed as a protocol threat; language is not treated as the channel.
- Industry A2A guides (e.g. Palo Alto) flag poisoned Agent Cards and prompt injection in card text, not script/language bias at routing time.

### Alignment Forum

- Hidden-reasoning taxonomy: “linguistic drift” / language mixing is listed; authors expect monitors to still understand mixed language.
- Stages-of-oversight language-variant task: models switch response language eval vs deployment. That is a conditional-policy test, not emergent selection of the weakest-monitored language under A2A evasion pressure.

## A2A v1.0 field corrections

Pinned: protocol **1.0**, spec https://a2a-protocol.org/v1.0.0/specification/, SDK `a2a-sdk`.

The experiment spec prose still mentions some v0.3 names. Implementation uses v1.0:

| Spec prose | v1.0 actual |
|---|---|
| `/.well-known/agent.json` | `/.well-known/agent-card.json` |
| Top-level `url`, `protocolVersion` | `supportedInterfaces[]` with `url`, `protocolBinding`, `protocolVersion` |
| `supportsAuthenticatedExtendedCard` | `capabilities.extendedAgentCard` |
| `preferredTransport` / `additionalInterfaces` | replaced by `supportedInterfaces` |

Localised fields unchanged: card `name`/`description`; skill `name`/`description`/`tags`/`examples`. Never localise `id`, URLs, MIME types, enums, `protocolVersion`.
