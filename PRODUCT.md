# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

FastAPI serving Jinja templates, HTMX for interaction, Tailwind CSS, server-sent events for turn-by-turn call updates. Chosen by the user in place of the spec's Next.js frontend. Designed in Figma first, then built.

## Users

- Live demo audience (class, judges): watch on a projector while the team drives. They need to read the verdict, the scam type and the reason from across a room.
- Thai users trying it alone afterwards: paste a suspicious message or upload a call recording and want a clear answer and what to do next.

Both matter equally.

## Product Purpose

Callas checks Thai phone calls and text messages for scams. A call goes through speech-to-text, a speaker split (A/B), call memory and nlp-core (entities, summary, scam gate plus one of five types). Fixed rules then either warn the user or let a local LLM write a short safe reply that TTS speaks. Success: a viewer sees what each stage produced and the exact turn where Callas escalated.

## Positioning

Turn-by-turn call memory: a scam that starts late in a normal-sounding call is still caught, and the UI shows the turn where it flipped. Replies never repeat anything from the call; a leak check swaps any risky LLM reply for a fixed template. All models run locally; call text never leaves the machine.

## Operating Context

- Inputs: a typed/pasted Thai message, or an uploaded call audio file (generated 2-speaker calls).
- Outputs per input: transcript by speaker (calls only), entity highlights (PHONE, BANK_ACCOUNT, OTP, PERSON, ORG, MONEY, URL), summary, verdict (scam / not scam) with type and confidence, and either a playable Thai voice reply or an "escalated to user" state with the reason.
- Per-stage timing is logged; the target is under about 15 s per input.

## Capabilities and Constraints

- UI copy in English; content (messages, transcripts, replies) is Thai. Thai text must render well (Thai-capable font, line height for stacked vowels and tone marks).
- Five scam types: fake authority / call center, romance, investment, phishing / OTP, lottery / prize.
- Escalation is sticky for the rest of the call. The LLM never decides reply vs warn.
- No connection to LINE, Messenger, Instagram or real phone lines. Demo only.
- The classifier is an interim LLM version until WangchanBERTa is trained; confidence may be coarse.

## Brand Commitments

Name: Callas. No logo, palette or type commitments.

## Evidence on Hand

No real users, testimonials or benchmark numbers yet. Do not invent accuracy figures; eval numbers come only from `eval/results/`.

## Product Principles

1. Show the reason, not only the verdict: every warning names the turn and the signal.
2. Never echo sensitive data back; the UI may highlight entities, the voice reply may not speak them.
3. Legible at projector distance and on a phone.
4. Honest about uncertainty: low confidence and "unknown" are shown as such.

## Accessibility & Inclusion

Verdicts must not rely on color alone (icon + label). Audio replies get a text transcript. Keyboard-operable upload and playback.
